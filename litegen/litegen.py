import copy
import functools
import gc
import os
import random
import re
import warnings
from collections import OrderedDict
from functools import partial
from typing import Callable, Union

import numpy as np
import torch
import torch.distributed as dist
from easydict import EasyDict
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    CheckpointWrapper,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from litegen.offload.async_offload import async_offload_wrapper

from .distributed.comm_context import CommContext
from .distributed.group_initializer import CommMode, ParallelMode
from .distributed.initialize import initialize_distributed_env
from .module_convert import ModuleConverter
from .utils.const_registry import ConstRegistry
from .utils.ema import ShardedEMA


class LiteGen:
    """
    class for LiteGen
    """

    def __init__(self, config):
        super().__init__()
        self.config_setting_and_sanity_check(config)

        initialize_distributed_env(sequence_parallel=self.sp_size)

        self.set_random_seed(self.global_seed)
        self.module_converter = ModuleConverter(config)
        if self.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        self.model = None
        self.model_ema = None
        self.resume_step = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def config_setting_and_sanity_check(self, config):
        # experiment and filepath
        self.exp_name = getattr(config, "exp_name", "default_exp")
        if hasattr(config, "checkpoint_dir"):
            self.checkpoint_dir = config.checkpoint_dir
        elif hasattr(config, "results_dir"):
            self.checkpoint_dir = config.results_dir
        else:  # default
            self.checkpoint_dir = os.path.join("results", self.exp_name)
        self.init_from = getattr(config, "init_from", None)
        self.resume_from = getattr(config, "resume_from", None)
        self.auto_resume = getattr(config, "auto_resume", False)

        # precision
        self.allow_tf32 = getattr(config, "allow_tf32", True)
        self.precision = getattr(config, "precision", "bf16")
        assert self.precision in [
            "tf32",
            "fp32",
            "bf16",
            "fp16",
        ], f"Invalid precision '{self.precision}'. Expected one of: ['tf32', 'fp32', 'bf16', 'fp16']."
        self.precision = {"fp32": torch.float, "tf32": torch.float, "bf16": torch.bfloat16, "fp16": torch.float16}[
            self.precision
        ]
        if hasattr(config, "grad_precision"):
            self.grad_precision = config.grad_precision
            assert self.grad_precision in [
                "tf32",
                "fp32",
                "bf16",
                "fp16",
            ], f"Invalid grad_precision '{self.grad_precision}'. Expected one of: ['tf32', 'fp32', 'bf16', 'fp16']."
        else:
            self.grad_precision = None
        self.grad_precision = {"fp32": torch.float, "tf32": torch.float, "bf16": torch.bfloat16, "fp16": torch.float16}[
            self.grad_precision or self.precision
        ]

        # optimizer
        self.learning_rate = getattr(config, "lr", 0.0001)
        self.weight_decay = getattr(config, "weight_decay", 0.0)
        self.fused_optimizer = config.fused_optimizer

        # ddp strategy
        self.zero_degree = config.zero_degree if hasattr(config, "zero_degree") else None
        self.group_zero = config.group_zero if hasattr(config, "group_zero") else False
        if self.group_zero:
            assert self.zero_degree == 3, "Group Zero is only supported for ZeRO3 currently."

        # sequence parallel
        self.sp_size = getattr(config, "sp_size", 1)

        # module convert
        self.fused_layernorm = getattr(config, "fused_layernorm", False)

        # activation optimize
        self.ac_offload = getattr(config, "ac_offload", False)
        self.selective_ratio = getattr(config, "selective_ratio", 1)  # ratio NOT use activation checkpoint

        # training settings
        self.global_seed = getattr(config, "global_seed", 0)
        assert config.num_workers >= 0, f"num_workers should be a non-negative number, but got {config.num_workers}"
        self.num_workers = getattr(config, "num_workers", 4)
        self.pin_memory = getattr(config, "pin_memory", True)
        assert (
            isinstance(config.global_batch_size, int) and config.global_batch_size > 0
        ), f"global_batch_size should be a non-negative number, but got {config.global_batch_size}"
        self.global_batch_size = config.global_batch_size
        assert config.max_steps > 0, f"Invalid max_steps: {config.max_steps}, it should be a positive number."
        self.max_steps = config.max_steps

        # encoder
        encoder_cfg = {
            "fsdp": getattr(config.encoder, "fsdp", True),
            "group": getattr(config.encoder, "group", False),
        }
        self.encoder_cfg = EasyDict(encoder_cfg)

        # ema
        ema_cfg = {
            "enable": getattr(config.ema, "enable", False),
            "sharded": getattr(config.ema, "sharded", True),
            "resume_from": getattr(config.ema, "resume_from", None),
        }
        self.ema_cfg = EasyDict(ema_cfg)

    def set_random_seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.cuda.manual_seed(seed)

    def _setup_sharded_encoder(self, model: nn.Module) -> Union[FSDP, nn.Module]:
        if not self.encoder_cfg.fsdp:
            return model

        from transformers import CLIPTextModelWithProjection, T5PreTrainedModel

        if isinstance(model, CLIPTextModelWithProjection):
            auto_wrap_fn = functools.partial(
                lambda_auto_wrap_policy, lambda_fn=lambda m: m in list(model.text_model.encoder.layers)
            )
        elif isinstance(model, T5PreTrainedModel):
            auto_wrap_fn = None
        else:
            auto_wrap_fn = None
            warnings.warn(f"auto_wrap_fn is not set for {model.__class__.__name__}, use default auto_wrap_fn.")

        pgroup = (
            CommContext().get_intra_node_process_group()
            if self.encoder_cfg.group
            else CommContext().get_group(CommMode.GLOBAL)
        )

        model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_fn,
            process_group=pgroup,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=MixedPrecision(
                param_dtype=next(model.parameters()).dtype,
            ),
            device_id=torch.cuda.current_device(),
            sync_module_states=True,
            limit_all_gathers=True,
            use_orig_params=True,
        )
        torch.cuda.synchronize()
        return model

    def _auto_get_transformers_module_list(self, model):
        if hasattr(model, "get_fsdp_wrap_module_list"):
            return model.get_fsdp_wrap_module_list()
        else:
            # find the module list that
            def find_repeated_module_lists(model):
                repeated_module_lists = []

                from collections import defaultdict

                def _check_module_list(module):
                    module_count = defaultdict(int)
                    for _, child in module.named_children():
                        if isinstance(child, nn.ModuleList):
                            for sub_module in child:
                                module_count[type(sub_module)] += 1
                            if len(module_count) == 1:
                                repeated_module_lists.append(child)
                        else:
                            _check_module_list(child)

                _check_module_list(model)
                return repeated_module_lists

            repeated_module_lists = find_repeated_module_lists(model)
            if len(repeated_module_lists) > 0:
                module_list = [list(modulelist) for modulelist in repeated_module_lists]
                return module_list
            else:
                return None

    def _setup_ddp_strategy_for_model(self, model: nn.Module) -> FSDP:
        # TODO: change intra node group impl
        mp_pp_world_size = 1  # mp_world_size * pp_world_size

        hsdp_hard_condition = (
            CommContext().get_inter_node_process_group() is not None
            and CommContext().get_local_world_size() % mp_pp_world_size == 0
        )
        hsdp_soft_condition = hsdp_hard_condition and mp_pp_world_size <= CommContext().get_local_world_size() // 4

        # select ddp strategy
        fsdp_strategy = None
        if self.zero_degree is not None:
            # ZeRO 1,2,3
            if self.zero_degree == 3:
                if self.group_zero:
                    assert hsdp_hard_condition, "Hard conditions for hsdp are not met."
                    if not hsdp_soft_condition:
                        warnings.warn("Soft conditions for hsdp are not met. It is suggested to use fsdp.")
                    fsdp_strategy = "hsdp"
                else:
                    fsdp_strategy = "fsdp"
            elif self.zero_degree == 2:
                fsdp_strategy = "sdp"
            elif self.zero_degree in (0, 1):
                fsdp_strategy = None
            else:
                raise ValueError(f"Invalid zero degree: {self.zero_degree}")
        else:  # TODO to be removed
            fsdp_strategy = "hsdp" if hsdp_soft_condition else "sdp"
            print(f"Using automatically decided data parallel strategy: {fsdp_strategy}.")

        # setup ddp according to fsdp_strategy
        if fsdp_strategy is None:
            model = DDP(
                model.cuda(), device_ids=[CommContext().get_global_rank() % CommContext().get_local_world_size()]
            )
        else:
            if fsdp_strategy == "hsdp":
                intra_node_dp_pg = None
                for i in range(dist.get_world_size() // CommContext().get_local_world_size()):
                    for j in range(mp_pp_world_size):
                        ranks = list(range(j, CommContext().get_local_world_size(), mp_pp_world_size))
                        ranks = [x + i * CommContext().get_local_world_size() for x in ranks]
                        group = dist.new_group(ranks)
                        if dist.get_rank() in ranks:
                            intra_node_dp_pg = group
                assert intra_node_dp_pg is not None
                process_group = (intra_node_dp_pg, CommContext().get_inter_node_process_group())
            else:
                process_group = CommContext().get_group(CommMode.GLOBAL)

            auto_wrap_module = self._auto_get_transformers_module_list(model)
            if auto_wrap_module is None:
                auto_wrap_policy = None
                warnings.warn("Default auto_wrap_policy is used.")
            else:
                auto_wrap_policy = functools.partial(
                    lambda_auto_wrap_policy,
                    lambda_fn=lambda m: m in auto_wrap_module,
                )
            model = FSDP(
                model,
                auto_wrap_policy=auto_wrap_policy,
                process_group=process_group,
                sharding_strategy={
                    "fsdp": ShardingStrategy.FULL_SHARD,
                    "hsdp": ShardingStrategy.HYBRID_SHARD,
                    "sdp": ShardingStrategy.SHARD_GRAD_OP,
                }[fsdp_strategy],
                mixed_precision=MixedPrecision(
                    param_dtype=self.precision,
                    reduce_dtype=self.grad_precision,
                ),
                device_id=torch.cuda.current_device(),
                sync_module_states=True,
                limit_all_gathers=True,
                use_orig_params=True,
            )
        torch.cuda.synchronize()
        return model

    def _setup_activation_checkpoint_for_model(self, model):
        check_modules_list = self._auto_get_transformers_module_list(model)
        block_idx = 0
        cut_off = 1 / 2
        p = 1 - self.selective_ratio

        def selective_checkpointing(submodule):
            nonlocal block_idx
            nonlocal cut_off
            if submodule in check_modules_list:
                block_idx += 1
                if block_idx * p >= cut_off:
                    cut_off += 1
                    return True
            return False

        # Activation checkpoint
        non_reentrant_wrapper = functools.partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )
        apply_activation_checkpointing(
            model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=selective_checkpointing
        )

        return model

    def _setup_activation_offload_for_model(self, model):
        def offloading_check_fn(submodule):
            if isinstance(submodule, CheckpointWrapper):
                return True
            return False

        # Activation offload
        h2d_stream = torch.cuda.Stream()
        d2h_stream = torch.cuda.Stream()
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=partial(
                async_offload_wrapper, pin_memory=True, prefetch=True, h2d_stream=h2d_stream, d2h_stream=d2h_stream
            ),
            check_fn=offloading_check_fn,
        )

        return model

    def convert_module(self, model: nn.Module) -> nn.Module:
        model = self.module_converter(model)

        return model

    def setup_ema_model(self, model_ema):
        assert self.ema_cfg.enable, "EMA model is not enabled."
        if self.zero_degree == 3:
            model_ema = self._setup_ddp_strategy_for_model(model_ema)
            model_ema = self._setup_activation_checkpoint_for_model(model_ema)
            if self.ac_offload:
                model_ema = self._setup_activation_offload_for_model(model_ema)
        elif self.ema_cfg.sharded:
            model_ema = ShardedEMA(self.model, group=None)
        else:
            pass

        return model_ema

    def _load_model_checkpoint(self, model):
        load_from = None
        is_resume = False
        if self.resume_from:
            assert self.auto_resume is False, "Error, auto_resume should be False when assigned resume in config."
            load_from = self.resume_from
            is_resume = True
            print(f"Resume checkpoint from the specified path: {load_from}")
        elif self.auto_resume:

            def load_latest_checkpoint(checkpoint_dir):
                checkpoint_pattern = re.compile(rf"{self.exp_name}_step(\d+)\.pth")

                max_step = -1
                latest_checkpoint = None

                for filename in os.listdir(checkpoint_dir):
                    match = checkpoint_pattern.match(filename)
                    if match:
                        step = int(match.group(1))
                        if step > max_step:
                            max_step = step
                            latest_checkpoint = filename

                if latest_checkpoint is not None:
                    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
                    return checkpoint_path
                else:
                    return None

            load_from = load_latest_checkpoint(self.checkpoint_dir)

            if load_from is not None:
                is_resume = True
                print(f"Auto resuming from: {load_from}")
            else:
                print(
                    "No checkpoint found by auto resuming mode, attempting to initialize from config.init_from:"
                    f" {self.init_from}"
                )
                load_from = self.init_from
                is_resume = False
        elif self.init_from:
            load_from = self.init_from
            is_resume = False
            print(f"Initialize from the specified path: {load_from}")

        if load_from is not None:
            model = self._load_model(model, "", load_from, is_resume=is_resume)
        else:
            print("No checkpoint specified for loading.")

        return model

    def _setup_sequence_parallel(self, model):
        from .distributed.collective import (
            gather_from_sequence_parallel_region,
            scatter_to_sequence_parallel_region,
        )

        assert self.sp_size >= 1
        # setup forward hook for the first and last block
        transformers_modules = self._auto_get_transformers_module_list(model)
        sp_size = CommContext().get_world_size(ParallelMode.SEQUENCE_PARALLEL)

        import inspect

        first_module_args_name = list(inspect.signature(transformers_modules[0].forward).parameters.keys())
        first_module_divisible_args = []
        for args_name in first_module_args_name:
            if "hidden_states" in args_name:
                first_module_divisible_args.append(True)
            else:
                first_module_divisible_args.append(False)

        def before_first_block(module, args):  # pylint: disable=W0613
            # TODO to consider the kwargs that contains the divisible arguments.
            new_args = []
            global divisible_args_shape_checker  # pylint: disable=W0601
            divisible_args_shape_checker = {}
            for i, is_divisible in enumerate(first_module_divisible_args):
                if is_divisible:
                    hidden_states = args[i]
                    eff_seq_len = hidden_states.size(1)
                    eff_seq_len_varname = first_module_args_name[i] + "_eff_seq_len"
                    globals()[eff_seq_len_varname] = eff_seq_len
                    if hidden_states.size(1) % sp_size != 0:
                        hidden_states = torch.cat(
                            [
                                hidden_states,
                                torch.zeros(
                                    [hidden_states.size(0), sp_size - eff_seq_len % sp_size, hidden_states.size(2)],
                                    dtype=hidden_states.dtype,
                                    device=hidden_states.device,
                                ),
                            ],
                            dim=1,
                        )
                    hidden_states = scatter_to_sequence_parallel_region(hidden_states, rank0_only=False)
                    new_args.append(hidden_states)
                    divisible_args_shape_checker[hidden_states.shape] = eff_seq_len
                else:
                    new_args.append(args[i])
            return tuple(new_args)

        def after_last_block(module, args, output):  # pylint: disable=W0613
            global divisible_args_shape_checker  # pylint: disable=W0602
            new_output = []
            for i in range(len(output)):
                if isinstance(output[i], torch.Tensor):
                    hidden_states = output[i]
                    if hidden_states.shape in divisible_args_shape_checker:
                        eff_seq_len = divisible_args_shape_checker[hidden_states.shape]
                        hidden_states = gather_from_sequence_parallel_region(hidden_states, rank0_only=True)
                        hidden_states = hidden_states[:, :eff_seq_len, :].contiguous()
                        new_output.append(hidden_states)
                else:
                    new_output.append(output[i])

            return tuple(new_output)

        def before_each_attention(module, args, kwargs):  # pylint: disable=W0613
            for i, is_divisible in enumerate(first_module_divisible_args):
                if is_divisible:
                    eff_seq_len_varname = first_module_args_name[i] + "_eff_seq_len"
                    eff_seq_len = globals()[eff_seq_len_varname]
                    kwargs[eff_seq_len_varname] = eff_seq_len

            return args, kwargs

        # register hook for transformer block
        transformers_modules[0].register_forward_pre_hook(before_first_block)
        transformers_modules[-1].register_forward_hook(after_last_block)

        find_attn_flag = False
        for transformer_module in transformers_modules:
            for _, module in transformer_module.named_modules():
                if isinstance(module, ConstRegistry().support_attention):
                    module.register_forward_pre_hook(before_each_attention, with_kwargs=True)
                    find_attn_flag = True

        assert find_attn_flag, "Error: No supported attention module found in the specified transformer modules."

        return model

    def _initialize_module(self, model: nn.Module):
        """
        initialize and setup for a module.
        """
        assert isinstance(model, nn.Module)
        model_requires_grad = False
        for param in model.parameters():
            if param.requires_grad:
                model_requires_grad = True
                break
        if model_requires_grad:
            assert (
                self.model is None
            ), "Only one module that contains trainable parameters is allowed, but got multiple."
            # setup sequence parallel
            model = self.convert_module(model)

            if self.sp_size >= 1:
                model = self._setup_sequence_parallel(model)

            if self.ema_cfg.enable:
                model_ema = copy.deepcopy(model)
            else:
                model_ema = None

            # load checkpoint
            model = self._load_model_checkpoint(model)
            # setup module
            model = self._setup_ddp_strategy_for_model(model)
            model = self._setup_activation_checkpoint_for_model(model)
            if self.ac_offload:
                model = self._setup_activation_offload_for_model(model)
            self.model = model

            # setup ema
            if self.ema_cfg.enable:
                if self.auto_resume:
                    assert (
                        not self.ema_cfg.resume_from
                    ), "Error: 'resume_from' of ema should not be set when 'auto_resume' is enabled."
                load_from = getattr(self.ema_cfg, "resume_from", None)
                if load_from is None and self.auto_resume:

                    def load_latest_checkpoint(checkpoint_dir):
                        checkpoint_pattern = re.compile(rf"{self.exp_name}_step(\d+)\.pth")

                        max_step = -1
                        latest_checkpoint = None

                        for filename in os.listdir(checkpoint_dir):
                            match = checkpoint_pattern.match(filename)
                            if match:
                                step = int(match.group(1))
                                if step > max_step:
                                    max_step = step
                                    latest_checkpoint = filename

                        if latest_checkpoint is not None:
                            checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
                            return checkpoint_path
                        else:
                            return None

                    model_checkpoint_filename = load_latest_checkpoint(self.checkpoint_dir)
                    model_ckpt_fp_wo_ext, _ = os.path.splitext(model_checkpoint_filename)
                    ema_filename = f"{model_ckpt_fp_wo_ext}.ema.pth"
                    load_from = ema_filename
                if load_from is not None:
                    model_ema = self._load_ema(
                        model_ema,
                        filepath=load_from,
                        strict=False,
                    )

                self.model_ema = self.setup_ema_model(model_ema)
                if load_from is None:
                    self.update_ema(self.model, decay=0)
        else:
            # treat as the encoder if the model does not contain trainable parameters
            model = self._setup_sharded_encoder(model)

        return model

    def _load_model(self, model, output_folder, filename, is_resume=False, strict=True):
        resume_step = torch.tensor(0, dtype=torch.long, device=self.device)
        if CommContext().get_global_rank() == 0:
            state_dict = torch.load(os.path.join(output_folder, filename), map_location="cpu")
            if "step" in state_dict.keys():
                if is_resume:
                    resume_step = torch.tensor(state_dict["step"], dtype=torch.long, device=self.device)
                state_dict.pop("step")
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
            print("Model initialization result:")
            print(f"  Missing keys: {missing_keys}")
            print(f"  Unexpected keys: {unexpected_keys}")
        dist.broadcast(resume_step, src=0)
        self.resume_step = resume_step.item()
        dist.barrier()
        return model

    def _load_ema(self, model_ema, filepath, strict=True):
        if CommContext().get_global_rank() == 0:
            missing_keys, unexpected_keys = model_ema.load_state_dict(
                torch.load(filepath, map_location="cpu"), strict=strict
            )
            print("Model EMA initialization result:")
            print(f"  Missing keys: {missing_keys}")
            print(f"  Unexpected keys: {unexpected_keys}")
        dist.barrier()
        return model_ema

    def save_model(self, output_folder=None, filename=None, step=None):
        state_dict = get_model_state_dict(
            self.model,
            options=StateDictOptions(
                full_state_dict=True,
                cpu_offload=True,
            ),
        )
        if CommContext().get_global_rank() == 0:
            if step is not None:
                state_dict["step"] = step
            if output_folder is None:
                output_folder = self.checkpoint_dir
            if filename is None:
                filename = f"{self.exp_name}_step{step}.pth" if step is not None else f"{self.exp_name}.pth"
            else:
                if step is not None:
                    warnings.warn(
                        "Both 'filename' and 'step' are provided. 'step' will be ignored, and the provided 'filename'"
                        " will be used. Consider removing 'step' if it's unnecessary, or use save_model(step=step) to"
                        " automatically set the checkpoint filename and save the model."
                    )

            os.makedirs(output_folder, exist_ok=True)
            torch.save(state_dict, os.path.join(output_folder, filename))
        del state_dict

    def save_ema(self, output_folder=None, filename=None, step=None):
        state_dict = get_model_state_dict(
            self.model_ema,
            options=StateDictOptions(
                full_state_dict=True,
                cpu_offload=True,
            ),
        )
        if output_folder is None:
            output_folder = self.checkpoint_dir
        if filename is None:
            if step is None:
                filename = f"{self.exp_name}.ema.pth"
            else:
                filename = f"{self.exp_name}_step{step}.ema.pth"

        if CommContext().get_global_rank() == 0:
            os.makedirs(output_folder, exist_ok=True)
            torch.save(state_dict, os.path.join(output_folder, filename))
        del state_dict
        dist.barrier()

    def save_optimizer(self, output_folder=None, filename=None, step=None):
        state_dict = get_optimizer_state_dict(
            self.model,
            self.optimizer,
            options=StateDictOptions(
                full_state_dict=True,
                cpu_offload=True,
            ),
        )
        if output_folder is None:
            output_folder = self.checkpoint_dir
        if filename is None:
            if step is None:
                filename = f"{self.exp_name}.optim_state.pth"
            else:
                filename = f"{self.exp_name}_step{step}.optim_state.pth"

        if CommContext().get_global_rank() == 0:
            torch.save(state_dict, os.path.join(output_folder, filename))

        del state_dict
        dist.barrier()

    def _load_optimizer(self, optimizer, output_folder, filename):
        state_dict = torch.load(os.path.join(output_folder, filename), map_location="cpu")

        optimizer.load_state_dict(
            state_dict,
        )
        dist.barrier()
        return optimizer

    def load_optimizer_state(self, optimizer):
        def get_optimizer_filename(model_checkpoint_filename):
            model_ckpt_fp_wo_ext, _ = os.path.splitext(model_checkpoint_filename)
            optimizer_filename = f"{model_ckpt_fp_wo_ext}.optim_state.pth"

            return optimizer_filename

        load_from = None
        if self.resume_from:
            assert self.auto_resume is False, "Error, auto_resume should be False when assigned resume in config."
            load_from = get_optimizer_filename(self.resume_from)
        elif self.auto_resume:

            def load_latest_checkpoint(checkpoint_dir):
                checkpoint_pattern = re.compile(rf"{self.exp_name}_step(\d+)\.pth")

                max_step = -1
                latest_checkpoint = None

                for filename in os.listdir(checkpoint_dir):
                    match = checkpoint_pattern.match(filename)
                    if match:
                        step = int(match.group(1))
                        if step > max_step:
                            max_step = step
                            latest_checkpoint = filename

                if latest_checkpoint is not None:
                    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
                    return checkpoint_path
                else:
                    return None

            load_model_from = load_latest_checkpoint(self.checkpoint_dir)
            if load_model_from is None:
                print(
                    "No model checkpoint found by auto resuming mode, attempting to initialize"
                    f" from config.init_from: {self.init_from}, and the optimizer state"
                    " will not be resumed."
                )
                load_from = None
            else:
                load_from = get_optimizer_filename(load_model_from)

        if load_from is not None:
            print(f"Resume the optimizer state from {load_from}")
            optimizer = self._load_optimizer(optimizer, "", load_from)
        else:
            print("No optimizer state checkpoint for loading.")

        return optimizer

    @torch.no_grad()
    def update_ema(self, model=None, decay=0.9999):
        if model is not None:
            assert model is self.model, "Error, the given model is not correspond to the configuried ema model."

        assert self.ema_cfg.enable, "EMA model is not enabled."
        if self.zero_degree == 3 or not self.ema_cfg.sharded:
            ema_params = OrderedDict(self.model_ema.named_parameters())
            model_params = OrderedDict(model.named_parameters())
            assert set(ema_params.keys()) == set(model_params.keys())

            for name, param in model_params.items():
                ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)
        elif self.ema_cfg.sharded:
            self.model_ema.update(model, decay, only_trainable=True)

    def _initialize_optimizer(self, optimizer):
        assert self.model is not None, "Please initialize model first."
        if self.zero_degree == 1:
            assert isinstance(
                self.model, DDP
            ), "ZeroRedundancyOptimizer only supports DDP wrapped model when using zero 1."

            def _optimizer_to_zero(optimizer):
                optimizer_cls = type(optimizer)
                from torch.distributed.optim import ZeroRedundancyOptimizer

                if optimizer_cls in ConstRegistry().support_optimizers:
                    optimizer_defaults = optimizer.defaults
                    del optimizer
                    zero_optimizer = ZeroRedundancyOptimizer(
                        self.model.parameters(), optimizer_class=optimizer_cls, **optimizer_defaults
                    )
                else:
                    raise NotImplementedError(
                        f"Optimizer {optimizer_cls} is not supported currently. Supported optimizers are"
                        f" {ConstRegistry().support_optimizers}"
                    )
                return zero_optimizer

            optimizer = _optimizer_to_zero(optimizer)
        else:

            def _recreate_optimizer(optimizer):
                optimizer_cls = type(optimizer)
                optimizer_defaults = optimizer.defaults

                del optimizer
                new_optimizer = optimizer_cls(self.model.parameters(), **optimizer_defaults)

                return new_optimizer

            optimizer = _recreate_optimizer(optimizer)

        # load optimizer state dict
        optimizer = self.load_optimizer_state(optimizer)

        for param_group in optimizer.param_groups:
            param_group["lr"] = self.learning_rate
            param_group["weight_decay"] = self.weight_decay

        self.optimizer = optimizer
        return optimizer

    def _initialize_dataset(self, obj):
        assert isinstance(
            obj, ConstRegistry().support_datasets
        ), f"only support dataset with type {ConstRegistry().support_datasets}, but got {type(obj)}"

        def get_train_sampler(dataset, rank, world_size, global_batch_size, max_steps, resume_step, seed):
            sample_indices = torch.empty([max_steps * global_batch_size // world_size], dtype=torch.long)
            epoch_id, fill_ptr, offs = 0, 0, 0

            while fill_ptr < sample_indices.size(0):
                g = torch.Generator()
                g.manual_seed(seed + epoch_id)
                epoch_sample_indices = torch.randperm(len(dataset), generator=g)
                epoch_id += 1
                epoch_sample_indices = epoch_sample_indices[(rank + offs) % world_size :: world_size]
                offs = (offs + world_size - len(dataset) % world_size) % world_size
                epoch_sample_indices = epoch_sample_indices[: sample_indices.size(0) - fill_ptr]
                sample_indices[fill_ptr : fill_ptr + epoch_sample_indices.size(0)] = epoch_sample_indices
                fill_ptr += epoch_sample_indices.size(0)

            return sample_indices[resume_step * global_batch_size // world_size :].tolist()

        def default_collate_fn(samples):
            image = [x[0] for x in samples]
            caps = [x[1] for x in samples]
            return image, caps

        sampler = get_train_sampler(
            obj,
            CommContext().get_local_rank(ParallelMode.DATA_PARALLEL),
            CommContext().get_world_size(ParallelMode.DATA_PARALLEL),
            self.global_batch_size,
            self.max_steps,
            self.resume_step,
            self.global_seed,
        )
        local_batch_size = self.global_batch_size // CommContext().get_world_size(ParallelMode.DATA_PARALLEL)

        collate_fn = getattr(obj, "collate_fn", default_collate_fn)
        dataloader = DataLoader(
            obj,
            batch_size=local_batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )
        return dataloader

    def _initialize_func_compile(self, func):
        func = torch.compile(func)
        return func

    def initialize(self, *args):
        results = []
        # initialize the module firstly
        for obj in args:
            if isinstance(obj, nn.Module):
                obj = self._initialize_module(obj)
                results.append(obj)
            else:
                results.append(None)

        # initialize others
        i = 0
        for obj in args:
            if isinstance(obj, nn.Module):
                i += 1
                continue
            elif isinstance(obj, torch.optim.Optimizer):
                obj = self._initialize_optimizer(obj)
            elif isinstance(obj, ConstRegistry().support_datasets):
                obj = self._initialize_dataset(obj)
            elif isinstance(obj, Callable):
                obj = self._initialize_func_compile(obj)
            else:
                raise NotImplementedError(f"type {type(obj)} not supported.")

            assert results[i] is None
            results[i] = obj
            i += 1

        gc.collect()
        torch.cuda.empty_cache()
        return results if len(results) > 1 else results[0]
