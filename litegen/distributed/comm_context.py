import datetime
import gc

import torch
from torch import distributed as dist

from .utils import _RANK_STR, _WORLD_SIZE_STR, Config, EnvSetting, SingletonMeta

from .group_initializer import (
    CommMode,
    DPGroupInitializer,
    TPGroupInitializer,
    SPGroupInitializer,
    IntraNodeGroupInitializer,
    InterNodeGroupInitializer,
)


class CommContext(metaclass=SingletonMeta):
    """
    Communication context for torch distribution.
    """

    def __init__(self) -> None:
        self._local_ranks = {}
        self._global_ranks = {}
        self._world_sizes = {}
        self._ranks_in_group = {}
        self._node_cuda_num = torch.cuda.device_count()
        # Build communication groups
        self._groups = {}

        self._tensor_parallel_size = 1
        self._data_parallel_size = 1
        self._sequence_parallel_size = 1

    def is_initialized(self):
        return self._groups.get(CommMode.GLOBAL) is not None

    def init_distributed_env(
        self,
        world_size=None,
        rank=None,
        backend="nccl",
        dist_url="env://",
        timeout=1800,
    ):
        """
        init torch dist process group env.

        Args:
            world_size(int): global world size.
            rank(int): global rank.
            backend(str): `nccl` default.
            dist_url(str): init method of process group.
            timeout(int): torch dist process group init timeout.
        """
        dist.init_process_group(
            backend=backend,
            init_method=dist_url,
            world_size=world_size,
            rank=rank,
            timeout=datetime.timedelta(0, timeout),
        )
        torch.distributed.barrier()
        torch.cuda.set_device(f"cuda:{rank%torch.cuda.device_count()}")
        ranks = list(range(world_size))
        # Register global group
        self._register_group(rank, world_size, dist.GroupMember.WORLD, ranks, CommMode.GLOBAL)
        self._global_ranks[CommMode.GLOBAL] = rank

    def _register_group(self, local_rank, world_size, process_group, ranks_in_group, mode):
        self._local_ranks[mode] = local_rank
        self._world_sizes[mode] = world_size
        self._groups[mode] = process_group
        self._ranks_in_group[mode] = ranks_in_group

    def get_group(self, comm_mode: CommMode):
        return self._groups[comm_mode]

    def get_world_size(self, comm_mode: CommMode):
        return self._world_sizes[comm_mode]

    def get_global_rank(self):
        return self._global_ranks[CommMode.GLOBAL]

    def get_local_rank(self, comm_mode: CommMode):
        return self._local_ranks[comm_mode]

    def get_ranks_in_group(self, comm_mode: CommMode):
        return self._ranks_in_group[comm_mode]
    
    def get_local_world_size(self):
        """
        return the world size in one node
        """
        return EnvSetting.LOCAL_WORLD_SIZE

    def get_intra_node_process_group(self):
        return self.get_group(CommMode.INTRA_NODE)
    
    def get_inter_node_process_group(self):
        return self.get_group(CommMode.INTER_NODE)

    def is_main_rank(self):
        return self.get_global_rank() == 0
        
        

    def init_groups(self, config: Config):
        """
        register all parallel groups used in nerf.

        Args:
            tensor_parallel_size(int): tensor parallel group world size.
        """
        rank = self.get_global_rank()
        world_size = self.get_world_size(CommMode.GLOBAL)
        if "tp_size" in config:
            self._tensor_parallel_size = config.tp_size
        if "sp_size" in config:
            self._sequence_parallel_size = config.sp_size
        assert world_size % (self._tensor_parallel_size * self._sequence_parallel_size) == 0
        self._data_parallel_size = world_size // self._tensor_parallel_size // self._sequence_parallel_size

        assert self._tensor_parallel_size == 1 or self._sequence_parallel_size == 1, "TP and SP can't be used together currently."

        initializers = []
        initializers.append(DPGroupInitializer(rank, world_size, self._data_parallel_size))
        initializers.append(TPGroupInitializer(rank, world_size, self._tensor_parallel_size))
        initializers.append(SPGroupInitializer(rank, world_size, self._sequence_parallel_size))
        initializers.append(IntraNodeGroupInitializer(rank, world_size, self.get_local_world_size()))
        initializers.append(InterNodeGroupInitializer(rank, world_size, self.get_local_world_size()))

        for initializer in initializers:
            group_info_to_register = initializer.init_dist_group()
            self._register_group(*group_info_to_register)

    def destroy(self):
        """
        destroy torch dist env and clear all parallel groups used in nerf.
        """
        if not self.is_initialized():
            return

        dist.barrier()
        dist.destroy_process_group()
        self._local_ranks.clear()
        self._global_ranks.clear()
        self._world_sizes.clear()
        self._ranks_in_group.clear()
        self._groups.clear()

        self._tensor_parallel_size = 1
        self._data_parallel_size = 1
        self._sequence_parallel_size = 1
        gc.collect()
