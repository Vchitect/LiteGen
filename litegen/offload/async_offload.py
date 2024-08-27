from typing import Optional, Tuple, List, Iterator, Optional, Tuple, Dict, Any
import queue

import torch
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import ActivationWrapper
from torch.autograd.graph import saved_tensors_hooks
from torch.distributed.utils import _replace_by_prefix
import torch.distributed as dist

        
_CHECKPOINT_WRAPPED_MODULE = "_checkpoint_wrapped_module"
_CHECKPOINT_PREFIX = _CHECKPOINT_WRAPPED_MODULE + "."

BLOCK_CNT = None    # used to mark the block index for register the offload wrapper


def get_cnt():
    # Not support pipeline parallel for now.
    global BLOCK_CNT
    if BLOCK_CNT is None:
        BLOCK_CNT = 1
        return BLOCK_CNT-1
    else:
        BLOCK_CNT += 1
        return BLOCK_CNT-1


class SingletonMeta(type):
    """
    single meta class.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance

        return cls._instances[cls]
    

class OffloadItem:

    def __init__(self, act=None, ref_cnt=0, event=None):
        self.act = act
        self.ref_cnt = ref_cnt
        self.event = event
    
    def get_event(self):
        return self.event
    
    def has_event(self):
        return self.event is not None


class OffloadManager(metaclass=SingletonMeta):

    def __init__(self, check=False):
        self.items = {}
        self.check = check

    ############ cpu/gpu tensor interface ##########

    def assert_exist(self, key):
        assert key in self.items
        
    def exist(self, key):
        return key in self.items
        
    def assert_not_exist(self, key):
        assert key not in self.items
        
    def put(self, key, act, event=None):
        if key in self.items:
            # if dist.get_rank() == 0:
            #     print(f"in put, {key=}, act={torch.sum(act)}, device={act.device}, ref_cnt={self.items[key].ref_cnt}, shape={act.shape}, numel={act.numel()}, {event=}")
            self.items[key].act = act
            self.items[key].ref_cnt += 1
            self.items[key].event = event
        else:
            # if dist.get_rank() == 0:
            #     print(f"in put, {key=}, act={torch.sum(act)}, device={act.device}, ref_cnt=0, shape={act.shape}, numel={act.numel()}, {event=}")
            self.items[key] = OffloadItem(act, 1, event)

    def get(self, key):
        self.assert_exist(key)
        item = self.items[key]

        act = item.act
        # if dist.get_rank() == 0:
        #     print(f"in get, {key=}, act={torch.sum(act)}, device={act.device}, ref_cnt={self.items[key].ref_cnt}, shape={act.shape}, numel={act.numel()}, event={item.event}")
        if item.has_event():
            # ensure data movement is done before return to user.
            item.get_event().wait()

        item.ref_cnt -= 1
        if item.ref_cnt == 0:
            self.clear(key)
        return act
    
    def empty(self):
        return len(self.items) == 0
    
    def clear(self, key=None):
        # if dist.get_rank() == 0:
        #     print(f"in clear, {key=}")
        if key is None:
            self.items.clear()
        else:
            self.assert_exist(key)
            self.items.pop(key)

    ############ event interface ##########

    def get_event(self, key):
        self.assert_exist(key)
        item = self.items[key]
        event = item.get_event()
        return event
    
    def has_event(self, key):
        if not self.exist(key):
            return False
        item = self.items[key]
        return item.has_event()


def check_key(tensor):
    """
    Desc:
        check whether to use offload on this tensor.
    """
    return not hasattr(tensor, "block_idx")


# TODO: can add more prefetch strategy.
def get_key(tensor, prefetch=False):
    """
    Desc:
        get the unique key of this tensor to interact with `OffloadManager`.
    """
    if not hasattr(tensor, "inner_idx"):
        if prefetch:
            key = str(tensor.block_idx-1)
        else:
            key = str(tensor.block_idx)
    else:
        if prefetch:
            key = "_".join([str(tensor.block_idx-1), str(tensor.inner_idx)])
        else:
            key = "_".join([str(tensor.block_idx), str(tensor.inner_idx)])

    return key


def copy_key(src, dst):
    """
    Desc:
        Used to transfer the key between two tensor.
    """
    dst.block_idx = src.block_idx
    if hasattr(src, "inner_idx"):
        dst.inner_idx = src.inner_idx


class async_save_on_cpu(saved_tensors_hooks):

    def __init__(self, pin_memory=True, device_type="cuda", prefetch=True, check_fn=check_key, get_fn=get_key, copy_fn=copy_key, h2d_stream=torch.cuda.Stream(), d2h_stream=torch.cuda.Stream()):
        """
            can only support pin_memory = True.
        """
        assert pin_memory == True, "can only run with pin_memory = True"
        device_module = getattr(torch, device_type, torch.cuda)
        

        def pack_to_cpu(tensor):
            if check_fn(tensor):
                return (tensor.device, tensor)

            if not pin_memory:
                return (tensor.device, tensor.cpu())

            cpu_tensor = torch.empty(
                tensor.size(),
                dtype=tensor.dtype,
                layout=tensor.layout,
                pin_memory=(device_module.is_available() and not tensor.is_sparse),
            )
            copy_fn(tensor, cpu_tensor)
            working_stream = torch.cuda.current_stream()
            d2h_stream.wait_stream(working_stream)
            with torch.cuda.stream(d2h_stream):
                cpu_tensor.copy_(tensor, non_blocking=pin_memory)
                tensor.record_stream(d2h_stream)
                # event = torch.cuda.Event()
                # event.record()

            key = get_fn(cpu_tensor)
            # OffloadManager().put(key, cpu_tensor, event)
            OffloadManager().put(key, cpu_tensor)
            return (tensor.device, cpu_tensor)

        def unpack_from_cpu(packed):
            device, tensor = packed
            if check_fn(tensor):
                return tensor
            
            if not pin_memory:
                return tensor.to(device)

            working_stream = torch.cuda.current_stream()
            working_stream.wait_stream(d2h_stream)  # make sure all d2h copy is done before into backward

            cuda_tensor = torch.empty(
                tensor.size(),
                dtype=tensor.dtype,
                layout=tensor.layout,
                device=device
            )
            
            h2d_stream.wait_stream(working_stream)
            with torch.cuda.stream(h2d_stream):
                key = get_fn(tensor)
                event = None
                if OffloadManager().has_event(key): # call before `OffloadManager().get` to ensure key is still exists.
                    event = OffloadManager().get_event(key)
                cpu_tensor = OffloadManager().get(key)

                if cpu_tensor.device != torch.device("cpu"):
                    # wait prefetched tensor.
                    cuda_tensor = cpu_tensor
                    working_stream.wait_stream(h2d_stream)  # TODO: fail to use Event to sync, so use wait_stream instead.
                else:
                    # fallback to blocking h2d
                    cuda_tensor.copy_(cpu_tensor, non_blocking=pin_memory)
                    cuda_tensor.record_stream(h2d_stream)
                    cuda_tensor.record_stream(working_stream)
                    working_stream.wait_stream(h2d_stream)
                    # working_stream.wait_stream(h2d_stream)
                    # torch.cuda.synchronize()

            if prefetch:
                prefetch_key = get_fn(cpu_tensor, prefetch)
                if OffloadManager().exist(prefetch_key):
                    prefetch_cpu_tensor = OffloadManager().get(prefetch_key)
                    
                    # h2d_stream.wait_stream(working_stream)
                    with torch.cuda.stream(h2d_stream):
                        prefetch_cuda_tensor = prefetch_cpu_tensor.to(device, non_blocking=pin_memory)  # TODO: if use multi-device per process, device should be saved for being used here.
                        copy_fn(prefetch_cpu_tensor, prefetch_cuda_tensor)
                        prefetch_cuda_tensor.record_stream(h2d_stream)

                        event = torch.cuda.Event()
                        event.record()
                    OffloadManager().put(prefetch_key, prefetch_cuda_tensor, event)

            return cuda_tensor

        super().__init__(pack_to_cpu, unpack_from_cpu)


class AsyncOffloadWrapper(ActivationWrapper):
    def __init__(self, mod, pin_memory=True, device_type="cuda", prefetch=True, check_fn=check_key, get_fn=get_key, copy_fn=copy_key, h2d_stream=torch.cuda.Stream(), d2h_stream=torch.cuda.Stream()):
        super().__init__(mod)
        self.pin_memory = pin_memory
        self.device_type = device_type
        self.prefetch = prefetch
        self.check_fn = check_fn
        self.get_fn = get_fn
        self.copy_fn = copy_fn
        self.block_idx = get_cnt()
        self.h2d_stream = h2d_stream
        self.d2h_stream = d2h_stream

    def forward(self, x, *args, **kwargs):
        with async_save_on_cpu(pin_memory=self.pin_memory, device_type=self.device_type, prefetch=self.prefetch, 
                               check_fn=self.check_fn, get_fn=self.get_fn, copy_fn=self.copy_fn, h2d_stream=self.h2d_stream, d2h_stream=self.d2h_stream):
            x.block_idx = self.block_idx
            return self._checkpoint_wrapped_module(x, *args, **kwargs)
        
    def named_modules(
        self,
        *args,
        **kwargs,
    ):
        """
        Override :meth:`named_modules()` to intercept module names.

        remove all occurrences of ``_CHECKPOINT_PREFIX``.
        """
        for module_prefix, module in super().named_modules(*args, **kwargs):
            # print(f"bef replace, {module_prefix=}, aft, {module_prefix.replace(_CHECKPOINT_PREFIX, '')=}")
            yield module_prefix.replace(_CHECKPOINT_PREFIX, ""), module
        

def async_offload_wrapper(module: torch.nn.Module, pin_memory=True, device_type="cuda", prefetch=True, check_fn=check_key, get_fn=get_key, copy_fn=copy_key, h2d_stream=torch.cuda.Stream(), d2h_stream=torch.cuda.Stream()) -> torch.nn.Module:
    return AsyncOffloadWrapper(module, pin_memory, device_type, prefetch, check_fn, get_fn, copy_fn, h2d_stream, d2h_stream)

