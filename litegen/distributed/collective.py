# pylint: disable=W0613
from typing import Any, Optional

import torch
from torch import distributed as dist
from torch.distributed import ReduceOp

from .comm_context import CommContext
from .group_initializer import CommMode, ParallelMode

_all_gather_func = dist._all_gather_base if "all_gather_into_tensor" not in dir(dist) else dist.all_gather_into_tensor


def scatter(
    tensor,
    comm_mode: CommMode,
    scatter_list: Optional[list] = None,
    src: int = 0,
    async_op: bool = False,
):
    """
    custom scatter operation.

    Args:
        tensor(Tensor): Output tensor.
        comm_mode (CommMode): Communication mode registered in CommContext.
        scatter_list(list[Tensor]): List of tensors to scatter (default is
            None, must be specified on the source rank).
        src(int): Src rank.
        async_op(bool): Whether this op should be an async op.

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group
    """
    group = CommContext().get_group(comm_mode=comm_mode)
    return dist.scatter(
        tensor=tensor,
        scatter_list=scatter_list,
        src=src,
        group=group,
        async_op=async_op,
    )


def broadcast_object_list(object_list: list, comm_mode: CommMode, src: int = 0):
    """
    Broadcasts python objects based on torch.distributed.broadcast_object_list

    Args:
        object_list (List[Any]): List of input objects to broadcast.
            Each object must be picklable. Only objects on the ``src`` rank will
            be broadcast, but each rank must provide lists of equal sizes.
        src (int): Source rank from which to broadcast ``object_list``.
        comm_mode (CommMode): Communication mode registered in CommContext.

    Returns:
        ``None``
    """
    group = CommContext().get_group(comm_mode=comm_mode)
    return dist.broadcast_object_list(object_list, src=src, group=group)


def all_gather(output_tensor, input_tensor, comm_mode: CommMode, async_op: bool = False):
    """
    Single tensor all gather. Gathers a single tensor from all ranks, and puts them in a single output tensor.

    Args:
        output_tensor (Tensor): Output tensor. It should contain
            correctly-sized tensors to be used for output of the collective.
        input_tensor (Tensor): Tensor to be broadcast from current process.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        async_op (bool, optional): Whether this op should be an async op

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group
    """
    group = CommContext().get_group(comm_mode=comm_mode)
    return _all_gather_func(output_tensor=output_tensor, input_tensor=input_tensor, group=group, async_op=async_op)


def all_reduce(tensor, comm_mode: CommMode, op=ReduceOp.SUM, async_op: bool = False):
    """
    Reduces the tensor data across all machines in such a way that all get
    the final result.

    After the call ``tensor`` is going to be bitwise identical in all processes.

    Complex tensors are supported.

    Args:
        tensor (Tensor): Input and output of the collective. The function
            operates in-place.
        comm_mode (CommMode): Communication mode registered in CommContext.
        op (optional): One of the values from
            ``torch.distributed.ReduceOp``
            enum.  Specifies an operation used for element-wise reductions.
        async_op (bool, optional): Whether this op should be an async op.

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group
    """
    group = CommContext().get_group(comm_mode=comm_mode)
    return dist.all_reduce(tensor=tensor, op=op, group=group, async_op=async_op)


def broadcast(tensor, comm_mode: CommMode, src: int = 0, async_op: bool = False):
    """
    Broadcasts the tensor to the whole group.

    ``tensor`` must have the same number of elements in all processes
    participating in the collective.

    Args:
        tensor (Tensor): Data to be sent if ``src`` is the rank of current
            process, and tensor to be used to save received data otherwise.
        comm_mode (CommMode): Communication mode registered in CommContext.
        src (int): Source rank.
        async_op (bool, optional): Whether this op should be an async op.

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group

    """
    group = CommContext().get_group(comm_mode=comm_mode)
    return dist.broadcast(tensor=tensor, src=src, group=group, async_op=async_op)


def gather(tensor, comm_mode: CommMode, gather_list: Optional[list] = None, dst: int = 0, async_op: bool = False):
    """
    Gathers a list of tensors in a single process.

    Args:
        tensor (Tensor): Input tensor.
        comm_mode (CommMode): Communication mode registered in CommContext.
        gather_list (list[Tensor], optional): List of appropriately-sized
            tensors to use for gathered data (default is None, must be specified
            on the destination rank)
        dst (int, optional): Destination rank (default is 0)
        async_op (bool, optional): Whether this op should be an async op

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group

    """
    group = CommContext().get_group(comm_mode=comm_mode)
    return dist.gather(tensor, gather_list=gather_list, dst=dst, group=group, async_op=async_op)


class _AllToAllFunction(torch.autograd.Function):
    """
    class for all-to-all communication op
    """

    @staticmethod
    def forward(ctx: Any, input_: torch.Tensor, gather_dim: int, scatter_dim: int, group: Any) -> torch.Tensor:
        assert gather_dim != scatter_dim
        assert 0 <= gather_dim < input_.ndim
        assert 0 <= scatter_dim < input_.ndim
        world_size = dist.get_world_size(group)
        assert input_.size(scatter_dim) % world_size == 0

        ctx.gather_dim = gather_dim
        ctx.scatter_dim = scatter_dim
        ctx.group = group

        if world_size == 1:
            return input_

        inputs = [x.contiguous() for x in input_.chunk(world_size, dim=scatter_dim)]
        outputs = [torch.empty_like(x) for x in inputs]
        dist.all_to_all(outputs, inputs, group=group)

        return torch.cat(outputs, dim=gather_dim)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):
        group = ctx.group
        world_size = dist.get_world_size(group)
        gather_dim = ctx.gather_dim
        scatter_dim = ctx.scatter_dim

        if world_size == 1:
            return grad_output, None, None, None

        grad_outputs = [x.contiguous() for x in grad_output.chunk(world_size, dim=gather_dim)]
        grad_inputs = [torch.empty_like(x) for x in grad_outputs]

        dist.all_to_all(grad_inputs, grad_outputs, group=group)

        return torch.cat(grad_inputs, dim=scatter_dim), None, None, None


def all_to_all(input_: torch.Tensor, gather_dim: int, scatter_dim: int, group: Any) -> torch.Tensor:
    return _AllToAllFunction.apply(input_, gather_dim, scatter_dim, group)


def _sp_split(input_: torch.Tensor) -> torch.Tensor:
    sp_size = CommContext().get_world_size(ParallelMode.SEQUENCE_PARALLEL)
    sp_rank = CommContext().get_local_rank(ParallelMode.SEQUENCE_PARALLEL)
    if sp_size == 1:
        return input_
    assert input_.size(1) % sp_size == 0
    return input_.chunk(sp_size, dim=1)[sp_rank].contiguous()


def _sp_scatter(input_: torch.Tensor) -> torch.Tensor:
    sp_group = CommContext().get_group(ParallelMode.SEQUENCE_PARALLEL)
    sp_src = CommContext().get_ranks_in_group(ParallelMode.SEQUENCE_PARALLEL)[0]
    sp_rank = CommContext().get_local_rank(ParallelMode.SEQUENCE_PARALLEL)
    sp_size = CommContext().get_world_size(ParallelMode.SEQUENCE_PARALLEL)

    if sp_size == 1:
        return input_
    assert input_.size(1) % sp_size == 0
    output = torch.empty(
        [x if i != 1 else x // sp_size for i, x in enumerate(input_.size())], dtype=input_.dtype, device=input_.device
    )
    dist.scatter(
        output,
        [x.contiguous() for x in input_.chunk(sp_size, dim=1)] if sp_rank == 0 else None,
        src=sp_src,
        group=sp_group,
    )
    return output


def _sp_gather(input_: torch.Tensor) -> torch.Tensor:
    sp_group = CommContext().get_group(ParallelMode.SEQUENCE_PARALLEL)
    sp_size = CommContext().get_world_size(ParallelMode.SEQUENCE_PARALLEL)

    if sp_size == 1:
        return input_
    output = [torch.empty_like(input_) for _ in range(sp_size)]
    dist.all_gather(output, input_, group=sp_group)
    return torch.cat(output, dim=1)


class _ScatterToSequenceParallelRegion(torch.autograd.Function):
    """
    class for scatter to sequence parallel communication
    """

    @staticmethod
    def forward(ctx: Any, input_: torch.Tensor, rank0_only: bool = True) -> torch.Tensor:
        if rank0_only:
            return _sp_scatter(input_)
        else:
            return _sp_split(input_)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        sp_size = CommContext().get_world_size(ParallelMode.SEQUENCE_PARALLEL)
        return _sp_gather(grad_output / sp_size), None


class _GatherFromSequenceParallelRegion(torch.autograd.Function):
    """
    class for gather op for sequence parallel
    """

    @staticmethod
    def forward(ctx: Any, input_: torch.Tensor, rank0_only: bool = True) -> torch.Tensor:
        ctx.rank0_only = rank0_only
        return _sp_gather(input_)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        sp_size = CommContext().get_world_size(ParallelMode.SEQUENCE_PARALLEL)
        if ctx.rank0_only:
            return _sp_scatter(grad_output) * sp_size, None
        else:
            return _sp_split(grad_output) * sp_size, None


def scatter_to_sequence_parallel_region(input_: torch.Tensor, rank0_only: bool = True) -> torch.Tensor:
    return _ScatterToSequenceParallelRegion.apply(input_, rank0_only)


def gather_from_sequence_parallel_region(input_: torch.Tensor, rank0_only: bool = True) -> torch.Tensor:
    return _GatherFromSequenceParallelRegion.apply(input_, rank0_only)
