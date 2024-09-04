from abc import ABC, abstractmethod

from torch import distributed as dist


class CommMode:
    GLOBAL = "GLOBAL"
    INTRA_NODE = "INTRA_NODE"
    INTER_NODE = "INTER_NODE"


class ParallelMode(CommMode):
    DATA_PARALLEL = "DATA_PARALLEL"
    TENSOR_PARALLEL = "TENSOR_PARALLEL"
    SEQUENCE_PARALLEL = "SEQUENCE_PARALLEL"


class ProcessGroupInitializer(ABC):
    """Base class to initialize process group"""

    def __init__(self, rank, world_size) -> None:
        self.rank = rank
        self.world_size = world_size

        self.local_rank = None
        self.ranks_in_group = None
        self.process_group = None
        self.group_world_size = None
        self.mode = None

    @abstractmethod
    def init_dist_group(self):
        pass

    def _new_and_update_group_info(self, ranks, use_cpu=False):
        backend = "gloo" if use_cpu else "nccl"
        group = dist.new_group(ranks, backend=backend)

        if self.rank in ranks:
            self.local_rank = ranks.index(self.rank)
            self.group_world_size = len(ranks)
            self.process_group = group
            self.ranks_in_group = ranks
        else:
            self.group_world_size = len(ranks)


class DPGroupInitializer(ProcessGroupInitializer):
    """data parallel group initializer"""

    def __init__(self, rank, world_size, data_parallel_size) -> None:
        super().__init__(rank, world_size)
        self.data_parallel_size = data_parallel_size
        self.process_num_between_dp_rank = world_size // data_parallel_size
        self.mode = ParallelMode.DATA_PARALLEL

    def init_dist_group(self):
        for j in range(self.process_num_between_dp_rank):
            ranks = [i * self.process_num_between_dp_rank + j for i in range(self.data_parallel_size)]
            self._new_and_update_group_info(ranks)
        # ranks = [i * self.process_num_between_dp_rank for i in range(self.data_parallel_size)]
        # self._new_and_update_group_info(ranks)

        return (
            self.local_rank,
            self.group_world_size,
            self.process_group,
            self.ranks_in_group,
            self.mode,
        )


class TPGroupInitializer(ProcessGroupInitializer):
    """tensor parallel group initializer"""

    def __init__(self, rank, world_size, tensor_parallel_size) -> None:
        super().__init__(rank, world_size)
        self.tensor_parallel_size = tensor_parallel_size
        self.tensor_parallel_group_num = world_size // tensor_parallel_size
        self.mode = ParallelMode.TENSOR_PARALLEL

    def init_dist_group(self):
        for i in range(self.tensor_parallel_group_num):
            ranks = [i * self.tensor_parallel_size + j for j in range(self.tensor_parallel_size)]
            self._new_and_update_group_info(ranks)

        return (
            self.local_rank,
            self.group_world_size,
            self.process_group,
            self.ranks_in_group,
            self.mode,
        )


class SPGroupInitializer(ProcessGroupInitializer):
    """sequence parallel group initializer"""

    def __init__(self, rank, world_size, sequence_parallel_size) -> None:
        super().__init__(rank, world_size)
        self.sequence_parallel_size = sequence_parallel_size
        self.sequence_parallel_group_num = world_size // sequence_parallel_size
        self.mode = ParallelMode.SEQUENCE_PARALLEL

    def init_dist_group(self):
        for i in range(self.sequence_parallel_group_num):
            ranks = [(i * self.sequence_parallel_size + j) for j in range(self.sequence_parallel_size)]
            self._new_and_update_group_info(ranks)

        return (
            self.local_rank,
            self.group_world_size,
            self.process_group,
            self.ranks_in_group,
            self.mode,
        )


class IntraNodeGroupInitializer(ProcessGroupInitializer):
    """intra node group initializer"""

    def __init__(self, rank, world_size, local_world_size) -> None:
        super().__init__(rank, world_size)
        self.local_world_size = local_world_size
        self.node_count = world_size // local_world_size
        self.mode = CommMode.INTRA_NODE

    def init_dist_group(self):
        for i in range(self.node_count):
            ranks = list(range(i * self.local_world_size, (i + 1) * self.local_world_size))
            self._new_and_update_group_info(ranks)

        return (
            self.local_rank,
            self.group_world_size,
            self.process_group,
            self.ranks_in_group,
            self.mode,
        )


class InterNodeGroupInitializer(ProcessGroupInitializer):
    """inter node group initializer"""

    def __init__(self, rank, world_size, local_world_size) -> None:
        super().__init__(rank, world_size)
        self.local_world_size = local_world_size
        self.node_count = world_size // local_world_size
        self.mode = CommMode.INTER_NODE

    def init_dist_group(self):
        for i in range(self.local_world_size):
            ranks = [i + j * self.local_world_size for j in range(self.node_count)]
            self._new_and_update_group_info(ranks)

        return (
            self.local_rank,
            self.group_world_size,
            self.process_group,
            self.ranks_in_group,
            self.mode,
        )
