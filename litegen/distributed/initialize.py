from .comm_context import CommContext
from .utils.config import Config
from .utils.env import EnvSetting


def initialize_distributed_env(tensor_parallel=1, sequence_parallel=1, backend="nccl", dist_url="env://", timeout=1800):
    CommContext().init_distributed_env(
        world_size=EnvSetting.WORLD_SIZE,
        rank=EnvSetting.RANK,
        backend=backend,
        dist_url=dist_url,
        timeout=timeout,
    )
    parallel_config = Config(
        {
            "world_size": EnvSetting.WORLD_SIZE,
            "rank": EnvSetting.RANK,
            "tp_size": tensor_parallel,
            "sp_size": sequence_parallel,
        }
    )
    CommContext().init_groups(parallel_config)
