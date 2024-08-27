import os
import subprocess
from dataclasses import dataclass


def parse_env(env_name):
    """
    parse environment variable.

    Args:
        env_name(str): environment variable name.
    """
    value = os.getenv(env_name)
    if value in ["True", "ON", "1"]:
        return True
    elif value in [None, "False", "OFF", "0"]:
        return False
    else:
        raise NotImplementedError()


@dataclass
class EnvSetting:
    """Environment variable"""

    SLURM = "SLURM_PROCID" in os.environ
    COMM_TIMEOUT = int(os.environ.get("COMM_TIMEOUT", 1800))

    if SLURM:
        # slurm launcher env
        os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29981")
        os.environ["MASTER_ADDR"] = subprocess.getoutput(
            f"scontrol show hostname {os.environ['SLURM_NODELIST']} | head -n1"
        )
        WORLD_SIZE = int(os.environ["SLURM_NTASKS"])
        RANK = int(os.environ["SLURM_PROCID"])
        LOCAL_WORLD_SIZE = int(os.environ["SLURM_NTASKS_PER_NODE"])
    else:
        # torch dist launcher env
        WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
        RANK = int(os.environ.get("RANK", "0"))
        LOCAL_WORLD_SIZE = int(os.environ["SLURM_NTASKS_PER_NODE"])

    if RANK is None or WORLD_SIZE is None:
        raise NotImplementedError(
            "Can't parse RANK and WORLD_SIZE info from torch dist or slurm launcher, we only support these two ways"
            " now."
        )
