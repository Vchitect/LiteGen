import torch 
from litegen.distributed.comm_context import CommContext
from litegen.distributed.group_initializer import ParallelMode
from torch import distributed as dist

def encode_compute(x, encode_fn, micro_batch_size=None, vae_scale=0.18215):
    if micro_batch_size is None:
        x = encode_fn(x).latent_dist.sample().mul_(vae_scale)
    else:
        bs = micro_batch_size
        x_out = []
        for i in range(0, x.shape[0], bs):
            x_bs = x[i : i + bs]
            x_bs = encode_fn(x_bs).latent_dist.sample().mul_(vae_scale)
            x_out.append(x_bs)
        x = torch.cat(x_out, dim=0)
    return x

@torch.no_grad()
def vae_encode(x_mb, encode_fn, micro_batch_size=None, vae_scale=0.18215):
    if x_mb.dim() == 5:
        ret = []
        for x in x_mb:
            ret.append(vae_encode(x, encode_fn, micro_batch_size, vae_scale))
        return torch.stack(ret, dim=0)
    elif x_mb.dim() == 4:
        f, c, h, w = x_mb.shape
        if CommContext().is_initialized():
            sp_size = CommContext().get_world_size(ParallelMode.SEQUENCE_PARALLEL)
            sp_rank = CommContext().get_local_rank(ParallelMode.SEQUENCE_PARALLEL)
            sp_group = CommContext().get_group(ParallelMode.SEQUENCE_PARALLEL)
        else:
            sp_size = 1

        if sp_size > 1:
            if f % sp_size != 0:    # padding
                x_mb = torch.cat([x_mb, torch.zeros([sp_size - f % sp_size, c, h, w], device=x_mb.device, dtype=x_mb.dtype)], dim=0)
            x_mb = x_mb.chunk(sp_size, dim=0)[sp_rank]
        
        with torch.no_grad():
            x_mb = encode_compute(x_mb, encode_fn, micro_batch_size, vae_scale)
            
        if sp_size > 1:
            x_mb_global = torch.empty([x_mb.size(0) * sp_size, *x_mb.size()[1:]], dtype=x_mb.dtype, device=x_mb.device)
            dist.all_gather_into_tensor(x_mb_global, x_mb, group=sp_group)
            x_mb = x_mb_global[:f]

        return x_mb
    else:
        raise ValueError(f"Unsupported input shape: {x_mb.shape}")
