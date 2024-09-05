# pylint: disable=E1102, W1113

from typing import Optional

import torch
from einops import rearrange
from torch.nn import functional as F

from litegen.distributed.collective import all_to_all
from litegen.distributed.comm_context import CommContext
from litegen.distributed.group_initializer import ParallelMode
from models.vid_sd3.sparse_attention import SparseAttnProcessor


class SparseAttnProcessorSP(SparseAttnProcessor):
    # sequence parallel version of sparse attention processor

    # TODO: add two seq length for video and text
    def __call__(
        self,
        attn,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
        full_seqlen: Optional[int] = None,
        hidden_states_eff_seq_len: Optional[int] = None,
        encoder_hidden_states_eff_seq_len: Optional[int] = None,
        Frame: Optional[int] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        # residual = hidden_states

        eff_seq_len = hidden_states_eff_seq_len
        eff_seq_len_encoder = encoder_hidden_states_eff_seq_len

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        context_input_ndim = encoder_hidden_states.ndim
        if context_input_ndim == 4:
            batch_size, channel, height, width = encoder_hidden_states.shape
            encoder_hidden_states = encoder_hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size = encoder_hidden_states.shape[0]

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        # `context` projections.
        encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

        # sequence parallel start all to all
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        batchsize = query.shape[0] // Frame

        query = query.view(batch_size, -1, attn.heads, head_dim)
        key = key.view(batch_size, -1, attn.heads, head_dim)
        value = value.view(batch_size, -1, attn.heads, head_dim)
        encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(batch_size, -1, attn.heads, head_dim)
        encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(batch_size, -1, attn.heads, head_dim)
        encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(batch_size, -1, attn.heads, head_dim)

        query, key = query.to(value.dtype), key.to(value.dtype)
        sp_size = CommContext().get_world_size(ParallelMode.SEQUENCE_PARALLEL)
        sp_group = CommContext().get_group(ParallelMode.SEQUENCE_PARALLEL)

        query = all_to_all(query, gather_dim=1, scatter_dim=2, group=sp_group)
        key = all_to_all(key, gather_dim=1, scatter_dim=2, group=sp_group)
        value = all_to_all(value, gather_dim=1, scatter_dim=2, group=sp_group)

        if eff_seq_len % sp_size != 0:
            query = query[:, :eff_seq_len].contiguous()
            key = key[:, :eff_seq_len].contiguous()
            value = value[:, :eff_seq_len].contiguous()

        encoder_hidden_states_query_proj = all_to_all(
            encoder_hidden_states_query_proj, gather_dim=1, scatter_dim=2, group=sp_group
        )
        encoder_hidden_states_key_proj = all_to_all(
            encoder_hidden_states_key_proj, gather_dim=1, scatter_dim=2, group=sp_group
        )
        encoder_hidden_states_value_proj = all_to_all(
            encoder_hidden_states_value_proj, gather_dim=1, scatter_dim=2, group=sp_group
        )

        if eff_seq_len_encoder % sp_size != 0:
            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj[:, :eff_seq_len_encoder].contiguous()
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj[:, :eff_seq_len_encoder].contiguous()
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj[:, :eff_seq_len_encoder].contiguous()

        # attention
        xq_gather = torch.cat([query, encoder_hidden_states_query_proj], dim=1)
        xk_gather = torch.cat([key, encoder_hidden_states_key_proj], dim=1)
        xv_gather = torch.cat([value, encoder_hidden_states_value_proj], dim=1)

        # temporal
        query_spatial, key_spatial, value_spatial = xq_gather.clone(), xk_gather.clone(), xv_gather.clone()

        xq_gather_temporal = rearrange(xq_gather, "(B T) S H C -> (B S) T H C", T=Frame, B=batchsize)
        xk_gather_temporal = rearrange(xk_gather, "(B T) S H C -> (B S) T H C", T=Frame, B=batchsize)
        xv_gather_temporal = rearrange(xv_gather, "(B T) S H C -> (B S) T H C", T=Frame, B=batchsize)

        freqs_cis = freqs_cis[: xq_gather_temporal.shape[1], :]
        xq_gather_temporal, xk_gather_temporal = self.apply_rotary_emb(
            xq_gather_temporal, xk_gather_temporal, freqs_cis=freqs_cis
        )

        query_spatial = query_spatial.transpose(1, 2)
        key_spatial = key_spatial.transpose(1, 2)
        value_spatial = value_spatial.transpose(1, 2)

        xq_gather_temporal = xq_gather_temporal.transpose(1, 2)
        xk_gather_temporal = xk_gather_temporal.transpose(1, 2)
        xv_gather_temporal = xv_gather_temporal.transpose(1, 2)

        # batch_size_temp = xv_gather_temporal.shape[0]
        hidden_states_temp = hidden_states_temp = F.scaled_dot_product_attention(
            xq_gather_temporal, xk_gather_temporal, xv_gather_temporal, dropout_p=0.0, is_causal=False
        )

        hidden_states_temp = hidden_states_temp.transpose(1, 2)
        hidden_states_temp = hidden_states_temp.to(query.dtype)
        hidden_states_temp = rearrange(hidden_states_temp, "(B S) T H C -> (B T) S H C", T=Frame, B=batchsize)
        #######

        hidden_states = hidden_states = F.scaled_dot_product_attention(
            query_spatial, key_spatial, value_spatial, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = hidden_states.to(query.dtype)

        if Frame == 1:
            hidden_states_temp = hidden_states_temp * 0
        hidden_states = hidden_states + hidden_states_temp

        # Split the attention outputs.
        hidden_states, encoder_hidden_states = (
            hidden_states[:, :eff_seq_len],
            hidden_states[:, eff_seq_len:],
        )

        if eff_seq_len % sp_size != 0:
            hidden_states = torch.cat(
                [
                    hidden_states,
                    torch.zeros(
                        [
                            hidden_states.size(0),
                            sp_size - eff_seq_len % sp_size,
                            hidden_states.size(2),
                            hidden_states.size(3),
                        ],
                        dtype=hidden_states.dtype,
                        device=hidden_states.device,
                    ),
                ],
                dim=1,
            )

        if eff_seq_len_encoder % sp_size != 0:
            encoder_hidden_states = torch.cat(
                [
                    encoder_hidden_states,
                    torch.zeros(
                        [
                            encoder_hidden_states.size(0),
                            sp_size - eff_seq_len_encoder % sp_size,
                            encoder_hidden_states.size(2),
                            encoder_hidden_states.size(3),
                        ],
                        dtype=encoder_hidden_states.dtype,
                        device=encoder_hidden_states.device,
                    ),
                ],
                dim=1,
            )

        hidden_states = all_to_all(hidden_states, gather_dim=2, scatter_dim=1, group=sp_group)
        encoder_hidden_states = all_to_all(encoder_hidden_states, gather_dim=2, scatter_dim=1, group=sp_group)

        hidden_states = hidden_states.reshape(batch_size, -1, attn.heads * head_dim)
        encoder_hidden_states = encoder_hidden_states.reshape(batch_size, -1, attn.heads * head_dim)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        if not attn.context_pre_only:
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if context_input_ndim == 4:
            encoder_hidden_states = encoder_hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        return hidden_states, encoder_hidden_states
