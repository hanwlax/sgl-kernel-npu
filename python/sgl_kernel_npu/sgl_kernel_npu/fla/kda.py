from typing import Optional

import torch
import torch.nn.functional as F

from sgl_kernel_npu.fla.l2norm import l2norm_fwd
from sgl_kernel_npu.fla.utils import input_guard


def _repeat_kv_heads_for_kda(
    q: torch.Tensor, k: torch.Tensor, num_value_heads: int
) -> tuple[torch.Tensor, torch.Tensor]:
    num_key_heads = q.shape[-2]
    if num_value_heads == num_key_heads:
        return q, k

    if num_value_heads % num_key_heads != 0:
        raise ValueError(
            "KDA value heads must be a multiple of query/key heads, got "
            f"{num_value_heads=} and {num_key_heads=}."
        )

    repeat = num_value_heads // num_key_heads
    return q.repeat_interleave(repeat, dim=-2), k.repeat_interleave(repeat, dim=-2)


def _activate_kda_gate(
    g: torch.Tensor,
    A_log: Optional[torch.Tensor],
    dt_bias: Optional[torch.Tensor],
    lower_bound: Optional[float],
) -> torch.Tensor:
    g = g.to(torch.float32)
    if A_log is None:
        return g

    num_heads = g.shape[-2]
    head_dim = g.shape[-1]
    A = A_log.reshape(num_heads).to(torch.float32).view(1, 1, num_heads, 1)

    if dt_bias is None:
        x = g
    else:
        bias = dt_bias.reshape(num_heads, head_dim).to(torch.float32)
        x = g + bias.view(1, 1, num_heads, head_dim)

    if lower_bound is None:
        return -torch.exp(A) * F.softplus(x)
    return lower_bound * torch.sigmoid(torch.exp(A) * x)


def _gather_initial_state(
    initial_state: Optional[torch.Tensor],
    initial_state_indices: Optional[torch.Tensor],
    seq_idx: int,
    num_heads: int,
    head_dim: int,
    value_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, Optional[int]]:
    if initial_state is None:
        state = torch.zeros(
            num_heads, head_dim, value_dim, dtype=torch.float32, device=device
        )
        return state, None

    if initial_state_indices is None:
        slot = seq_idx
    else:
        slot = int(initial_state_indices[seq_idx].item())

    if slot < 0:
        state = torch.zeros(
            num_heads, head_dim, value_dim, dtype=torch.float32, device=device
        )
        return state, None

    # SGLang stores KDA states as [slot, H, V, K]. The recurrent update is
    # easier to express as [H, K, V], matching the update formula.
    state = initial_state[slot].transpose(-1, -2).to(torch.float32)
    return state, slot


def _run_kda_sequence(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    scale: float,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    outputs = []
    chunk_states = []
    state = state.contiguous()
    if q.shape[0] == 0:
        empty_out = v.new_empty(0, v.shape[-2], v.shape[-1])
        return empty_out, state, chunk_states

    for token_idx in range(q.shape[0]):
        q_t = q[token_idx].to(torch.float32) * scale
        k_t = k[token_idx].to(torch.float32)
        v_t = v[token_idx].to(torch.float32)
        g_t = g[token_idx].to(torch.float32)
        beta_t = beta[token_idx].to(torch.float32)

        state = state * torch.exp(g_t).unsqueeze(-1)
        v_delta = v_t - torch.einsum("hkv,hk->hv", state, k_t)
        v_delta = v_delta * beta_t.unsqueeze(-1)
        state = state + k_t.unsqueeze(-1) * v_delta.unsqueeze(-2)
        outputs.append(torch.einsum("hkv,hk->hv", state, q_t))
        if (token_idx + 1) % chunk_size == 0 or token_idx == q.shape[0] - 1:
            chunk_states.append(state.transpose(-1, -2).to(v.dtype))

    return torch.stack(outputs, dim=0), state, chunk_states


@input_guard
@torch.compiler.disable
def chunk_kda_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    initial_state_indices: torch.Tensor = None,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    lower_bound: Optional[float] = None,
    output_intermediate_states: bool = False,
    **kwargs,
):
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q, k, and v must have the same dtype.")
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4 or g.dim() != 4:
        raise ValueError("q, k, v, and g must use [B, T, H, D] layout.")
    if beta.dim() != 3:
        raise ValueError("beta must use [B, T, H] layout.")
    if cu_seqlens is not None and q.shape[0] != 1:
        raise ValueError("cu_seqlens mode expects flattened inputs with batch size 1.")

    if scale is None:
        scale = k.shape[-1] ** -0.5

    if use_qk_l2norm_in_kernel:
        q = l2norm_fwd(q.contiguous())
        k = l2norm_fwd(k.contiguous())

    q, k = _repeat_kv_heads_for_kda(q, k, v.shape[-2])
    g = _activate_kda_gate(g, A_log, dt_bias, lower_bound)
    initial_dtype = q.dtype
    chunk_size = kwargs.get("mamba_cache_chunk_size", 64)
    chunk_size = 64 if chunk_size is None else int(chunk_size)
    if chunk_size <= 0:
        raise ValueError(f"mamba_cache_chunk_size must be positive, got {chunk_size}.")

    batch = len(cu_seqlens) - 1 if cu_seqlens is not None else q.shape[0]
    out = torch.empty(
        q.shape[0], q.shape[1], v.shape[-2], v.shape[-1], dtype=v.dtype, device=v.device
    )
    intermediate_states = []

    for seq_idx in range(batch):
        if cu_seqlens is None:
            batch_idx = seq_idx
            start = 0
            end = q.shape[1]
        else:
            batch_idx = 0
            start = int(cu_seqlens[seq_idx].item())
            end = int(cu_seqlens[seq_idx + 1].item())
        if end <= start:
            continue

        state, slot = _gather_initial_state(
            initial_state=initial_state,
            initial_state_indices=initial_state_indices,
            seq_idx=seq_idx,
            num_heads=v.shape[-2],
            head_dim=k.shape[-1],
            value_dim=v.shape[-1],
            device=v.device,
        )

        seq_out, final_state, seq_intermediate_states = _run_kda_sequence(
            q=q[batch_idx, start:end],
            k=k[batch_idx, start:end],
            v=v[batch_idx, start:end],
            g=g[batch_idx, start:end],
            beta=beta[batch_idx, start:end],
            state=state,
            scale=scale,
            chunk_size=chunk_size,
        )
        out[batch_idx, start:end] = seq_out.to(initial_dtype)
        intermediate_states.append(seq_intermediate_states)

        # Match the CUDA chunk_kda contract: the chunk kernel updates the
        # committed state pool in-place and also returns chunk-boundary states
        # for prefix-cache tracking of unaligned sequence lengths.
        if initial_state is not None and slot is not None:
            initial_state[slot] = final_state.transpose(-1, -2).to(initial_state.dtype)

    if output_intermediate_states:
        if cu_seqlens is None:
            if intermediate_states and intermediate_states[0]:
                h = torch.stack(
                    [
                        torch.stack(seq_intermediate_states, dim=0)
                        for seq_intermediate_states in intermediate_states
                    ],
                    dim=0,
                )
            else:
                h = torch.empty(
                    batch,
                    0,
                    v.shape[-2],
                    v.shape[-1],
                    k.shape[-1],
                    dtype=v.dtype,
                    device=v.device,
                )
        elif intermediate_states:
            h = torch.stack(
                [
                    state
                    for seq_intermediate_states in intermediate_states
                    for state in seq_intermediate_states
                ],
                dim=0,
            ).unsqueeze(0)
        else:
            h = torch.empty(
                1,
                0,
                v.shape[-2],
                v.shape[-1],
                k.shape[-1],
                dtype=v.dtype,
                device=v.device,
            )
        return out, h

    return out
