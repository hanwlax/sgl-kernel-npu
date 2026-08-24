from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _kda_target_verify_task(
    pid_batch,
    pid_hv,
    pid_v,
    A_log_ptr,
    dt_bias_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    a_ptr,
    b_ptr,
    initial_state_ptr,
    initial_indices_ptr,
    snapshot_ptr,
    snapshot_indices_ptr,
    out_ptr,
    scale,
    stride_q_token: tl.constexpr,
    stride_q_head: tl.constexpr,
    stride_q_dim: tl.constexpr,
    stride_k_token: tl.constexpr,
    stride_k_head: tl.constexpr,
    stride_k_dim: tl.constexpr,
    stride_v_token: tl.constexpr,
    stride_v_head: tl.constexpr,
    stride_v_dim: tl.constexpr,
    stride_a_token: tl.constexpr,
    stride_a_head: tl.constexpr,
    stride_a_dim: tl.constexpr,
    stride_b_token: tl.constexpr,
    stride_b_head: tl.constexpr,
    initial_stride_0,
    initial_stride_1,
    initial_stride_2,
    initial_stride_3,
    snapshot_stride_0,
    snapshot_stride_1,
    snapshot_stride_2,
    snapshot_stride_3,
    snapshot_stride_4,
    H_Q: tl.constexpr,
    H_K: tl.constexpr,
    H_V: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    STEPS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    GATES_ARE_PREACTIVATED: tl.constexpr,
):
    offset_k = tl.arange(0, BK)
    offset_v = pid_v * BV + tl.arange(0, BV)
    mask_k = offset_k < K
    mask_v = offset_v < V
    mask_state = mask_v[:, None] & mask_k[None, :]

    q_ratio = H_V // H_Q
    k_ratio = H_V // H_K
    q_head = pid_hv // q_ratio
    k_head = pid_hv // k_ratio
    initial_idx = tl.load(initial_indices_ptr + pid_batch).to(tl.int64)
    snapshot_idx = tl.load(snapshot_indices_ptr + pid_batch).to(tl.int64)

    initial_offsets = (
        initial_idx * initial_stride_0
        + pid_hv * initial_stride_1
        + offset_v[:, None] * initial_stride_2
        + offset_k[None, :] * initial_stride_3
    )
    state = tl.load(
        initial_state_ptr + initial_offsets,
        mask=(initial_idx >= 0) & mask_state,
        other=0.0,
    ).to(tl.float32)

    A_log = tl.zeros((), dtype=tl.float32)
    dt_bias = tl.zeros((BK,), dtype=tl.float32)
    if not GATES_ARE_PREACTIVATED:
        A_log = tl.load(A_log_ptr + k_head).to(tl.float32)
        dt_bias = tl.load(
            dt_bias_ptr + k_head * K + offset_k,
            mask=mask_k,
            other=0.0,
        ).to(tl.float32)

    for step in tl.static_range(0, STEPS):
        token = pid_batch * STEPS + step
        q = tl.load(
            q_ptr
            + token * stride_q_token
            + q_head * stride_q_head
            + offset_k * stride_q_dim,
            mask=mask_k,
            other=0.0,
        ).to(tl.float32)
        k = tl.load(
            k_ptr
            + token * stride_k_token
            + k_head * stride_k_head
            + offset_k * stride_k_dim,
            mask=mask_k,
            other=0.0,
        ).to(tl.float32)
        value = tl.load(
            v_ptr
            + token * stride_v_token
            + pid_hv * stride_v_head
            + offset_v * stride_v_dim,
            mask=mask_v,
            other=0.0,
        ).to(tl.float32)
        a = tl.load(
            a_ptr
            + token * stride_a_token
            + k_head * stride_a_head
            + offset_k * stride_a_dim,
            mask=mask_k,
            other=0.0,
        ).to(tl.float32)
        beta_input = tl.load(
            b_ptr + token * stride_b_token + pid_hv * stride_b_head
        ).to(tl.float32)

        q = q / (tl.sqrt(tl.sum(q * q, axis=0)) + 1e-6)
        k = k / (tl.sqrt(tl.sum(k * k, axis=0)) + 1e-6)
        q *= scale

        if GATES_ARE_PREACTIVATED:
            gate = tl.exp(a)
            beta = beta_input
        else:
            gate_input = a + dt_bias
            softplus = tl.where(
                gate_input <= 20.0,
                tl.log(1.0 + tl.exp(gate_input)),
                gate_input,
            )
            gate = tl.exp(-tl.exp(A_log) * softplus)
            beta = 1.0 / (1.0 + tl.exp(-beta_input))

        state *= gate[None, :]
        value -= tl.sum(state * k[None, :], axis=1)
        value *= beta
        state += value[:, None] * k[None, :]
        output = tl.sum(state * q[None, :], axis=1)

        tl.store(
            out_ptr + (token * H_V + pid_hv) * V + offset_v,
            output,
            mask=mask_v,
        )
        snapshot_offsets = (
            snapshot_idx * snapshot_stride_0
            + step * snapshot_stride_1
            + pid_hv * snapshot_stride_2
            + offset_v[:, None] * snapshot_stride_3
            + offset_k[None, :] * snapshot_stride_4
        )
        tl.store(
            snapshot_ptr + snapshot_offsets,
            state,
            mask=(snapshot_idx >= 0) & mask_state,
        )


@triton.jit
def _kda_target_verify_kernel(
    A_log_ptr,
    dt_bias_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    a_ptr,
    b_ptr,
    initial_state_ptr,
    initial_indices_ptr,
    snapshot_ptr,
    snapshot_indices_ptr,
    out_ptr,
    scale,
    stride_q_token: tl.constexpr,
    stride_q_head: tl.constexpr,
    stride_q_dim: tl.constexpr,
    stride_k_token: tl.constexpr,
    stride_k_head: tl.constexpr,
    stride_k_dim: tl.constexpr,
    stride_v_token: tl.constexpr,
    stride_v_head: tl.constexpr,
    stride_v_dim: tl.constexpr,
    stride_a_token: tl.constexpr,
    stride_a_head: tl.constexpr,
    stride_a_dim: tl.constexpr,
    stride_b_token: tl.constexpr,
    stride_b_head: tl.constexpr,
    initial_stride_0,
    initial_stride_1,
    initial_stride_2,
    initial_stride_3,
    snapshot_stride_0,
    snapshot_stride_1,
    snapshot_stride_2,
    snapshot_stride_3,
    snapshot_stride_4,
    H_Q: tl.constexpr,
    H_K: tl.constexpr,
    H_V: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    STEPS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    GATES_ARE_PREACTIVATED: tl.constexpr,
):
    _kda_target_verify_task(
        tl.program_id(0),
        tl.program_id(1),
        tl.program_id(2),
        A_log_ptr,
        dt_bias_ptr,
        q_ptr,
        k_ptr,
        v_ptr,
        a_ptr,
        b_ptr,
        initial_state_ptr,
        initial_indices_ptr,
        snapshot_ptr,
        snapshot_indices_ptr,
        out_ptr,
        scale,
        stride_q_token,
        stride_q_head,
        stride_q_dim,
        stride_k_token,
        stride_k_head,
        stride_k_dim,
        stride_v_token,
        stride_v_head,
        stride_v_dim,
        stride_a_token,
        stride_a_head,
        stride_a_dim,
        stride_b_token,
        stride_b_head,
        initial_stride_0,
        initial_stride_1,
        initial_stride_2,
        initial_stride_3,
        snapshot_stride_0,
        snapshot_stride_1,
        snapshot_stride_2,
        snapshot_stride_3,
        snapshot_stride_4,
        H_Q,
        H_K,
        H_V,
        K,
        V,
        STEPS,
        BK,
        BV,
        GATES_ARE_PREACTIVATED,
    )


@triton.jit
def _kda_target_verify_grid_kernel(
    A_log_ptr,
    dt_bias_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    a_ptr,
    b_ptr,
    initial_state_ptr,
    initial_indices_ptr,
    snapshot_ptr,
    snapshot_indices_ptr,
    out_ptr,
    scale,
    stride_q_token: tl.constexpr,
    stride_q_head: tl.constexpr,
    stride_q_dim: tl.constexpr,
    stride_k_token: tl.constexpr,
    stride_k_head: tl.constexpr,
    stride_k_dim: tl.constexpr,
    stride_v_token: tl.constexpr,
    stride_v_head: tl.constexpr,
    stride_v_dim: tl.constexpr,
    stride_a_token: tl.constexpr,
    stride_a_head: tl.constexpr,
    stride_a_dim: tl.constexpr,
    stride_b_token: tl.constexpr,
    stride_b_head: tl.constexpr,
    initial_stride_0,
    initial_stride_1,
    initial_stride_2,
    initial_stride_3,
    snapshot_stride_0,
    snapshot_stride_1,
    snapshot_stride_2,
    snapshot_stride_3,
    snapshot_stride_4,
    H_Q: tl.constexpr,
    H_K: tl.constexpr,
    H_V: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    STEPS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    GATES_ARE_PREACTIVATED: tl.constexpr,
    NUM_TASKS: tl.constexpr,
    NUM_V_TILES: tl.constexpr,
    NUM_PROGRAMS: tl.constexpr,
):
    pid = tl.program_id(0)
    tasks_per_batch = H_V * NUM_V_TILES
    for task_id in range(pid, NUM_TASKS, NUM_PROGRAMS):
        pid_batch = task_id // tasks_per_batch
        remainder = task_id - pid_batch * tasks_per_batch
        pid_hv = remainder // NUM_V_TILES
        pid_v = remainder - pid_hv * NUM_V_TILES

        _kda_target_verify_task(
            pid_batch,
            pid_hv,
            pid_v,
            A_log_ptr,
            dt_bias_ptr,
            q_ptr,
            k_ptr,
            v_ptr,
            a_ptr,
            b_ptr,
            initial_state_ptr,
            initial_indices_ptr,
            snapshot_ptr,
            snapshot_indices_ptr,
            out_ptr,
            scale,
            stride_q_token,
            stride_q_head,
            stride_q_dim,
            stride_k_token,
            stride_k_head,
            stride_k_dim,
            stride_v_token,
            stride_v_head,
            stride_v_dim,
            stride_a_token,
            stride_a_head,
            stride_a_dim,
            stride_b_token,
            stride_b_head,
            initial_stride_0,
            initial_stride_1,
            initial_stride_2,
            initial_stride_3,
            snapshot_stride_0,
            snapshot_stride_1,
            snapshot_stride_2,
            snapshot_stride_3,
            snapshot_stride_4,
            H_Q,
            H_K,
            H_V,
            K,
            V,
            STEPS,
            BK,
            BV,
            GATES_ARE_PREACTIVATED,
        )


def _get_num_vector_cores(device: torch.device) -> int:
    try:
        device_index = device.index
        if device_index is None:
            device_index = torch.npu.current_device()
        properties = triton.runtime.driver.active.utils.get_device_properties(
            device_index
        )
        return int(properties.get("num_vectorcore", -1))
    except Exception:
        return -1


def kda_target_verify_npu_grid(
    *,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    intermediate_states_buffer: torch.Tensor,
    intermediate_state_indices: torch.Tensor,
    cache_steps: int,
    scale: Optional[float] = None,
    gates_are_preactivated: Optional[bool] = None,
) -> torch.Tensor:
    """KDA fixed-width target verification with per-step state snapshots.

    The persistent and intermediate state layout is the Ascend KDA layout
    ``[..., H_v, V, K]``. The persistent cache is read-only.

    When ``gates_are_preactivated`` is true, ``a`` is the log-decay
    ``-exp(A_log) * softplus(raw_a + dt_bias)`` and ``b`` is already sigmoid
    activated. Both gate tensors may include the SGLang leading singleton.
    When the flag is omitted, a paired leading singleton selects this mode.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must have shape [1, tokens, heads, dim]")
    if q.shape[0] != 1 or k.shape[0] != 1 or v.shape[0] != 1:
        raise ValueError("the leading q, k, and v dimension must be one")
    if cache_steps <= 0 or q.shape[1] % cache_steps != 0:
        raise ValueError("tokens must be divisible by positive cache_steps")
    if q.shape[1] != k.shape[1] or q.shape[1] != v.shape[1]:
        raise ValueError("q, k, and v token dimensions must match")

    batch = q.shape[1] // cache_steps
    h_q, key_dim = q.shape[2:]
    h_k = k.shape[2]
    h_v, value_dim = v.shape[2:]
    a_has_leading_singleton = a.ndim == 4
    b_has_leading_singleton = b.ndim == 3
    if a_has_leading_singleton != b_has_leading_singleton:
        raise ValueError("a and b must use the leading singleton together")
    if gates_are_preactivated is None:
        gates_are_preactivated = a_has_leading_singleton
    if a.ndim == 4:
        if a.shape[0] != 1:
            raise ValueError("4D a must have a leading singleton dimension")
        a = a.squeeze(0)
    if b.ndim == 3:
        if b.shape[0] != 1:
            raise ValueError("3D b must have a leading singleton dimension")
        b = b.squeeze(0)
    if k.shape[3] != key_dim:
        raise ValueError("q and k key dimensions must match")
    if h_v % h_q != 0 or h_v % h_k != 0:
        raise ValueError("value heads must be divisible by q and k heads")
    if tuple(a.shape) != (q.shape[1], h_k, key_dim):
        raise ValueError("a must have shape [tokens, H_k, K]")
    if tuple(b.shape) != (q.shape[1], h_v):
        raise ValueError("b must have shape [tokens, H_v]")
    if not gates_are_preactivated and (
        A_log.numel() != h_k or tuple(dt_bias.shape) != (h_k, key_dim)
    ):
        raise ValueError("A_log and dt_bias shapes do not match KDA heads")
    if initial_state_source.ndim != 4 or tuple(initial_state_source.shape[1:]) != (
        h_v,
        value_dim,
        key_dim,
    ):
        raise ValueError("initial state must have shape [pool, H_v, V, K]")
    if intermediate_states_buffer.ndim != 5 or tuple(
        intermediate_states_buffer.shape[1:]
    ) != (cache_steps, h_v, value_dim, key_dim):
        raise ValueError("intermediate state must have shape [scratch, T, H_v, V, K]")
    if initial_state_indices.ndim != 1 or initial_state_indices.numel() < batch:
        raise ValueError("initial_state_indices must contain at least B entries")
    if (
        intermediate_state_indices.ndim != 1
        or intermediate_state_indices.numel() < batch
    ):
        raise ValueError("intermediate_state_indices must contain at least B entries")

    # SGLang produces q/k/v as views of a packed QKV tensor. The kernel consumes
    # explicit strides so serving can avoid five per-layer materializations.
    tensors = [
        A_log,
        dt_bias,
        q,
        k,
        v,
        a,
        b,
        initial_state_source,
        initial_state_indices,
        intermediate_states_buffer,
        intermediate_state_indices,
    ]
    if any(t.device != q.device for t in tensors):
        raise ValueError("all tensors must be on the same device")
    A_log = A_log.contiguous()
    dt_bias = dt_bias.contiguous()
    initial_state_indices = initial_state_indices.contiguous()
    intermediate_state_indices = intermediate_state_indices.contiguous()
    if initial_state_source.dtype != intermediate_states_buffer.dtype:
        raise ValueError("persistent and intermediate state dtypes must match")
    if initial_state_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("initial_state_indices must be int32 or int64")
    if intermediate_state_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("intermediate_state_indices must be int32 or int64")

    if scale is None:
        scale = key_dim**-0.5
    if scale <= 0:
        raise ValueError("scale must be positive")

    out = torch.empty(
        (1, q.shape[1], h_v, value_dim), dtype=v.dtype, device=v.device
    )
    bk = triton.next_power_of_2(key_dim)
    if bk > 256:
        raise ValueError("key dimensions greater than 256 are unsupported")
    bv = min(64, triton.next_power_of_2(value_dim))
    num_v_tiles = triton.cdiv(value_dim, bv)
    num_tasks = batch * h_v * num_v_tiles
    kernel_args = (
        A_log,
        dt_bias,
        q,
        k,
        v,
        a,
        b,
        initial_state_source,
        initial_state_indices,
        intermediate_states_buffer,
        intermediate_state_indices,
        out,
        scale,
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        a.stride(0),
        a.stride(1),
        a.stride(2),
        b.stride(0),
        b.stride(1),
        initial_state_source.stride(0),
        initial_state_source.stride(1),
        initial_state_source.stride(2),
        initial_state_source.stride(3),
        intermediate_states_buffer.stride(0),
        intermediate_states_buffer.stride(1),
        intermediate_states_buffer.stride(2),
        intermediate_states_buffer.stride(3),
        intermediate_states_buffer.stride(4),
    )
    kernel_meta = {
        "H_Q": h_q,
        "H_K": h_k,
        "H_V": h_v,
        "K": key_dim,
        "V": value_dim,
        "STEPS": cache_steps,
        "BK": bk,
        "BV": bv,
        "GATES_ARE_PREACTIVATED": gates_are_preactivated,
        "num_warps": 1,
        "num_stages": 3,
        "multibuffer": False,
    }
    num_vector_cores = _get_num_vector_cores(q.device)
    use_physical_grid = (
        num_vector_cores > 0
        and num_tasks > 2 * num_vector_cores
    )
    if use_physical_grid:
        _kda_target_verify_grid_kernel[(num_vector_cores,)](
            *kernel_args,
            **kernel_meta,
            NUM_TASKS=num_tasks,
            NUM_V_TILES=num_v_tiles,
            NUM_PROGRAMS=num_vector_cores,
        )
    else:
        grid = (batch, h_v, num_v_tiles)
        _kda_target_verify_kernel[grid](*kernel_args, **kernel_meta)
    return out
