import pytest

torch = pytest.importorskip("torch")


def test_fused_recurrent_wrapper_guards_odd_value_heads_cpu():
    from pathlib import Path

    source = (
        Path(__file__).parents[3]
        / "python"
        / "sgl_kernel_npu"
        / "sgl_kernel_npu"
        / "fla"
        / "fused_sigmoid_gating_recurrent.py"
    ).read_text(encoding="utf-8")

    assert "assert HV % BHV == 0" in source
    assert "initial_state_source.transpose(-1, -2).contiguous()" in source
    assert "initial_state_source.copy_(kernel_state_source.transpose(-1, -2))" in source


def test_activate_kda_gate_safe_gate_matches_kimi_formula_cpu():
    from sgl_kernel_npu.fla.kda import _activate_kda_gate

    g = torch.tensor([[[[0.5, -1.0], [1.0, 0.25]]]], dtype=torch.float32)
    a_log = torch.log(torch.tensor([2.0, 0.5], dtype=torch.float32)).view(1, 1, 2, 1)
    dt_bias = torch.tensor([[0.25, -0.5], [0.1, -0.2]], dtype=torch.float32)
    lower_bound = -5.0

    actual = _activate_kda_gate(
        g=g,
        A_log=a_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
    )
    expected = lower_bound * torch.sigmoid(
        torch.exp(a_log) * (g + dt_bias.view(1, 1, 2, 2))
    )

    torch.testing.assert_close(actual, expected)


def test_repeat_kv_heads_requires_multiple_value_heads_cpu():
    from sgl_kernel_npu.fla.kda import _repeat_kv_heads_for_kda

    q = torch.empty(1, 2, 3, 4)
    k = torch.empty(1, 2, 3, 4)

    with pytest.raises(ValueError, match="multiple of query/key heads"):
        _repeat_kv_heads_for_kda(q, k, num_value_heads=5)


def _requires_npu():
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("KDA fallback contract tests require an available NPU")


def _make_inputs(batch: int, seq_len: int, heads: int, k_dim: int, v_dim: int):
    dtype = torch.float16
    device = "npu"
    q = torch.randn(batch, seq_len, heads, k_dim, dtype=dtype, device=device)
    k = torch.randn(batch, seq_len, heads, k_dim, dtype=dtype, device=device)
    v = torch.randn(batch, seq_len, heads, v_dim, dtype=dtype, device=device)
    g = torch.randn(batch, seq_len, heads, k_dim, dtype=dtype, device=device)
    beta = torch.randn(batch, seq_len, heads, dtype=torch.float32, device=device).sigmoid()
    return q, k, v, g, beta.to(dtype)


def test_chunk_kda_npu_intermediate_states_keep_batch_dim():
    _requires_npu()
    from sgl_kernel_npu.fla.kda import chunk_kda_npu

    batch, seq_len, heads, k_dim, v_dim, chunk_size = 2, 5, 3, 4, 6, 2
    q, k, v, g, beta = _make_inputs(batch, seq_len, heads, k_dim, v_dim)
    initial_state = torch.zeros(batch, heads, v_dim, k_dim, dtype=v.dtype, device=v.device)
    initial_state_indices = torch.arange(batch, dtype=torch.int32, device=v.device)

    out, h = chunk_kda_npu(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        initial_state_indices=initial_state_indices,
        output_intermediate_states=True,
        mamba_cache_chunk_size=chunk_size,
    )

    assert out.shape == (batch, seq_len, heads, v_dim)
    assert h.shape == (batch, 3, heads, v_dim, k_dim)
    assert torch.count_nonzero(initial_state).item() > 0


def test_chunk_kda_npu_varlen_intermediate_states_match_sglang_layout():
    _requires_npu()
    from sgl_kernel_npu.fla.kda import chunk_kda_npu

    seq_lens = [3, 5]
    total_tokens = sum(seq_lens)
    heads, k_dim, v_dim, chunk_size = 3, 4, 6, 2
    q, k, v, g, beta = _make_inputs(1, total_tokens, heads, k_dim, v_dim)
    initial_state = torch.zeros(
        len(seq_lens), heads, v_dim, k_dim, dtype=v.dtype, device=v.device
    )
    initial_state_indices = torch.arange(len(seq_lens), dtype=torch.int32, device=v.device)
    cu_seqlens = torch.tensor([0, 3, total_tokens], dtype=torch.int64, device=v.device)

    out, h = chunk_kda_npu(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        initial_state_indices=initial_state_indices,
        cu_seqlens=cu_seqlens,
        output_intermediate_states=True,
        mamba_cache_chunk_size=chunk_size,
    )

    assert out.shape == (1, total_tokens, heads, v_dim)
    assert h.shape == (1, 5, heads, v_dim, k_dim)
    assert torch.count_nonzero(initial_state).item() > 0
