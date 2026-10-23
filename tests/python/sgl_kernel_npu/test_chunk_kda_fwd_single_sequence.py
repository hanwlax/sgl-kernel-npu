"""Single-sequence varlen/dense equivalence, including public intermediates."""

import pytest
import sgl_kernel_npu  # noqa: F401
import torch
import torch_npu  # noqa: F401
from utils import require_npu_op

pytestmark = require_npu_op("chunk_kda_fwd")
chunk_kda_fwd = torch.ops.npu.chunk_kda_fwd


@pytest.fixture(autouse=True)
def require_a3():
    name = torch.npu.get_device_name(torch.npu.current_device())
    if not name.startswith("Ascend910_93"):
        pytest.skip("The single-sequence dense specialization targets A3.")


def make_inputs(tokens, heads=2, sequences=1, gate_dtype=torch.float32):
    torch.manual_seed(42)
    shape = (1, tokens, heads, 128)
    q = torch.nn.functional.normalize(torch.randn(shape), dim=-1).bfloat16()
    k = torch.nn.functional.normalize(torch.randn(shape), dim=-1).bfloat16()
    v = torch.randn(shape).bfloat16()
    g = (-torch.rand(shape) * 0.08 - 0.001).to(gate_dtype)
    beta = torch.sigmoid(torch.randn(1, tokens, heads)).to(gate_dtype)
    state = torch.randn(sequences, heads, 128, 128) * 0.01
    return [x.npu() for x in (q, k, v, g, beta, state)]


def run(
    inputs,
    *,
    cu=None,
    chunk_size=64,
    layout="BSND",
    state_v_first=True,
    intermediates=True,
    initial=True,
    chunk_indices=None,
    raw_gate=False
):
    q, k, v, g, beta, state = inputs
    heads = v.shape[2]
    if layout == "BNSD":
        q, k, v, g = [x.transpose(1, 2) for x in (q, k, v, g)]
        beta = beta.transpose(1, 2)
    elif layout == "TND":
        q, k, v, g, beta = [x.squeeze(0) for x in (q, k, v, g, beta)]
    elif layout == "NTD":
        q, k, v, g = [x.squeeze(0).transpose(0, 1) for x in (q, k, v, g)]
        beta = beta.squeeze(0).transpose(0, 1)
    if not state_v_first:
        state = state.transpose(-1, -2).contiguous()
    return chunk_kda_fwd(
        q,
        k,
        v,
        g,
        beta,
        initial_state=state if initial else None,
        a_log=torch.zeros(heads, device="npu") if raw_gate else None,
        dt_bias=torch.zeros(heads * 128, device="npu") if raw_gate else None,
        cu_seqlens=cu,
        chunk_indices=chunk_indices,
        scale=128**-0.5,
        chunk_size=chunk_size,
        layout=layout,
        state_v_first=state_v_first,
        use_gate_in_kernel=raw_gate,
        safe_gate=raw_gate,
        output_gk=intermediates,
        output_w=intermediates,
        output_u=intermediates,
        output_qg=intermediates,
        output_kg=intermediates,
        output_v_new=intermediates,
        output_h=intermediates,
    )


def assert_exact(actual, expected):
    assert len(actual) == len(expected) == 11
    for a, b in zip(actual, expected):
        if b is None:
            assert a is None
        else:
            assert a.dtype == b.dtype
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            assert torch.equal(
                a.cpu().contiguous().view(torch.uint8),
                b.cpu().contiguous().view(torch.uint8),
            )


@pytest.mark.parametrize("tokens", [1, 63, 64, 65, 127, 129])
@pytest.mark.parametrize("chunk_size", [64, 128])
def test_single_sequence_tail(tokens, chunk_size):
    inputs = make_inputs(tokens)
    cu = torch.tensor([0, tokens], device="npu", dtype=torch.int64)
    assert_exact(
        run(inputs, cu=cu, chunk_size=chunk_size), run(inputs, chunk_size=chunk_size)
    )


@pytest.mark.parametrize("layout", ["BSND", "BNSD", "TND", "NTD"])
@pytest.mark.parametrize("state_v_first", [False, True])
@pytest.mark.parametrize("gate_dtype", [torch.bfloat16, torch.float32])
def test_single_sequence_layout_and_dtype(layout, state_v_first, gate_dtype):
    inputs = make_inputs(129, gate_dtype=gate_dtype)
    cu = torch.tensor([0, 129], device="npu", dtype=torch.int64)
    kwargs = dict(layout=layout, state_v_first=state_v_first)
    assert_exact(run(inputs, cu=cu, **kwargs), run(inputs, **kwargs))


@pytest.mark.parametrize("initial", [False, True])
def test_single_sequence_raw_gate(initial):
    inputs = make_inputs(129)
    cu = torch.tensor([0, 129], device="npu", dtype=torch.int64)
    kwargs = dict(raw_gate=True, initial=initial)
    assert_exact(run(inputs, cu=cu, **kwargs), run(inputs, **kwargs))


@pytest.mark.parametrize("tokens", [16383, 16384])
def test_kimi_attention_tp16_prefill(tokens):
    inputs = make_inputs(tokens, heads=6)
    cu = torch.tensor([0, tokens], device="npu", dtype=torch.int64)
    assert_exact(
        run(inputs, cu=cu, intermediates=False), run(inputs, intermediates=False)
    )


def test_single_sequence_state_carry_and_metadata_reuse():
    inputs = make_inputs(129)
    cu = torch.tensor([0, 129], device="npu", dtype=torch.int64)
    dense_inputs = list(inputs)
    for _ in range(3):
        actual = run(inputs, cu=cu)
        expected = run(dense_inputs)
        assert_exact(actual, expected)
        inputs[-1] = actual[1]
        dense_inputs[-1] = expected[1]
    # Reusing the same metadata storage must not bypass fresh boundary checks.
    cu[-1] = 128
    with pytest.raises(RuntimeError, match="last element"):
        run(inputs, cu=cu)


def test_explicit_chunk_indices():
    inputs = make_inputs(129)
    cu = torch.tensor([0, 129], device="npu", dtype=torch.int64)
    indices = torch.tensor([[0, 0], [0, 1], [0, 2]], device="npu", dtype=torch.int64)
    assert_exact(run(inputs, cu=cu, chunk_indices=indices), run(inputs))


@pytest.mark.parametrize(
    "boundaries, message",
    [
        ([1, 64], "must be 0"),
        ([0, 63], "last element"),
        ([0, 65, 64], "nondecreasing"),
        ([0], "at least"),
    ],
)
def test_invalid_boundaries(boundaries, message):
    inputs = make_inputs(64)
    cu = torch.tensor(boundaries, device="npu", dtype=torch.int64)
    with pytest.raises(RuntimeError, match=message):
        run(inputs, cu=cu)


def test_invalid_boundary_dtype():
    inputs = make_inputs(64)
    cu = torch.tensor([0, 64], device="npu", dtype=torch.int32)
    with pytest.raises(RuntimeError, match="must be int64"):
        run(inputs, cu=cu)


def test_multiple_sequences_keep_independent_states():
    inputs = make_inputs(129, sequences=2)
    cu = torch.tensor([0, 63, 129], device="npu", dtype=torch.int64)
    actual = run(inputs, cu=cu)
    parts = []
    for seq, (start, end) in enumerate([(0, 63), (63, 129)]):
        part = [x[:, start:end].contiguous() for x in inputs[:5]]
        part.append(inputs[5][seq : seq + 1])
        parts.append(run(part))
    # Output is token-major; state and h are sequence/chunk-major. The other
    # intermediates are head-major with token dimension 2.
    dimensions = [1, 0, 2, 2, 2, 2, 2, 2, 2, 2, 1]
    expected = tuple(
        torch.cat([part[index] for part in parts], dim=dim)
        for index, dim in enumerate(dimensions)
    )
    assert_exact(actual, expected)
