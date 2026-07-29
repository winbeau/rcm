"""CPU gates for actual retained-K accounting and context propagation."""
from __future__ import annotations

import json

import pytest
import torch

from rcm.utils.retained_k import RetainedKTelemetry


def test_retained_k_uses_actual_ragged_lengths(tmp_path):
    telemetry = RetainedKTelemetry()
    telemetry.record(
        layer_idx=3,
        batch_size=2,
        num_heads=2,
        block_idx=4,
        mode="readonly",
        pass_name="denoise0",
        stream="cond",
        denoise_step=0,
        cu_seqlens_k=torch.tensor([0, 3, 8, 10, 17], dtype=torch.int32),
        max_seqlen_k=7,
        query_tokens=4,
        frame_seq_length=4,
        dense_prefix_tokens=20,
    )

    record = telemetry.records[0]
    assert record["per_batch_head_retained_tokens"] == [[3, 5], [2, 7]]
    assert record["total_retained_tokens"] == 17
    assert record["min_retained_tokens"] == 2
    assert record["mean_retained_tokens"] == pytest.approx(4.25)
    assert record["max_retained_tokens"] == 7
    assert record["mean_retained_fraction_of_dense"] == pytest.approx(0.2125)
    assert record["pass_name"] == "denoise0"

    output = tmp_path / "retained_k.jsonl"
    telemetry.write_jsonl(output)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert rows == telemetry.records


def test_retained_k_rejects_malformed_prefix_sums():
    telemetry = RetainedKTelemetry()
    common = {
        "layer_idx": 0,
        "batch_size": 1,
        "num_heads": 2,
        "block_idx": 0,
        "mode": "append",
        "pass_name": "append",
        "stream": "cond",
        "denoise_step": -1,
        "max_seqlen_k": 4,
        "query_tokens": 4,
        "frame_seq_length": 4,
        "dense_prefix_tokens": 4,
    }

    with pytest.raises(ValueError, match="start at zero"):
        telemetry.record(cu_seqlens_k=torch.tensor([1, 3, 5]), **common)
    with pytest.raises(ValueError, match="not monotonic"):
        telemetry.record(cu_seqlens_k=torch.tensor([0, 4, 3]), **common)
    with pytest.raises(ValueError, match="disagrees with observed max"):
        telemetry.record(cu_seqlens_k=torch.tensor([0, 2, 4]), **common)


def test_causal_state_propagates_telemetry_and_pass_metadata():
    pytest.importorskip("torch.nn.attention.flex_attention")
    from rcm.utils.kv_cache import CausalInferenceState, KVCache, KVCacheMode

    telemetry = RetainedKTelemetry()
    state = CausalInferenceState(
        mode=KVCacheMode.READONLY,
        kv_caches=[KVCache(max_len=8)],
        block_cursor=2,
        retained_k_observer=telemetry,
        pass_name="denoise0",
        denoise_step=0,
        stream_name="uncond",
    )

    context = state.attn_ctx(0)

    assert context is not None
    assert context.retained_k_observer is telemetry
    assert context.pass_name == "denoise0"
    assert context.denoise_step == 0
    assert context.stream_name == "uncond"
    assert context.mode == KVCacheMode.READONLY
