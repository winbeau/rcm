"""Opt-in actual retained-K telemetry for rCM PyramidKV runs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class RetainedKTelemetry:
    """Collect per-block, per-head retained token lengths.

    The lengths come from the ragged ``cu_seqlens_k`` emitted by the cache
    readout, not from configured capacities. This distinction matters for
    MergeStrategy, whose capacity is not necessarily a whole-frame count.
    """

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def record(
        self,
        *,
        layer_idx: int,
        batch_size: int,
        num_heads: int,
        block_idx: int,
        mode: str,
        pass_name: str,
        stream: str,
        denoise_step: int,
        cu_seqlens_k,
        max_seqlen_k: int,
        query_tokens: int,
        frame_seq_length: int,
        dense_prefix_tokens: int,
    ) -> None:
        values = [int(v) for v in cu_seqlens_k.detach().cpu().tolist()]
        expected = batch_size * num_heads + 1
        if len(values) != expected:
            raise ValueError(f"cu_seqlens_k has {len(values)} entries, expected {expected}")
        if values[0] != 0:
            raise ValueError(f"cu_seqlens_k must start at zero, got {values[0]}")
        lengths = [values[i + 1] - values[i] for i in range(len(values) - 1)]
        if any(length < 0 for length in lengths):
            raise ValueError(f"cu_seqlens_k is not monotonic: {values}")
        observed_max = max(lengths, default=0)
        if int(max_seqlen_k) != observed_max:
            raise ValueError(
                f"max_seqlen_k={int(max_seqlen_k)} disagrees with observed max={observed_max}"
            )
        per_batch_head = [lengths[i * num_heads : (i + 1) * num_heads] for i in range(batch_size)]
        mean_retained = float(sum(lengths) / len(lengths)) if lengths else 0.0
        dense_prefix_tokens = int(dense_prefix_tokens)
        self.records.append(
            {
                "layer_idx": int(layer_idx),
                "block_idx": int(block_idx),
                "mode": str(mode),
                "pass_name": str(pass_name),
                "stream": str(stream),
                "denoise_step": int(denoise_step),
                "batch_size": int(batch_size),
                "num_heads": int(num_heads),
                "query_tokens": int(query_tokens),
                "frame_seq_length": int(frame_seq_length),
                "dense_prefix_tokens": dense_prefix_tokens,
                "max_seqlen_k": int(max_seqlen_k),
                "total_retained_tokens": int(sum(lengths)),
                "min_retained_tokens": int(min(lengths)) if lengths else 0,
                "mean_retained_tokens": mean_retained,
                "max_retained_tokens": observed_max,
                "mean_retained_fraction_of_dense": (
                    mean_retained / dense_prefix_tokens if dense_prefix_tokens else 0.0
                ),
                "per_batch_head_retained_tokens": per_batch_head,
            }
        )

    def write_jsonl(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            for record in self.records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
