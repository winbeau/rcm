# Frame-level attention observation for causal video rollout.
#
# Plugs into the `attn_observer` extension point that DistributedAttention
# already exposes (rcm/utils/a2a_cp.py:_collect_attention_stats). The observer
# is handed post-RoPE `query` and the fully materialized post-RoPE `key` right
# before the attention op runs, which is exactly the pair whose logits we want
# to summarize.
#
# What "frame-level attention" means here matches the convention used by the
# Self-Forcing / Pyramid-Forcing analysis notebooks: for a (query frame, key
# frame) pair, the scalar is the *mean pre-softmax logit* over every
# (query token, key token) pair in that frame block.
#
#     A[h, qf, kf] = mean_{i in qf, j in kf}  scale * <q_i^h, k_j^h>
#
# Computing that naively costs O(F^2 * frame_tokens^2 * D) and materializes
# [H, frame_tokens, frame_tokens] score blocks. It does not need to: the dot
# product is bilinear and the mean is linear, so the mean of the logits is the
# logit of the means,
#
#     mean_{i,j} <q_i, k_j> = < mean_i q_i , mean_j k_j >
#
# which lets us mean-pool each frame's tokens first and then contract. That is
# an exact identity in exact arithmetic, not an approximation, and it drops the
# cost to O(F^2 * D) -- for 480p (frame_tokens = 1560) that is ~2.4M times less
# work, and the peak allocation goes from gigabytes to kilobytes. Set
# `method="naive"` to run the literal definition instead; `verify_against_naive`
# checks the two agree.

import math
from typing import Dict, Iterable, List, Optional

import torch


def _pool_frames(x_B_S_H_D: torch.Tensor, frame_tokens: int) -> torch.Tensor:
    """Mean-pool token positions within each frame.

    Args:
        x_B_S_H_D: [B, S, H, D] with S a whole number of frames.
        frame_tokens: tokens per frame.

    Returns:
        [B, F, H, D] where F = S // frame_tokens.
    """
    b, s, h, d = x_B_S_H_D.shape
    if s % frame_tokens != 0:
        raise ValueError(f"sequence length {s} is not a multiple of frame_tokens={frame_tokens}")
    return x_B_S_H_D.view(b, s // frame_tokens, frame_tokens, h, d).mean(dim=2)


def frame_logits_pooled(query: torch.Tensor, key: torch.Tensor, frame_tokens: int, scale: float) -> torch.Tensor:
    """Frame-level mean logits via mean-pooling. O(F^2 * D)."""
    q_pooled = _pool_frames(query.float(), frame_tokens)  # [B, Fq, H, D]
    k_pooled = _pool_frames(key.float(), frame_tokens)  # [B, Fk, H, D]
    # Average over batch as well, matching the naive reduction below.
    return torch.einsum("bqhd,bkhd->hqk", q_pooled, k_pooled) * (scale / q_pooled.shape[0])


def frame_logits_naive(query: torch.Tensor, key: torch.Tensor, frame_tokens: int, scale: float, chunk_frames: int = 4) -> torch.Tensor:
    """Frame-level mean logits by materializing token-level scores.

    The literal definition, kept as a reference implementation to validate
    `frame_logits_pooled` against. Chunked over key frames so the intermediate
    score block stays bounded.
    """
    q = query.float().transpose(1, 2)  # [B, H, Sq, D]
    k = key.float().transpose(1, 2)  # [B, H, Sk, D]
    b, h, sq, _ = q.shape
    sk = k.shape[2]
    fq, fk = sq // frame_tokens, sk // frame_tokens

    out = torch.zeros(h, fq, fk, dtype=torch.float32, device=q.device)
    for kf_start in range(0, fk, chunk_frames):
        kf_end = min(kf_start + chunk_frames, fk)
        k_chunk = k[:, :, kf_start * frame_tokens : kf_end * frame_tokens, :]
        for qf in range(fq):
            q_frame = q[:, :, qf * frame_tokens : (qf + 1) * frame_tokens, :]
            scores = torch.matmul(q_frame, k_chunk.transpose(-2, -1)) * scale  # [B,H,ft,chunk*ft]
            for kf_local in range(kf_end - kf_start):
                block = scores[:, :, :, kf_local * frame_tokens : (kf_local + 1) * frame_tokens]
                out[:, qf, kf_start + kf_local] = block.mean(dim=(0, 2, 3))
            del scores
        del k_chunk
    return out


class FrameAttentionObserver:
    """Accumulates a [num_heads, F, F] frame-level attention matrix per layer.

    Implements the `observe(...)` protocol that
    `DistributedAttention._collect_attention_stats` calls. Attach it to the
    `CausalInferenceState` of the pass you want to sample -- for causal rollout
    that is the once-per-chunk `KVCacheMode.APPEND` forward, which runs at
    t=0 on the denoised chunk and therefore sees clean context.

    Rows are written per chunk, so the matrix fills in block-triangularly as
    the rollout advances.
    """

    def __init__(
        self,
        layer_indices: Iterable[int],
        frame_tokens: int,
        num_heads: int,
        num_frames: int,
        first_chunk_frames: int = 1,
        chunk_frames: int = 1,
        method: str = "pooled",
        naive_chunk_frames: int = 4,
        store_device: str = "cpu",
        verify_once: bool = False,
    ):
        if method not in {"pooled", "naive"}:
            raise ValueError(f"method must be 'pooled' or 'naive', got {method!r}")
        self.layer_indices = sorted(set(int(i) for i in layer_indices))
        self.frame_tokens = int(frame_tokens)
        self.num_heads = int(num_heads)
        self.num_frames = int(num_frames)
        self.first_chunk_frames = int(first_chunk_frames)
        self.chunk_frames = int(chunk_frames)
        self.method = method
        self.naive_chunk_frames = int(naive_chunk_frames)
        self.store_device = store_device
        self.verify_once = bool(verify_once)
        self.verification_stats: Optional[Dict[str, float]] = None

        self.full: Dict[int, torch.Tensor] = {
            layer: torch.zeros(self.num_heads, self.num_frames, self.num_frames, dtype=torch.float32, device=store_device)
            for layer in self.layer_indices
        }
        # Which (layer, q_block) pairs have been written, so a second forward
        # pass over the same chunk cannot silently overwrite the first.
        self._seen: set = set()
        self._seen_by_layer: Dict[int, set[int]] = {layer: set() for layer in self.layer_indices}
        self._key_frame_ends: Dict[int, Dict[int, int]] = {layer: {} for layer in self.layer_indices}
        self._query_frame_spans: Dict[int, Dict[int, List[int]]] = {
            layer: {} for layer in self.layer_indices
        }
        self.observed_blocks: List[int] = []

    def q_frame_start(self, q_block_idx: int) -> int:
        """First frame index covered by chunk `q_block_idx`."""
        if q_block_idx == 0:
            return 0
        return self.first_chunk_frames + (q_block_idx - 1) * self.chunk_frames

    def observe(self, layer_idx: int, q_block_idx: int, query: torch.Tensor, key: torch.Tensor, cached_len: int):
        del cached_len  # frame bookkeeping comes from q_block_idx and key length
        if layer_idx not in self.full:
            return
        if (layer_idx, q_block_idx) in self._seen:
            raise ValueError(f"duplicate attention observation for layer={layer_idx}, block={q_block_idx}")
        self._seen.add((layer_idx, q_block_idx))
        self._seen_by_layer[layer_idx].add(int(q_block_idx))

        if self.verify_once and self.verification_stats is None:
            self.verification_stats = self.verify_against_naive(query, key)

        scale = 1.0 / math.sqrt(query.shape[-1])
        if self.method == "pooled":
            frame_attn = frame_logits_pooled(query, key, self.frame_tokens, scale)
        else:
            frame_attn = frame_logits_naive(query, key, self.frame_tokens, scale, self.naive_chunk_frames)

        q_start = self.q_frame_start(q_block_idx)
        q_end = q_start + frame_attn.shape[1]
        k_end = frame_attn.shape[2]
        if q_end > self.num_frames or k_end > self.num_frames:
            raise ValueError(
                f"observation for block {q_block_idx} spans frames q[{q_start}:{q_end}] k[:{k_end}] "
                f"but the accumulator only holds {self.num_frames} frames"
            )

        self.full[layer_idx][:, q_start:q_end, :k_end] = frame_attn.to(self.store_device)
        self._key_frame_ends[layer_idx][int(q_block_idx)] = int(k_end)
        self._query_frame_spans[layer_idx][int(q_block_idx)] = [
            int(q_start),
            int(q_end),
        ]
        if q_block_idx not in self.observed_blocks:
            self.observed_blocks.append(q_block_idx)

    def coverage(self, expected_blocks: int) -> Dict[str, object]:
        """Return per-layer block/key-horizon coverage and fail on gaps."""
        expected = set(range(int(expected_blocks)))
        per_layer = {}
        final_block = int(expected_blocks) - 1
        for layer in self.layer_indices:
            seen = self._seen_by_layer[layer]
            key_ends = self._key_frame_ends[layer]
            query_spans = self._query_frame_spans[layer]
            missing = sorted(expected - seen)
            extra = sorted(seen - expected)
            invalid_horizons = []
            for block in sorted(expected.intersection(seen)):
                q_start, q_end = query_spans[block]
                key_end = key_ends[block]
                if key_end != q_end:
                    invalid_horizons.append(
                        {
                            "block": block,
                            "query_span": [q_start, q_end],
                            "key_frame_end": key_end,
                        }
                    )
            final_key_end = key_ends.get(final_block, 0)
            tensor_shape = list(self.full[layer].shape)
            complete = (
                not missing
                and not extra
                and not invalid_horizons
                and final_key_end == self.num_frames
                and tensor_shape
                == [self.num_heads, self.num_frames, self.num_frames]
            )
            per_layer[str(layer)] = {
                "observed_blocks": sorted(seen),
                "missing_blocks": missing,
                "extra_blocks": extra,
                "invalid_horizons": invalid_horizons,
                "final_key_frame_end": final_key_end,
                "tensor_shape": tensor_shape,
                "complete": complete,
            }
        complete = all(row["complete"] for row in per_layer.values())
        result = {
            "expected_blocks": int(expected_blocks),
            "layers": per_layer,
            "complete": complete,
        }
        if not complete:
            raise RuntimeError(f"incomplete attention coverage: {result}")
        return result

    def block_sizes(self, num_blocks: Optional[int] = None) -> List[int]:
        """Frames per chunk, in rollout order (the analysis notebooks' `block_sizes`)."""
        if num_blocks is None:
            num_blocks = 1 + (self.num_frames - self.first_chunk_frames) // self.chunk_frames
        return [self.first_chunk_frames] + [self.chunk_frames] * (num_blocks - 1)

    def verify_against_naive(self, query: torch.Tensor, key: torch.Tensor, rtol: float = 2e-3, atol: float = 2e-3) -> Dict[str, float]:
        """Check the pooled fast path reproduces the literal definition.

        Returns the observed error statistics; raises if they exceed tolerance.
        Tolerances are loose because the two orderings accumulate float32
        rounding differently over ~1560 terms.
        """
        scale = 1.0 / math.sqrt(query.shape[-1])
        pooled = frame_logits_pooled(query, key, self.frame_tokens, scale)
        naive = frame_logits_naive(query, key, self.frame_tokens, scale, self.naive_chunk_frames)
        abs_err = (pooled - naive).abs()
        denom = naive.abs().clamp_min(1e-6)
        stats = {
            "max_abs_err": abs_err.max().item(),
            "mean_abs_err": abs_err.mean().item(),
            "max_rel_err": (abs_err / denom).max().item(),
            "naive_absmax": naive.abs().max().item(),
        }
        if not torch.allclose(pooled, naive, rtol=rtol, atol=atol):
            raise AssertionError(f"pooled frame logits disagree with the naive reference: {stats}")
        return stats
