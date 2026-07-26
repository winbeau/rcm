"""G0(b1) — the ragged plumbing must reproduce vanilla rCM attention.

Checks, in order of what they would catch:

1. A block-by-block rollout through the ragged path equals one full-sequence
   block-causal FlexAttention pass. Catches wrong `cu_seqlens`, wrong per-head
   packing order, and a wrong causality assumption.
2. The RoPE bridge applied to a growing cache equals rCM's own key rotation.
   Catches position bookkeeping drift as the cache grows across blocks.
3. Rotating each block's key once at append time (what a real cache does)
   equals rotating the whole prefix at read time (what rCM does).

Everything here is the *port's* plumbing, with no pyramidkv strategies in play,
so a failure localizes to this glue rather than to the policy stack.

    PYTHONPATH=. pytest experiments/pyramid_port/test_g0b_plumbing_parity.py -q
"""
from __future__ import annotations

import pytest
import torch

pytest.importorskip("flash_attn", reason="ragged path needs flash-attn")

from rcm.networks.wan2pt1 import VideoRopePosition3DEmb  # noqa: E402
from rcm.pyramidkv import rope as pyramidkv_rope  # noqa: E402
from rcm.utils.blockmask import AttnMaskSpec, BlockPattern, FlexOrSdpaLocalAttention  # noqa: E402
from rcm.utils.rope import apply_rope  # noqa: E402

from rcm.utils.pyramid_attention import pack_dense_kv, ragged_attention  # noqa: E402
from rcm.utils.pyramid_rope import (  # noqa: E402
    build_pos_3d,
    build_pyramidkv_freq_table,
)

HEAD_DIM = 128
N_HEADS = 12

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="flash-attn ragged path needs CUDA"
)


def _setup(num_frames, height, width, chunk_frames, dtype=torch.bfloat16, seed=0):
    torch.manual_seed(seed)
    dev = "cuda"
    frame_tokens = height * width
    seq = num_frames * frame_tokens
    q = torch.randn(1, seq, N_HEADS, HEAD_DIM, device=dev, dtype=dtype)
    k = torch.randn(1, seq, N_HEADS, HEAD_DIM, device=dev, dtype=dtype)
    v = torch.randn(1, seq, N_HEADS, HEAD_DIM, device=dev, dtype=dtype)
    emb = VideoRopePosition3DEmb(head_dim=HEAD_DIM, len_h=128, len_w=128, len_t=32).to(dev)
    pattern = BlockPattern(
        frame_tokens=frame_tokens, first_chunk_frames=chunk_frames, chunk_frames=chunk_frames
    )
    return dev, q, k, v, emb, pattern, frame_tokens


def _rotate_full(emb, x, num_frames, height, width):
    freqs = emb.generate_embeddings(
        torch.Size([1, num_frames, height, width, HEAD_DIM]), t_start=0
    )
    return apply_rope(x, freqs, fused=False)


@pytest.mark.parametrize(
    "num_frames,height,width,chunk_frames",
    [
        (6, 6, 8, 1),   # chunk_t=1, the rCM extraction geometry
        (6, 6, 8, 3),   # chunk_t=3, Self-Forcing's block size
        (9, 5, 7, 3),
    ],
)
def test_ragged_rollout_equals_full_block_causal(num_frames, height, width, chunk_frames):
    """A growing per-head cache, read block by block, equals one masked pass."""
    dev, q, k, v, emb, pattern, frame_tokens = _setup(num_frames, height, width, chunk_frames)

    q_rot = _rotate_full(emb, q, num_frames, height, width)
    k_rot = _rotate_full(emb, k, num_frames, height, width)

    local = FlexOrSdpaLocalAttention().to(dev)
    spec = AttnMaskSpec(mode="block_causal", pattern=pattern, q_block_offset=0)
    ref = local(q_rot, k_rot, v, attn_meta=spec)

    num_blocks = num_frames // chunk_frames
    got = torch.empty_like(ref)
    for blk in range(num_blocks):
        lo = blk * chunk_frames * frame_tokens
        hi = lo + chunk_frames * frame_tokens
        k_flat, v_flat, cu_k, max_k = pack_dense_kv(k_rot[:, :hi], v[:, :hi])
        got[:, lo:hi] = ragged_attention(q_rot[:, lo:hi], k_flat, v_flat, cu_k, max_k)

    diff = (ref.float() - got.float()).abs().max().item()
    scale = ref.float().abs().max().item()
    # Measured 2.8e-3 against a 2.7e-3 kernel noise floor (flash-attn varlen vs
    # FlexAttention in bf16, no ragged packing involved) -- see report_g0b.py.
    # 1e-2 leaves ~3.5x margin over the observed value while staying tight
    # enough that a real plumbing error cannot hide under it.
    assert diff / scale < 1e-2, f"max|Δ| = {diff:.3e} (rel {diff / scale:.2e})"


@pytest.mark.parametrize("chunk_frames", [1, 3])
def test_bridge_rope_on_growing_cache_matches_rcm(chunk_frames):
    """Position bookkeeping must not drift as the cache grows."""
    num_frames, height, width = 9, 5, 7
    dev, _, k, _, emb, _, frame_tokens = _setup(num_frames, height, width, chunk_frames, seed=3)

    ref = _rotate_full(emb, k, num_frames, height, width)

    table = build_pyramidkv_freq_table(
        HEAD_DIM, max_pos=max(num_frames, height, width), device=dev
    )
    b, s, h, d = k.shape
    pos = build_pos_3d(torch.arange(num_frames, device=dev), height, width, device=dev)
    k_flat = k.permute(0, 2, 1, 3).reshape(b * h * s, d)
    rotated = pyramidkv_rope.apply_rope_to_flat_k(k_flat, pos.repeat(b * h, 1), freqs=table)
    got = rotated.reshape(b, h, s, d).permute(0, 2, 1, 3)

    diff = (ref.float() - got.float()).abs().max().item()
    assert diff < 2e-2, f"max|Δ| = {diff:.3e}"


def test_rotate_at_append_equals_rotate_at_read():
    """Per-block rotation at append time == whole-prefix rotation at read time.

    A real cache rotates each block once when it arrives. rCM instead rotates
    the full materialized key every step. These must agree, otherwise the port
    would have to re-rotate the entire cache on every denoising step.
    """
    num_frames, height, width, chunk_frames = 9, 5, 7, 3
    dev, _, k, _, emb, _, frame_tokens = _setup(num_frames, height, width, chunk_frames, seed=5)

    read_time = _rotate_full(emb, k, num_frames, height, width)

    append_time = torch.empty_like(k)
    for blk in range(num_frames // chunk_frames):
        t0 = blk * chunk_frames
        lo, hi = t0 * frame_tokens, (t0 + chunk_frames) * frame_tokens
        freqs = emb.generate_embeddings(
            torch.Size([1, chunk_frames, height, width, HEAD_DIM]), t_start=t0
        )
        append_time[:, lo:hi] = apply_rope(k[:, lo:hi], freqs, fused=False)

    diff = (read_time.float() - append_time.float()).abs().max().item()
    assert diff == 0.0, f"max|Δ| = {diff:.3e}"


def test_ragged_equals_dense_when_all_heads_keep_everything():
    """Degenerate identity: uniform cache lengths must reproduce dense SDPA."""
    num_frames, height, width = 4, 6, 8
    dev, q, k, v, emb, _, _ = _setup(num_frames, height, width, 1, seed=7)
    q_rot = _rotate_full(emb, q, num_frames, height, width)
    k_rot = _rotate_full(emb, k, num_frames, height, width)

    local = FlexOrSdpaLocalAttention().to(dev)
    ref = local(q_rot, k_rot, v, attn_meta=AttnMaskSpec(mode="none"))

    k_flat, v_flat, cu_k, max_k = pack_dense_kv(k_rot, v)
    got = ragged_attention(q_rot, k_flat, v_flat, cu_k, max_k)

    diff = (ref.float() - got.float()).abs().max().item()
    scale = ref.float().abs().max().item()
    # Measured 2.8e-3 against a 2.7e-3 kernel noise floor (flash-attn varlen vs
    # FlexAttention in bf16, no ragged packing involved) -- see report_g0b.py.
    # 1e-2 leaves ~3.5x margin over the observed value while staying tight
    # enough that a real plumbing error cannot hide under it.
    assert diff / scale < 1e-2, f"max|Δ| = {diff:.3e} (rel {diff / scale:.2e})"


def test_per_head_lengths_actually_differ():
    """Guard the gate itself: a heterogeneous pack must change the output.

    Without this, every test above would still pass if `pack_dense_kv` silently
    produced uniform lengths and the ragged path were never exercised.
    """
    num_frames, height, width = 6, 6, 8
    dev, q, k, v, emb, _, frame_tokens = _setup(num_frames, height, width, 1, seed=11)
    q_rot = _rotate_full(emb, q, num_frames, height, width)
    k_rot = _rotate_full(emb, k, num_frames, height, width)

    b, s, h, d = k_rot.shape
    keep = [s - (i % 3) * frame_tokens for i in range(h)]  # 3 distinct lengths
    k_parts, v_parts, lengths = [], [], []
    for hi in range(h):
        n = keep[hi]
        k_parts.append(k_rot[0, s - n :, hi])
        v_parts.append(v[0, s - n :, hi])
        lengths.append(n)
    k_flat = torch.cat(k_parts, dim=0)
    v_flat = torch.cat(v_parts, dim=0)
    cu_k = torch.tensor([0] + lengths, dtype=torch.int32, device=dev).cumsum(0, dtype=torch.int32)

    ragged_out = ragged_attention(q_rot, k_flat, v_flat, cu_k, max(lengths))

    k_all, v_all, cu_all, max_all = pack_dense_kv(k_rot, v)
    full_out = ragged_attention(q_rot, k_all, v_all, cu_all, max_all)

    assert len(set(lengths)) == 3, "fixture must produce heterogeneous lengths"
    # Heads truncated by 0 frames agree; truncated heads must not.
    untouched = [hi for hi in range(h) if keep[hi] == s]
    truncated = [hi for hi in range(h) if keep[hi] != s]
    for hi in untouched:
        d0 = (ragged_out[0, :, hi].float() - full_out[0, :, hi].float()).abs().max().item()
        assert d0 < 1e-3, f"head {hi} kept everything but changed by {d0:.3e}"
    moved = max(
        (ragged_out[0, :, hi].float() - full_out[0, :, hi].float()).abs().max().item()
        for hi in truncated
    )
    assert moved > 1e-2, f"truncating heads did not change the output ({moved:.3e})"
