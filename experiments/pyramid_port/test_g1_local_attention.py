"""G1 — `PyramidLocalAttention` as rCM's `local_attn`.

G0(b2) drove `AdaptiveKVCache` by hand. This drives the module rCM will
actually call, through the same `AttnContext` / `AttnMaskSpec` objects the
inference script builds, so it also covers the parts G0 could not:
mode dispatch, per-CFG-stream cache isolation, and the guards that stop the
module running under a configuration that would silently produce wrong output.

    PYTHONPATH=. pytest experiments/pyramid_port/test_g1_local_attention.py -q
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

pytest.importorskip("flash_attn", reason="ragged path needs flash-attn")

from rcm.utils.blockmask import AttnMaskSpec, BlockPattern, FlexOrSdpaLocalAttention  # noqa: E402
from rcm.utils.kv_cache import AttnContext, CausalInferenceState, KVCache, KVCacheMode  # noqa: E402
from rcm.utils.pyramid_attention import (  # noqa: E402
    PyramidLocalAttention,
    PyramidSpec,
    retain_everything_spec,
)
from rcm.utils.rope import RopeCache  # noqa: E402

from experiments.pyramid_port.test_g0b_plumbing_parity import _rotate_full, _setup, HEAD_DIM  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

N_HEADS = 12


def _ctx(mode, block_idx, kv_cache, *, cached_k_rotated=False, fast_infer=False):
    return AttnContext(
        mode=mode,
        kv_cache=kv_cache,
        block_range=block_idx,
        layer_idx=0,
        q_block_idx=block_idx,
        rope=RopeCache(cached_k_rotated=cached_k_rotated),
        fast_infer=fast_infer,
    )


def _rollout(module, q_rot, k_raw, v, pattern, num_frames, chunk_frames, frame_tokens,
             kv_cache, num_denoise=0):
    """Run rCM's per-chunk schedule: `num_denoise` READONLY, then one APPEND."""
    out = torch.empty_like(q_rot)
    for blk in range(num_frames // chunk_frames):
        lo = blk * chunk_frames * frame_tokens
        hi = lo + chunk_frames * frame_tokens
        meta = AttnMaskSpec(mode="block_causal", pattern=pattern, q_block_offset=blk)
        for _ in range(num_denoise):
            module(q_rot[:, lo:hi], k_raw[:, lo:hi], v[:, lo:hi],
                   attn_ctx=_ctx(KVCacheMode.READONLY, blk, kv_cache), attn_meta=meta)
        out[:, lo:hi] = module(
            q_rot[:, lo:hi], k_raw[:, lo:hi], v[:, lo:hi],
            attn_ctx=_ctx(KVCacheMode.APPEND, blk, kv_cache), attn_meta=meta,
        )
    return out


@pytest.mark.parametrize("chunk_frames", [1, 3])
@pytest.mark.parametrize("num_denoise", [0, 4])
def test_retain_everything_module_matches_vanilla(chunk_frames, num_denoise):
    """The module, wired as rCM calls it, reproduces block-causal attention."""
    num_frames, height, width = 6, 6, 8
    dev, q, k, v, emb, pattern, frame_tokens = _setup(
        num_frames, height, width, chunk_frames, seed=23
    )
    q_rot = _rotate_full(emb, q, num_frames, height, width)
    k_rot = _rotate_full(emb, k, num_frames, height, width)

    ref = FlexOrSdpaLocalAttention().to(dev)(
        q_rot, k_rot, v,
        attn_meta=AttnMaskSpec(mode="block_causal", pattern=pattern, q_block_offset=0),
    )

    spec = retain_everything_spec(
        num_layers=1, num_heads=N_HEADS, head_dim=HEAD_DIM,
        latent_h=height, latent_w=width, max_frames=num_frames,
    )
    module = PyramidLocalAttention(spec).to(dev)
    got = _rollout(module, q_rot, k, v, pattern, num_frames, chunk_frames,
                   frame_tokens, KVCache(max_len=1), num_denoise=num_denoise)

    diff = (ref.float() - got.float()).abs().max().item()
    scale = ref.float().abs().max().item()
    assert diff / scale < 1e-2, f"max|Δ| = {diff:.3e} (rel {diff / scale:.2e})"


def test_cfg_streams_do_not_share_a_cache():
    """One module instance serves both CFG streams; their caches must not mix.

    rCM allocates a separate `KVCache` per stream, so those objects are the
    stream identity. Feeding stream B different keys must not perturb stream A.
    """
    num_frames, height, width, chunk_frames = 4, 6, 8, 1
    dev, q, k, v, emb, pattern, frame_tokens = _setup(
        num_frames, height, width, chunk_frames, seed=29
    )
    q_rot = _rotate_full(emb, q, num_frames, height, width)
    k_other = torch.randn_like(k)

    spec = retain_everything_spec(1, N_HEADS, HEAD_DIM, height, width, num_frames)

    cache_a, cache_b = KVCache(max_len=1), KVCache(max_len=1)

    solo = PyramidLocalAttention(spec).to(dev)
    alone = _rollout(solo, q_rot, k, v, pattern, num_frames, chunk_frames,
                     frame_tokens, cache_a)

    shared = PyramidLocalAttention(spec).to(dev)
    interleaved = torch.empty_like(q_rot)
    for blk in range(num_frames):
        lo, hi = blk * frame_tokens, (blk + 1) * frame_tokens
        meta = AttnMaskSpec(mode="block_causal", pattern=pattern, q_block_offset=blk)
        interleaved[:, lo:hi] = shared(
            q_rot[:, lo:hi], k[:, lo:hi], v[:, lo:hi],
            attn_ctx=_ctx(KVCacheMode.APPEND, blk, cache_a), attn_meta=meta,
        )
        shared(  # the other stream, same module, different KVCache object
            q_rot[:, lo:hi], k_other[:, lo:hi], v[:, lo:hi],
            attn_ctx=_ctx(KVCacheMode.APPEND, blk, cache_b), attn_meta=meta,
        )

    torch.testing.assert_close(alone, interleaved)


def test_block_zero_starts_a_fresh_clip():
    """Re-running from block 0 must not attend to the previous clip's frames."""
    num_frames, height, width = 4, 6, 8
    dev, q, k, v, emb, pattern, frame_tokens = _setup(num_frames, height, width, 1, seed=31)
    q_rot = _rotate_full(emb, q, num_frames, height, width)

    spec = retain_everything_spec(1, N_HEADS, HEAD_DIM, height, width, num_frames)
    module = PyramidLocalAttention(spec).to(dev)
    kv = KVCache(max_len=1)

    first = _rollout(module, q_rot, k, v, pattern, num_frames, 1, frame_tokens, kv)
    second = _rollout(module, q_rot, k, v, pattern, num_frames, 1, frame_tokens, kv)

    torch.testing.assert_close(first, second)


def test_rejects_post_rope_keys():
    spec = retain_everything_spec(1, N_HEADS, HEAD_DIM, 6, 8, 4)
    module = PyramidLocalAttention(spec).to("cuda")
    dev, q, k, v, emb, pattern, ft = _setup(4, 6, 8, 1, seed=37)
    meta = AttnMaskSpec(mode="block_causal", pattern=pattern, q_block_offset=0)
    with pytest.raises(ValueError, match="cached_k_rotated"):
        module(q[:, :ft], k[:, :ft], v[:, :ft],
               attn_ctx=_ctx(KVCacheMode.APPEND, 0, KVCache(1), cached_k_rotated=True),
               attn_meta=meta)


def test_rejects_fast_infer():
    spec = retain_everything_spec(1, N_HEADS, HEAD_DIM, 6, 8, 4)
    module = PyramidLocalAttention(spec).to("cuda")
    dev, q, k, v, emb, pattern, ft = _setup(4, 6, 8, 1, seed=41)
    meta = AttnMaskSpec(mode="block_causal", pattern=pattern, q_block_offset=0)
    with pytest.raises(ValueError, match="fast_infer"):
        module(q[:, :ft], k[:, :ft], v[:, :ft],
               attn_ctx=_ctx(KVCacheMode.APPEND, 0, KVCache(1), fast_infer=True),
               attn_meta=meta)


def test_rejects_grid_mismatch():
    """A wrong latent_h/latent_w would corrupt every position silently."""
    spec = retain_everything_spec(1, N_HEADS, HEAD_DIM, latent_h=7, latent_w=8, max_frames=4)
    module = PyramidLocalAttention(spec).to("cuda")
    dev, q, k, v, emb, pattern, ft = _setup(4, 6, 8, 1, seed=43)  #真实 grid 是 6x8
    meta = AttnMaskSpec(mode="block_causal", pattern=pattern, q_block_offset=0)
    with pytest.raises(ValueError, match="frame_tokens"):
        module(q[:, :ft], k[:, :ft], v[:, :ft],
               attn_ctx=_ctx(KVCacheMode.APPEND, 0, KVCache(1)), attn_meta=meta)


def test_disabled_mode_falls_back():
    """Bidirectional / teacher-forcing passes must not go through the cache."""
    num_frames, height, width = 3, 6, 8
    dev, q, k, v, emb, pattern, ft = _setup(num_frames, height, width, 1, seed=47)
    spec = retain_everything_spec(1, N_HEADS, HEAD_DIM, height, width, num_frames)
    fallback = FlexOrSdpaLocalAttention().to(dev)
    module = PyramidLocalAttention(spec, fallback=fallback).to(dev)

    ref = fallback(q, k, v, attn_meta=AttnMaskSpec(mode="none"))
    got = module(q, k, v, attn_ctx=None, attn_meta=AttnMaskSpec(mode="none"))
    torch.testing.assert_close(ref, got)


def test_rope_cache_does_not_grow_quadratically():
    """`key_freqs` must stay None while pyramid KV is on.

    `_compute_rope` keys `_rope_cache` on `t_offset` and never evicts, and the
    non-fast_infer branch builds a `key_freqs` covering the whole cached prefix
    -- O(t_offset) per entry, so O(blocks**2) overall. That was 42.9 GiB at 481
    latent frames, the entire memory excess the pyramid arm carried, and none of
    it is ever read: `manages_kv_cache` bypasses `apply_rope(key, key_freqs)`.

    The bug is invisible in output (the tensors are discarded), so only a memory
    assertion catches a regression.
    """
    from rcm.networks.wan2pt1 import WanModel

    num_frames, height, width = 8, 6, 8
    _, _, _, _, _, pattern, _ = _setup(num_frames, height, width, 1, seed=53)

    net = WanModel(model_type="t2v", num_layers=1, dim=1536, ffn_dim=8960,
                   num_heads=N_HEADS).to("cuda")
    spec = retain_everything_spec(1, N_HEADS, HEAD_DIM, latent_h=height,
                                  latent_w=width, max_frames=num_frames)
    net.enable_pyramid_kv(spec)

    # `t_offset` is derived from (pattern, block_cursor), so walk the cursor the
    # way the rollout does rather than setting the offset directly.
    for block_idx in range(num_frames):
        state = CausalInferenceState(mode=KVCacheMode.APPEND, kv_caches=None,
                                     pattern=pattern, block_cursor=block_idx,
                                     fast_infer=False)
        rope = net._compute_rope(1, 1, height, width, state,
                                 AttnMaskSpec(mode="block_causal", pattern=pattern,
                                              q_block_offset=block_idx),
                                 use_fused=False)
        assert rope.key_freqs is None, (
            f"key_freqs allocated at block {block_idx}; it is never read under "
            "pyramid KV and accumulates one growing entry per block"
        )

    # Same walk with pyramid off must still build key_freqs -- otherwise the
    # guard is vacuous and would pass even if the branch were removed outright.
    net.disable_pyramid_kv()
    state = CausalInferenceState(mode=KVCacheMode.APPEND, kv_caches=None,
                                 pattern=pattern, block_cursor=4, fast_infer=False)
    rope = net._compute_rope(1, 1, height, width, state,
                             AttnMaskSpec(mode="block_causal", pattern=pattern,
                                          q_block_offset=4),
                             use_fused=False)
    assert rope.key_freqs is not None


def test_real_strategies_produce_three_policy_types():
    """The paper's spec must route labels to three distinct middle strategies.

    Guards against a config typo silently collapsing every head onto one policy,
    which would still run and still produce video.
    """
    csv = str(Path(__file__).with_name("rcm-head-labels-thp6.4-ths0.8.csv"))


    if not Path(csv).exists():
        pytest.skip(f"labels csv not found at {csv}")
    spec = PyramidSpec(
        num_layers=30, num_heads=12, head_dim=HEAD_DIM,
        latent_h=30, latent_w=52, labels_csv=csv, max_frames=72,
    )
    counts = spec.composition_summary()
    assert set(counts) >= {"CyclicStrategy", "StrideStrategy", "MergeStrategy"}, counts
    assert sum(counts.values()) == 360, counts
