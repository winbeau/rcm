"""G0(b2) — a retain-everything AdaptiveKVCache must reproduce vanilla rCM.

b1 validated the port's own plumbing with a trivial cache. This swaps in the
real `pyramidkv.AdaptiveKVCache`, configured so no head drops anything, and
requires the same answer. What it catches:

- `update()` being driven with the wrong `current_start` / `grid_sizes`
- the flat readout ordering `(b * H + h)` disagreeing with the packing b1 assumed
- `pos_3d` coming back from the cache in a different convention than the bridge
- the noisy/clean double-pass corrupting committed state

If this passes, the only thing left between here and G1 is the strategies
themselves actually dropping frames.

    PYTHONPATH=. pytest experiments/pyramid_port/test_g0b2_adaptive_cache_identity.py -q
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

_DEFAULT_PF = Path(__file__).resolve().parents[3] / "Pyramid-Forcing"
_PF_ROOT = Path(os.environ.get("PYRAMIDKV_ROOT", _DEFAULT_PF))
if str(_PF_ROOT) not in sys.path:
    sys.path.insert(0, str(_PF_ROOT))

pytest.importorskip("flash_attn", reason="ragged path needs flash-attn")
pyramidkv = pytest.importorskip("pyramidkv")

from rcm.utils.blockmask import AttnMaskSpec, BlockPattern, FlexOrSdpaLocalAttention  # noqa: E402

from experiments.pyramid_port.ragged_attention import ragged_attention  # noqa: E402
from experiments.pyramid_port.rope_bridge import build_pyramidkv_freq_table  # noqa: E402
from experiments.pyramid_port.test_g0b_plumbing_parity import _rotate_full, _setup, HEAD_DIM  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs CUDA"
)

N_HEADS = 12


def _retain_everything_cache(num_frames: int, frame_seqlen: int, layer_idx: int = 0):
    """An AdaptiveKVCache whose composition drops nothing.

    All middle strategies off and `recent_frames` wider than the clip, so every
    head's window is the whole prefix. Any deviation from vanilla rCM then comes
    from the machinery, not from the policy.
    """
    from pyramidkv import PyramidKVConfig, build_compositions

    capacity = num_frames * frame_seqlen * 4  # never binding
    config = PyramidKVConfig(
        config_path=None,
        num_layers=1,
        num_heads=N_HEADS,
        default_capacity=capacity,
        frame_seq_length=frame_seqlen,
    )
    config.compositions = build_compositions(
        num_layers=1,
        num_heads=N_HEADS,
        capacities=config.capacity_map,
        csv_path=None,
        cyclic_enabled=False,
        lag_enabled=False,
        stride_enabled=False,
        merge_enabled=False,
        osc_sink_frames=0,
        stable_sink_frames=0,
        recent_frames=num_frames + 8,
        stable_recent_frames=num_frames + 8,
    )
    config.policies = config.compositions

    from pyramidkv import AdaptiveKVCache

    # frame_seq_length is not an AdaptiveKVCache kwarg; the base class reads it
    # off the config, which is why it is set there.
    return AdaptiveKVCache(
        config,
        batch_size=1,
        num_heads=N_HEADS,
        head_dim=HEAD_DIM,
        layer_idx=layer_idx,
        tail_len=capacity,
    )


@pytest.mark.parametrize("chunk_frames", [1, 3])
def test_retain_everything_matches_block_causal(chunk_frames):
    num_frames, height, width = 6, 6, 8
    dev, q, k, v, emb, pattern, frame_tokens = _setup(
        num_frames, height, width, chunk_frames, seed=13
    )
    q_rot = _rotate_full(emb, q, num_frames, height, width)
    k_rot_ref = _rotate_full(emb, k, num_frames, height, width)

    local = FlexOrSdpaLocalAttention().to(dev)
    spec = AttnMaskSpec(mode="block_causal", pattern=pattern, q_block_offset=0)
    ref = local(q_rot, k_rot_ref, v, attn_meta=spec)

    cache = _retain_everything_cache(num_frames, frame_tokens)
    table = build_pyramidkv_freq_table(
        HEAD_DIM, max_pos=max(num_frames, height, width), device=dev
    )
    grid_sizes = torch.tensor([[chunk_frames, height, width]], dtype=torch.long, device=dev)

    got = torch.empty_like(ref)
    for blk in range(num_frames // chunk_frames):
        lo = blk * chunk_frames * frame_tokens
        hi = lo + chunk_frames * frame_tokens
        # Raw (pre-RoPE) K/V go into the cache; RoPE is applied at readout, which
        # is what `cached_k_rotated=False` buys us on the rCM side.
        cache.update(
            k[:, lo:hi],
            v[:, lo:hi],
            current_start=lo,
            grid_sizes=grid_sizes,
            cache_update_mode="clean",
        )
        k_flat, v_flat, cu_k, max_k, pos = cache.get_flat_kv_and_pos()

        expected_len = hi
        lengths = (cu_k[1:] - cu_k[:-1]).tolist()
        assert set(lengths) == {expected_len}, (
            f"block {blk}: retain-everything cache dropped tokens, lengths={sorted(set(lengths))}"
        )

        k_flat = cache.apply_rope_to_flat_k(k_flat, pos, freqs=table)
        got[:, lo:hi] = ragged_attention(q_rot[:, lo:hi], k_flat, v_flat, cu_k, max_k)

    diff = (ref.float() - got.float()).abs().max().item()
    scale = ref.float().abs().max().item()
    assert diff / scale < 1e-2, f"max|Δ| = {diff:.3e} (rel {diff / scale:.2e})"


def test_noisy_passes_do_not_corrupt_committed_state():
    """rCM runs several READONLY denoising steps before one APPEND per chunk.

    Mapping READONLY -> "default" and APPEND -> "clean", repeated noisy updates
    on the same block must leave the same cache as a single clean update.
    """
    num_frames, height, width, chunk_frames = 6, 6, 8, 1
    dev, q, k, v, emb, pattern, frame_tokens = _setup(
        num_frames, height, width, chunk_frames, seed=17
    )
    grid_sizes = torch.tensor([[chunk_frames, height, width]], dtype=torch.long, device=dev)

    def run(num_noisy: int):
        cache = _retain_everything_cache(num_frames, frame_tokens)
        for blk in range(num_frames):
            lo, hi = blk * frame_tokens, (blk + 1) * frame_tokens
            for _ in range(num_noisy):
                cache.update(
                    k[:, lo:hi], v[:, lo:hi], current_start=lo,
                    grid_sizes=grid_sizes, cache_update_mode="default",
                )
            cache.update(
                k[:, lo:hi], v[:, lo:hi], current_start=lo,
                grid_sizes=grid_sizes, cache_update_mode="clean",
            )
        k_flat, v_flat, cu_k, max_k, pos = cache.get_flat_kv_and_pos()
        return k_flat, cu_k, pos

    k0, cu0, pos0 = run(0)
    k4, cu4, pos4 = run(4)  # rCM's --num_steps 4

    torch.testing.assert_close(cu0, cu4)
    assert k0.shape == k4.shape, f"{k0.shape} vs {k4.shape}"
    torch.testing.assert_close(k0, k4)
    torch.testing.assert_close(pos0, pos4)


def test_cache_positions_match_the_bridge_convention():
    """The cache's own `pos_3d` must equal what `build_pos_3d` produces."""
    from experiments.pyramid_port.rope_bridge import build_pos_3d

    num_frames, height, width = 4, 6, 8
    dev, _, k, v, _, _, frame_tokens = _setup(num_frames, height, width, 1, seed=19)
    grid_sizes = torch.tensor([[1, height, width]], dtype=torch.long, device=dev)

    cache = _retain_everything_cache(num_frames, frame_tokens)
    for blk in range(num_frames):
        lo, hi = blk * frame_tokens, (blk + 1) * frame_tokens
        cache.update(
            k[:, lo:hi], v[:, lo:hi], current_start=lo,
            grid_sizes=grid_sizes, cache_update_mode="clean",
        )
    _, _, cu_k, _, pos = cache.get_flat_kv_and_pos()

    expected = build_pos_3d(torch.arange(num_frames, device=dev), height, width, device=dev)
    seq_len = int(cu_k[1] - cu_k[0])
    assert seq_len == num_frames * frame_tokens
    torch.testing.assert_close(pos[:seq_len].long(), expected)
