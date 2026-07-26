"""G0(a) — rCM and pyramidkv must apply the *same* rotation.

A mismatch here does not raise; it degrades video quality, which would be
misread as the method failing on rCM. This is the gate that catches it before
any quality number is produced.

Needs `pyramidkv` importable — point `PYRAMIDKV_ROOT` at the Pyramid-Forcing
checkout (default: the sibling submodule in the NeurIPS2026 super-repo).

    PYTHONPATH=. pytest experiments/pyramid_port/test_g0_rope_parity.py -q
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

pyramidkv_rope = pytest.importorskip(
    "pyramidkv.rope", reason=f"pyramidkv not importable from {_PF_ROOT}"
)

from rcm.networks.wan2pt1 import VideoRopePosition3DEmb  # noqa: E402
from rcm.utils.rope import apply_rope  # noqa: E402

from experiments.pyramid_port.rope_bridge import (  # noqa: E402
    build_pos_3d,
    build_pyramidkv_freq_table,
)

HEAD_DIM = 128  # Wan2.1 T2V 1.3B: 30 layers x 12 heads x 128


def _rcm_embedder(device):
    emb = VideoRopePosition3DEmb(head_dim=HEAD_DIM, len_h=128, len_w=128, len_t=32)
    emb.to(device)
    return emb


def _rcm_rotate(x, freqs, fused):
    return apply_rope(x, freqs, fused=fused)


def _pyramidkv_rotate(x, pos_3d, table):
    """Run x through pyramidkv's flat-K path and restore the [B, S, H, D] layout."""
    b, s, h, d = x.shape
    # apply_rope_to_flat_k takes [N, D] with one pos row per token; every head of
    # a token shares that token's position, so tile the positions across heads.
    flat = x.permute(0, 2, 1, 3).reshape(b * h * s, d)
    pos = pos_3d.repeat(b * h, 1)
    out = pyramidkv_rope.apply_rope_to_flat_k(flat, pos, freqs=table)
    return out.reshape(b, h, s, d).permute(0, 2, 1, 3)


def _max_abs_diff(a, b):
    return (a.float() - b.float()).abs().max().item()


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize(
    "num_frames,height,width,t_start",
    [
        (4, 6, 8, 0),      # small, from the origin
        (4, 6, 8, 17),     # non-zero temporal offset (mid-rollout block)
        (1, 15, 26, 0),    # one latent frame, chunk_t=1 geometry
        (3, 15, 26, 69),   # a late block at 480p-ish aspect
    ],
)
def test_rope_parity_sequential_positions(num_frames, height, width, t_start, fused):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if fused and device == "cpu":
        pytest.skip("fused path needs flash_attn on CUDA")
    torch.manual_seed(0)

    emb = _rcm_embedder(device)
    b, n_heads = 1, 12
    seq = num_frames * height * width
    x = torch.randn(b, seq, n_heads, HEAD_DIM, device=device, dtype=torch.bfloat16)

    freqs = emb.generate_embeddings(
        torch.Size([b, num_frames, height, width, HEAD_DIM]), t_start=t_start
    )
    ref = _rcm_rotate(x, freqs, fused=fused)

    t_indices = torch.arange(t_start, t_start + num_frames, device=device)
    pos_3d = build_pos_3d(t_indices, height, width, device=device)
    table = build_pyramidkv_freq_table(
        HEAD_DIM, max_pos=max(t_start + num_frames, height, width), device=device
    )
    got = _pyramidkv_rotate(x, pos_3d, table)

    # bf16 round-trip through two different float32 kernels; 2^-7 is one bf16 ulp
    # at magnitude 1, and the inputs are unit-normal.
    assert _max_abs_diff(ref, got) < 2e-2, f"max|Δ| = {_max_abs_diff(ref, got)}"


def test_rope_parity_arbitrary_t_indices():
    """The dynamic-RoPE case: non-contiguous, out-of-order anchor times.

    This is what the middle strategies actually produce — cyclic/stride anchors
    carry their own `t`, remapped at readout. If parity only held for contiguous
    ranges the port would be silently wrong exactly where it matters.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(1)

    emb = _rcm_embedder(device)
    b, n_heads, height, width = 1, 12, 6, 8
    t_indices = torch.tensor([0, 6, 12, 31, 7, 3], device=device)
    num_frames = t_indices.numel()
    seq = num_frames * height * width
    x = torch.randn(b, seq, n_heads, HEAD_DIM, device=device, dtype=torch.bfloat16)

    freqs = emb.generate_embeddings(
        torch.Size([b, num_frames, height, width, HEAD_DIM]), t_indices=t_indices
    )
    ref = _rcm_rotate(x, freqs, fused=False)

    pos_3d = build_pos_3d(t_indices, height, width, device=device)
    table = build_pyramidkv_freq_table(
        HEAD_DIM, max_pos=int(t_indices.max()) + 1, device=device
    )
    got = _pyramidkv_rotate(x, pos_3d, table)

    assert _max_abs_diff(ref, got) < 2e-2, f"max|Δ| = {_max_abs_diff(ref, got)}"


def test_rope_parity_float32_is_tight():
    """Same check in fp32, where the tolerance can be near machine precision.

    Separates 'the conventions agree' from 'bf16 rounding is noisy'.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(2)

    emb = _rcm_embedder(device)
    b, n_heads, num_frames, height, width, t_start = 1, 4, 3, 6, 8, 5
    seq = num_frames * height * width
    x = torch.randn(b, seq, n_heads, HEAD_DIM, device=device, dtype=torch.float32)

    freqs = emb.generate_embeddings(
        torch.Size([b, num_frames, height, width, HEAD_DIM]), t_start=t_start
    )
    ref = _rcm_rotate(x, freqs, fused=False)

    t_indices = torch.arange(t_start, t_start + num_frames, device=device)
    pos_3d = build_pos_3d(t_indices, height, width, device=device)
    table = build_pyramidkv_freq_table(
        HEAD_DIM, max_pos=max(t_start + num_frames, height, width), device=device
    )
    got = _pyramidkv_rotate(x, pos_3d, table)

    assert _max_abs_diff(ref, got) < 1e-5, f"max|Δ| = {_max_abs_diff(ref, got)}"


def test_split_widths_match():
    """The split the two sides use must be identical, not merely compatible."""
    c = HEAD_DIM // 2
    pyramidkv_split = [c - 2 * (c // 3), c // 3, c // 3]

    dim_h = HEAD_DIM // 6 * 2
    dim_t = HEAD_DIM - 2 * dim_h
    rcm_split = [dim_t // 2, dim_h // 2, dim_h // 2]

    assert pyramidkv_split == rcm_split == [22, 21, 21]
