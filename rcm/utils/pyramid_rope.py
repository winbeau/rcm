"""Bridge between rCM's RoPE representation and pyramidkv's.

rCM (`rcm/networks/wan2pt1.py::VideoRopePosition3DEmb.generate_embeddings`)
returns **angles** laid out per token: `[T*H*W, head_dim // 2]`, formed as
`cat([freqs_t, freqs_h, freqs_w], dim=-1)`, and `rcm/utils/rope.py::apply_rope`
consumes them with `interleaved=True`.

pyramidkv (`pyramidkv/rope.py::apply_rope_to_flat_k`) wants a **complex table
indexed by position**: `[max_pos, head_dim // 2]`, split along dim 1 into
`[c - 2*(c//3), c//3, c//3]` and indexed by `pos_3d[:, 0/1/2]` respectively.

The two agree on axis order (t, h, w), on the interleaved pairing, and — for
head_dim 128 — on the split sizes (22 / 21 / 21). This module only changes the
representation; `test_g0_rope_parity.py` asserts the two produce the same
rotation.
"""
from __future__ import annotations

import torch


def rope_axis_frequencies(head_dim: int, h_ntk: float = 1.0, w_ntk: float = 1.0, t_ntk: float = 1.0):
    """Per-axis base frequencies, mirroring `VideoRopePosition3DEmb`.

    Returns `(temporal, h_spatial, w_spatial)` 1-D tensors whose lengths sum to
    `head_dim // 2`.
    """
    dim_h = head_dim // 6 * 2
    dim_w = dim_h
    dim_t = head_dim - 2 * dim_h
    assert head_dim == dim_h + dim_w + dim_t, f"bad dim: {head_dim}"

    dim_spatial_range = torch.arange(0, dim_h, 2, dtype=torch.float32)[: dim_h // 2] / dim_h
    dim_temporal_range = torch.arange(0, dim_t, 2, dtype=torch.float32)[: dim_t // 2] / dim_t

    # NTK factors are applied exactly as generate_embeddings does: the caller
    # passes the already-exponentiated ratios that the module stores.
    temporal = 1.0 / ((10000.0 * t_ntk) ** dim_temporal_range)
    h_spatial = 1.0 / ((10000.0 * h_ntk) ** dim_spatial_range)
    w_spatial = 1.0 / ((10000.0 * w_ntk) ** dim_spatial_range)
    return temporal, h_spatial, w_spatial


def build_pyramidkv_freq_table(
    head_dim: int,
    max_pos: int,
    *,
    device=None,
    dtype=torch.complex64,
    h_ntk: float = 1.0,
    w_ntk: float = 1.0,
    t_ntk: float = 1.0,
) -> torch.Tensor:
    """`[max_pos, head_dim // 2]` complex table in pyramidkv's layout.

    Column block `[0 : dim_t/2]` is the temporal axis, then h, then w — the
    same order and the same widths that `apply_rope_to_flat_k` splits out.
    Row `p` holds the frequencies for *position value* `p` on every axis; the
    caller indexes each block with a different component of `pos_3d`.
    """
    temporal, h_spatial, w_spatial = rope_axis_frequencies(head_dim, h_ntk, w_ntk, t_ntk)
    pos = torch.arange(max_pos, dtype=torch.float32)
    angles = torch.cat(
        [
            torch.outer(pos, temporal),
            torch.outer(pos, h_spatial),
            torch.outer(pos, w_spatial),
        ],
        dim=1,
    )
    table = torch.polar(torch.ones_like(angles), angles).to(dtype)
    return table if device is None else table.to(device)


def build_pos_3d(t_indices: torch.Tensor, height: int, width: int, device=None) -> torch.Tensor:
    """`[T*H*W, 3]` of `(t, y, x)` in rCM's token order.

    `generate_embeddings` lays tokens out t-major then h then w:
    `t_token = repeat_interleave(t_indices, H*W)`,
    `h_idx = arange(H).repeat_interleave(W).repeat(T)`,
    `w_idx = arange(W).repeat(H).repeat(T)`.
    """
    t_indices = torch.as_tensor(t_indices, dtype=torch.long)
    if device is not None:
        t_indices = t_indices.to(device)
    dev = t_indices.device
    num_frames = t_indices.numel()

    t = t_indices.repeat_interleave(height * width)
    h = torch.arange(height, device=dev).repeat_interleave(width).repeat(num_frames)
    w = torch.arange(width, device=dev).repeat(height).repeat(num_frames)
    return torch.stack([t, h, w], dim=1)
