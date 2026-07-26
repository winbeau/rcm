"""Per-head ragged attention, mirroring Pyramid-Forcing's varlen contract.

Each ``(batch, head)`` pair becomes one sequence in ``flash_attn_varlen_func``,
so different heads can retain different numbers of KV tokens without any
custom kernel. This is exactly what
``Pyramid-Forcing/wan/modules/attention/core.py::run_varlen`` does; it is
restated here so the rCM port does not have to import Self-Forcing's ``wan``
package.

No mask is applied. For one query block whose cache holds only the past plus
that block, block-causal attention degenerates to full attention — rCM's own
``FlexOrSdpaLocalAttention.forward`` takes the same shortcut
(``mode == "block_causal" and KV <= blocks_to_tokens(q_blk + 1)`` →
``return self.attn(q, k, v)``).
"""
from __future__ import annotations

import torch

try:
    from flash_attn import flash_attn_varlen_func

    FLASH_ATTN_2_AVAILABLE = True
except ImportError:  # pragma: no cover - environment dependent
    flash_attn_varlen_func = None
    FLASH_ATTN_2_AVAILABLE = False


def ragged_attention(
    q: torch.Tensor,
    k_flat: torch.Tensor,
    v_flat: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_k: int,
    *,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Attend `q` `[B, Lq, H, D]` against a ragged per-(b, h) KV set.

    Args:
        q: queries, already RoPE-rotated.
        k_flat / v_flat: `[total_k, D]`, concatenated over `(b, h)` in the order
            `b * H + h`.
        cu_seqlens_k: `[B*H + 1]` int32 prefix sums of each sequence's length.
        max_seqlen_k: longest per-head cache length.

    Returns:
        `[B, Lq, H, D]` in the same dtype as `q`.
    """
    if not FLASH_ATTN_2_AVAILABLE:
        raise RuntimeError("flash_attn is required for the ragged path")

    b, lq, h, d = q.shape
    if cu_seqlens_k.numel() != b * h + 1:
        raise ValueError(
            f"cu_seqlens_k must have B*H+1 = {b * h + 1} entries, got {cu_seqlens_k.numel()}"
        )

    out_dtype = q.dtype
    compute_dtype = out_dtype if out_dtype in (torch.float16, torch.bfloat16) else torch.bfloat16

    q_flat = q.transpose(1, 2).reshape(b * h * lq, d).unsqueeze(1).to(compute_dtype)
    k_in = k_flat.unsqueeze(1).to(compute_dtype)
    v_in = v_flat.unsqueeze(1).to(compute_dtype)

    cu_seqlens_q = torch.arange(
        0, (b * h + 1) * lq, step=lq, dtype=torch.int32, device=q.device
    )

    out = flash_attn_varlen_func(
        q=q_flat,
        k=k_in,
        v=v_in,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k.to(torch.int32),
        max_seqlen_q=lq,
        max_seqlen_k=int(max_seqlen_k),
        softmax_scale=softmax_scale,
        causal=False,
    )
    if isinstance(out, tuple):  # FA3 returns (out, lse)
        out = out[0]
    return out.reshape(b, h, lq, d).permute(0, 2, 1, 3).to(out_dtype)


def pack_dense_kv(k: torch.Tensor, v: torch.Tensor):
    """Pack a dense `[B, S, H, D]` KV into the ragged layout, keeping everything.

    The degenerate case used by the G0 identity gate: every head keeps the same
    tokens, so the ragged path must reproduce dense attention exactly.
    """
    b, s, h, d = k.shape
    k_flat = k.permute(0, 2, 1, 3).reshape(b * h * s, d)
    v_flat = v.permute(0, 2, 1, 3).reshape(b * h * s, d)
    cu_seqlens_k = torch.arange(
        0, (b * h + 1) * s, step=s, dtype=torch.int32, device=k.device
    )
    return k_flat, v_flat, cu_seqlens_k, s
