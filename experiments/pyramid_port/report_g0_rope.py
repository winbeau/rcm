"""Print the actual RoPE parity error magnitudes behind G0(a)'s pass/fail."""
import sys
import torch

from rcm.utils.pyramid_rope import build_pos_3d, build_pyramidkv_freq_table
from experiments.pyramid_port.test_g0_rope_parity import (
    HEAD_DIM,
    _max_abs_diff,
    _pyramidkv_rotate,
    _rcm_embedder,
    _rcm_rotate,
)

CASES = [
    ("sequential t0=0", (4, 6, 8, 0), torch.bfloat16, False),
    ("sequential t0=0", (4, 6, 8, 0), torch.bfloat16, True),
    ("sequential t0=17", (4, 6, 8, 17), torch.bfloat16, True),
    ("chunk_t=1 geometry", (1, 15, 26, 0), torch.bfloat16, True),
    ("late block t0=69", (3, 15, 26, 69), torch.bfloat16, True),
    ("sequential t0=5", (3, 6, 8, 5), torch.float32, False),
    ("sequential t0=5", (3, 6, 8, 5), torch.float64, False),
]


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    emb = _rcm_embedder(dev)
    print(f"{'case':<24}{'dtype':>10}{'fused':>7}{'max|d|':>13}{'rel':>11}")
    for name, (T, H, W, t0), dt, fused in CASES:
        torch.manual_seed(0)
        x = torch.randn(1, T * H * W, 12, HEAD_DIM, device=dev, dtype=dt)
        freqs = emb.generate_embeddings(torch.Size([1, T, H, W, HEAD_DIM]), t_start=t0)
        try:
            ref = _rcm_rotate(x, freqs, fused=fused)
        except Exception as exc:
            print(f"{name:<24}{str(dt).split('.')[-1]:>10}{str(fused):>7}   skipped: {type(exc).__name__}")
            continue
        t_idx = torch.arange(t0, t0 + T, device=dev)
        pos = build_pos_3d(t_idx, H, W, device=dev)
        table = build_pyramidkv_freq_table(HEAD_DIM, max_pos=max(t0 + T, H, W), device=dev)
        got = _pyramidkv_rotate(x, pos, table)
        d = _max_abs_diff(ref, got)
        rel = d / ref.float().abs().max().item()
        print(f"{name:<24}{str(dt).split('.')[-1]:>10}{str(fused):>7}{d:>13.3e}{rel:>11.2e}")

    # dynamic-RoPE case: non-contiguous, out-of-order anchor times
    torch.manual_seed(1)
    T_idx = torch.tensor([0, 6, 12, 31, 7, 3], device=dev)
    H, W = 6, 8
    x = torch.randn(1, T_idx.numel() * H * W, 12, HEAD_DIM, device=dev, dtype=torch.float32)
    freqs = emb.generate_embeddings(torch.Size([1, T_idx.numel(), H, W, HEAD_DIM]), t_indices=T_idx)
    ref = _rcm_rotate(x, freqs, fused=False)
    pos = build_pos_3d(T_idx, H, W, device=dev)
    table = build_pyramidkv_freq_table(HEAD_DIM, max_pos=int(T_idx.max()) + 1, device=dev)
    got = _pyramidkv_rotate(x, pos, table)
    d = _max_abs_diff(ref, got)
    print(f"{'out-of-order t_indices':<24}{'float32':>10}{'False':>7}{d:>13.3e}"
          f"{d / ref.abs().max().item():>11.2e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
