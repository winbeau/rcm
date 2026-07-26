"""Is the per-head KV cache actually bounded?

Peak sampling memory for the pyramid arm grows super-linearly with clip length
(0.040 -> 0.071 -> 0.138 GiB per latent frame at 72/121/241 frames), so its
advantage over the dense baseline peaks at 3.5x and falls back to 2.5x by 481
frames. sink + stride(cap) + cyclic(period x bucket) + merge(cap) + recent is
supposed to be a constant number of frames per head, so something is not
respecting its bound.

This drives one layer's cache directly -- no model, no GPU time wasted on a
rollout -- and reports the retained length per head as the clip grows.

    PYTHONPATH=. python experiments/pyramid_port/probe_cache_growth.py
"""
from __future__ import annotations

import argparse

import torch

from rcm.utils.pyramid_attention import PyramidSpec


def probe(spec: PyramidSpec, num_frames: int, device="cuda", dtype=torch.bfloat16):
    from rcm.pyramidkv import AdaptiveKVCache

    frame_tokens = spec.frame_seq_length
    n_heads = spec.num_heads
    cache = AdaptiveKVCache(
        spec.kv_config(),
        batch_size=1,
        num_heads=n_heads,
        head_dim=spec.head_dim,
        layer_idx=0,
        tail_len=spec.default_capacity,
    )
    grid = torch.tensor([[1, spec.latent_h, spec.latent_w]], dtype=torch.long, device=device)

    torch.manual_seed(0)
    trace = []
    for t in range(num_frames):
        k = torch.randn(1, frame_tokens, n_heads, spec.head_dim, device=device, dtype=dtype)
        v = torch.randn_like(k)
        cache.update(k, v, current_start=t * frame_tokens, grid_sizes=grid,
                     cache_update_mode="clean")
        if (t + 1) % 24 == 0 or t == num_frames - 1:
            _, _, cu, _, _ = cache.get_flat_kv_and_pos()
            lens = (cu[1:] - cu[:-1]).tolist()
            frames = [l / frame_tokens for l in lens]
            # The readout being flat does not prove the cache holds nothing else:
            # internal per-strategy storage can grow while the selected view stays
            # constant. Track the allocator to separate the two.
            held = torch.cuda.memory_allocated() / 2**30
            peak = torch.cuda.max_memory_allocated() / 2**30
            trace.append((t + 1, min(frames), sum(frames) / len(frames), max(frames),
                          sum(lens), held, peak))
        del k, v
    return trace


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--labels", default="assets/rcm-head-labels-thp6.4-ths0.8.csv")
    p.add_argument("--frames", type=int, default=241)
    p.add_argument("--latent-h", type=int, default=30)
    p.add_argument("--latent-w", type=int, default=52)
    args = p.parse_args()

    spec = PyramidSpec(
        num_layers=30, num_heads=12, head_dim=128,
        latent_h=args.latent_h, latent_w=args.latent_w,
        labels_csv=args.labels, max_frames=max(256, args.frames),
    )
    print(f"policies: {spec.composition_summary()}")
    print(f"frame_seq_length={spec.frame_seq_length}  "
          f"sink osc/stable={spec.osc_sink_frames}/{spec.stable_sink_frames}  "
          f"recent={spec.recent_frames}  stride cap={spec.stride_capacity}  "
          f"cyclic {spec.cyclic_period}x{spec.cyclic_bucket_cap}  "
          f"merge cap={spec.merge_capacity}")
    print()
    print(f"{'frames fed':>11}{'mean f/head':>13}{'max f/head':>12}"
          f"{'readout MiB':>13}{'held GiB':>11}{'peak GiB':>11}")
    for fed, lo, mean, hi, tok, held, peak in probe(spec, args.frames):
        mib = tok * 128 * 2 * 2 / 2**20  # K and V, bf16
        print(f"{fed:>11}{mean:>13.1f}{hi:>12.1f}{mib:>13.0f}{held:>11.2f}{peak:>11.2f}")

    print("\n'readout' is what get_flat_kv_and_pos returns for ONE layer; 'held' is "
          "everything this one cache keeps alive. If held climbs while readout stays "
          "flat, the strategies are retaining frames they never serve.")


if __name__ == "__main__":
    main()
