"""Print the actual error magnitudes behind G0(b1), plus a kernel-noise floor.

The parity tests use a loose relative tolerance because they compare two
different attention kernels (flash-attn varlen vs FlexAttention) in bf16. That
tolerance is only meaningful next to the noise floor of running the *same*
comparison with no ragged path involved -- if the ragged error is at the floor,
the plumbing is exact and the tolerance is just kernel jitter.
"""
import torch

from rcm.utils.blockmask import AttnMaskSpec, BlockPattern, FlexOrSdpaLocalAttention
from experiments.pyramid_port.ragged_attention import pack_dense_kv, ragged_attention
from experiments.pyramid_port.test_g0b_plumbing_parity import _rotate_full, _setup, HEAD_DIM


def rel(a, b):
    d = (a.float() - b.float()).abs().max().item()
    return d, d / a.float().abs().max().item()


def main():
    print(f"{'case':<44}{'max|d|':>12}{'rel':>11}")
    for num_frames, height, width, chunk in [(6, 6, 8, 1), (6, 6, 8, 3), (9, 5, 7, 3)]:
        dev, q, k, v, emb, pattern, ft = _setup(num_frames, height, width, chunk)
        q_rot = _rotate_full(emb, q, num_frames, height, width)
        k_rot = _rotate_full(emb, k, num_frames, height, width)
        local = FlexOrSdpaLocalAttention().to(dev)
        spec = AttnMaskSpec(mode="block_causal", pattern=pattern, q_block_offset=0)
        ref = local(q_rot, k_rot, v, attn_meta=spec)

        got = torch.empty_like(ref)
        for blk in range(num_frames // chunk):
            lo, hi = blk * chunk * ft, (blk + 1) * chunk * ft
            kf, vf, cu, mx = pack_dense_kv(k_rot[:, :hi], v[:, :hi])
            got[:, lo:hi] = ragged_attention(q_rot[:, lo:hi], kf, vf, cu, mx)
        d, r = rel(ref, got)
        print(f"{f'ragged rollout T={num_frames} chunk={chunk}':<44}{d:>12.3e}{r:>11.2e}")

        # Noise floor: same two kernels, same dense inputs, no ragged packing.
        # Whatever separates them here is kernel/dtype jitter, not the port.
        ref_full = local(q_rot, k_rot, v, attn_meta=AttnMaskSpec(mode="none"))
        kf, vf, cu, mx = pack_dense_kv(k_rot, v)
        got_full = ragged_attention(q_rot, kf, vf, cu, mx)
        d, r = rel(ref_full, got_full)
        print(f"{f'  floor: dense flex vs varlen (no mask)':<44}{d:>12.3e}{r:>11.2e}")

    # fp32 inputs: ragged path still computes in bf16 internally, so this
    # isolates how much of the gap is the mandatory half-precision cast.
    dev, q, k, v, emb, pattern, ft = _setup(6, 6, 8, 1, dtype=torch.float32, seed=21)
    q_rot = _rotate_full(emb, q, 6, 6, 8)
    k_rot = _rotate_full(emb, k, 6, 6, 8)
    local = FlexOrSdpaLocalAttention().to(dev)
    ref = local(q_rot, k_rot, v, attn_meta=AttnMaskSpec(mode="none"))
    kf, vf, cu, mx = pack_dense_kv(k_rot, v)
    got = ragged_attention(q_rot, kf, vf, cu, mx)
    d, r = rel(ref, got)
    print(f"{'fp32 in, bf16 ragged compute':<44}{d:>12.3e}{r:>11.2e}")


if __name__ == "__main__":
    main()
