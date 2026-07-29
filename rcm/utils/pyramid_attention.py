"""Head-aware KV cache for rCM, backed by the vendored `pyramidkv`.

Drops into `DistributedAttention` through the `manages_kv_cache` /
`accepts_attn_ctx` hook that `rcm/utils/a2a_cp.py` already reads but that no
class in rCM sets. Nothing in `KVCache`, `_materialize_kv` or
`DistributedAttention` has to change.

Two wiring requirements on the caller (see `docs/005` §3.2):

* ``RopeCache.cached_k_rotated`` must be **False**, so `a2a_cp` hands this module
  the raw pre-RoPE key. RoPE is re-applied at readout from each token's stored
  `(t, y, x)`, which is what lets the middle strategies remap anchor time.
* ``fast_infer`` must be **off**. It routes through `KVCache.write_transient`,
  which assumes a dense contiguous buffer.

Per-head variable cache lengths need no custom kernel: every `(batch, head)`
pair becomes one sequence in `flash_attn_varlen_func`, exactly as
Pyramid-Forcing's `wan/modules/attention/core.py::run_varlen` does.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import nn

# PYRAMID_MEM_TRACE=1 prints a per-block allocator breakdown for layer 0. Off by
# default: it forces a sync-free read but still costs a print per sampled block.
_MEM_TRACE = os.environ.get("PYRAMID_MEM_TRACE") == "1"
_MEM_TRACE_EVERY = int(os.environ.get("PYRAMID_MEM_TRACE_EVERY", "40"))
# PYRAMID_MEM_CENSUS=1 additionally walks every live CUDA tensor in the process
# and prints the largest (shape, dtype) groups. The four-step breakdown showed
# `enter` climbing ~0.099 GiB/block while all four steps stayed flat, so the
# growth is held by something the breakdown never touches. A census does not
# need to guess where.
_MEM_CENSUS = os.environ.get("PYRAMID_MEM_CENSUS") == "1"

from rcm.pyramidkv import AdaptiveKVCache, PyramidKVConfig, build_compositions
from rcm.utils.kv_cache import AttnContext, KVCacheMode
from rcm.utils.pyramid_rope import build_pyramidkv_freq_table


_MEM_CENSUS_TOP = int(os.environ.get("PYRAMID_MEM_CENSUS_TOP", "40"))


def cuda_tensor_census(top: int = _MEM_CENSUS_TOP) -> list[tuple[str, int, float]]:
    """Every live CUDA tensor in the process, grouped by (shape, dtype).

    Deduplicates by storage pointer, so a view and its base are counted once --
    otherwise `dynamic_k[i]` (a view into `_dyn_store_k[i]`) would double-count
    the whole cache.

    The first version printed only the top 12, which summed to 7.44 GiB against
    an `enter` of 10.58 -- the growth was in the 3.14 GiB tail, invisible.
    A TOTAL and a REST row keep that from happening again: growth that hides in
    the tail still shows up as a moving REST.

    Returns ``(label, count, gib)`` sorted by size, with TOTAL and REST last.
    """
    import gc
    from collections import defaultdict

    groups: dict[str, list[int]] = defaultdict(list)
    seen: set[int] = set()
    for obj in gc.get_objects():
        try:
            if not torch.is_tensor(obj) or not obj.is_cuda:
                continue
            ptr = obj.untyped_storage().data_ptr()
        except Exception:
            continue
        if ptr in seen:
            continue
        seen.add(ptr)
        nbytes = obj.untyped_storage().nbytes()
        groups[f"{tuple(obj.shape)}:{obj.dtype}".replace("torch.", "")].append(nbytes)

    rows = [(label, len(sizes), sum(sizes) / 2 ** 30) for label, sizes in groups.items()]
    rows.sort(key=lambda r: -r[2])
    total_gib = sum(r[2] for r in rows)
    total_n = sum(r[1] for r in rows)
    head = rows[:top]
    rest_gib = total_gib - sum(r[2] for r in head)
    rest_n = total_n - sum(r[1] for r in head)
    head.append((f"REST[{len(rows) - len(head)} groups]", rest_n, rest_gib))
    head.append((f"TOTAL[{len(rows)} groups]", total_n, total_gib))
    return head

try:
    from flash_attn import flash_attn_varlen_func

    FLASH_ATTN_2_AVAILABLE = True
except ImportError:  # pragma: no cover - environment dependent
    flash_attn_varlen_func = None
    FLASH_ATTN_2_AVAILABLE = False


# ---------------------------------------------------------------------------
# ragged attention
# ---------------------------------------------------------------------------


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

    `k_flat` / `v_flat` are `[total_k, D]`, concatenated over `(b, h)` in the
    order `b * H + h`; `cu_seqlens_k` is the `[B*H + 1]` prefix sum.

    No mask is applied. For one query block whose cache holds only the past plus
    that block, block-causal attention degenerates to full attention — rCM's own
    `FlexOrSdpaLocalAttention.forward` takes the same shortcut.
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
    cu_seqlens_q = torch.arange(
        0, (b * h + 1) * lq, step=lq, dtype=torch.int32, device=q.device
    )

    out = flash_attn_varlen_func(
        q=q_flat,
        k=k_flat.unsqueeze(1).to(compute_dtype),
        v=v_flat.unsqueeze(1).to(compute_dtype),
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
    """Pack a dense `[B, S, H, D]` KV into the ragged layout, keeping everything."""
    b, s, h, d = k.shape
    k_flat = k.permute(0, 2, 1, 3).reshape(b * h * s, d)
    v_flat = v.permute(0, 2, 1, 3).reshape(b * h * s, d)
    cu_seqlens_k = torch.arange(
        0, (b * h + 1) * s, step=s, dtype=torch.int32, device=k.device
    )
    return k_flat, v_flat, cu_seqlens_k, s


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


@dataclass
class PyramidSpec:
    """Everything the per-layer attention modules share.

    `latent_h` / `latent_w` are the **patchified** grid, i.e. after the model's
    2x2 spatial patch, so `latent_h * latent_w` equals `BlockPattern.frame_tokens`
    (1560 at 480p 16:9). They cannot be recovered from `attn_meta`, which only
    carries the token count, so they are supplied here and cross-checked at the
    first forward.
    """

    num_layers: int
    num_heads: int
    head_dim: int
    latent_h: int
    latent_w: int
    labels_csv: Optional[str] = None
    max_frames: int = 256

    # Middle-strategy parameters. Defaults mirror configs/pyramid-forcing.yaml.
    cyclic_period: int = 6
    cyclic_bucket_cap: int = 4
    stride_interval: int = 6
    stride_capacity: int = 4
    merge_patch_size: int = 2
    merge_capacity: int = 4
    osc_sink_frames: int = 1
    stable_sink_frames: int = 3
    recent_frames: int = 4

    # Dynamic-RoPE clamp. The paper's 18/21 is pinned to Self-Forcing's 21-frame
    # training range; rCM's VideoRopePosition3DEmb uses len_t=32, so leaving the
    # Self-Forcing numbers here would clamp anchors into the wrong window.
    sink_time_clamp_min: int = 0
    sink_time_clamp_max: int = 0
    sink_grid_decoupling: bool = False

    default_capacity: int = 32760
    _config: PyramidKVConfig | None = field(default=None, repr=False)

    @property
    def frame_seq_length(self) -> int:
        return self.latent_h * self.latent_w

    def kv_config(self) -> PyramidKVConfig:
        """Build (once) the shared PyramidKVConfig with per-head compositions."""
        if self._config is not None:
            return self._config

        config = PyramidKVConfig(
            config_path=self.labels_csv,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            default_capacity=self.default_capacity,
            frame_seq_length=self.frame_seq_length,
        )
        compositions = build_compositions(
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            capacities=config.capacity_map,
            csv_path=self.labels_csv,
            # Label -> strategy routing, per the paper's taxonomy:
            #   -1 Wave   -> cyclic        1 Anchor -> stride       2 Veil -> merge
            cyclic_enabled=True,
            cyclic_period=self.cyclic_period,
            cyclic_bucket_cap=self.cyclic_bucket_cap,
            cyclic_osc_only=True,
            cyclic_dynamic_rope=True,
            stride_enabled=True,
            stride_interval=self.stride_interval,
            stride_capacity=self.stride_capacity,
            stride_dynamic_rope=True,
            merge_enabled=True,
            merge_patch_size=self.merge_patch_size,
            merge_capacity=self.merge_capacity,
            merge_dynamic_rope=True,
            lag_enabled=False,
            osc_sink_frames=self.osc_sink_frames,
            stable_sink_frames=self.stable_sink_frames,
            recent_frames=self.recent_frames,
            stable_recent_frames=self.recent_frames,
            label_phase_bucket_map={"-1": self.cyclic_bucket_cap, "1": 0, "2": 0},
            label_stride_enabled_map={"1": True, "2": False, "-1": False},
            label_stride_interval_map={"1": self.stride_interval},
            label_merge_enabled_map={"-1": False, "1": False, "2": True},
            label_merge_patch_size_map={"2": self.merge_patch_size},
            label_merge_capacity_map={"2": self.merge_capacity},
            label_sink_frames_map={
                "-1": self.osc_sink_frames,
                "1": self.stable_sink_frames,
                "2": self.stable_sink_frames,
            },
            label_recent_frames_map={
                "-1": self.recent_frames,
                "1": self.recent_frames,
                "2": self.recent_frames,
            },
        )
        config.compositions = compositions
        config.policies = compositions
        self._config = config
        return config

    def composition_summary(self) -> dict[str, int]:
        """Count of heads per policy type — cheap sanity check that labels loaded."""
        counts: dict[str, int] = {}
        for row in self.kv_config().compositions:
            for comp in row:
                key = type(comp.middle_strategies[0]).__name__ if comp.has_middle else "none"
                counts[key] = counts.get(key, 0) + 1
        return counts


def retain_everything_spec(num_layers, num_heads, head_dim, latent_h, latent_w, max_frames):
    """A spec whose composition drops nothing — the G0 identity configuration."""
    spec = PyramidSpec(
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        latent_h=latent_h,
        latent_w=latent_w,
        max_frames=max_frames,
    )
    config = PyramidKVConfig(
        config_path=None,
        num_layers=num_layers,
        num_heads=num_heads,
        default_capacity=max_frames * spec.frame_seq_length * 4,
        frame_seq_length=spec.frame_seq_length,
    )
    config.compositions = build_compositions(
        num_layers=num_layers,
        num_heads=num_heads,
        capacities=config.capacity_map,
        csv_path=None,
        cyclic_enabled=False,
        lag_enabled=False,
        stride_enabled=False,
        merge_enabled=False,
        osc_sink_frames=0,
        stable_sink_frames=0,
        recent_frames=max_frames + 8,
        stable_recent_frames=max_frames + 8,
    )
    config.policies = config.compositions
    spec._config = config
    return spec


# ---------------------------------------------------------------------------
# the local_attn module
# ---------------------------------------------------------------------------


class PyramidLocalAttention(nn.Module):
    """One per transformer layer; owns that layer's per-head KV cache.

    rCM builds a fresh `local_attn` inside every `WanSelfAttention`, so the
    layer index is implicit — but a single instance is reused across the
    conditional and unconditional CFG streams, which have *separate* rCM
    `KVCache` objects. Those objects are used as the stream identity, so each
    stream gets its own `AdaptiveKVCache`.
    """

    manages_kv_cache = True
    accepts_attn_ctx = True

    def __init__(self, spec: PyramidSpec, fallback: nn.Module | None = None):
        super().__init__()
        self.spec = spec
        self.fallback = fallback
        self._caches: dict[int, AdaptiveKVCache] = {}
        self._last_block: dict[int, int] = {}
        self._freqs: torch.Tensor | None = None
        self._checked_grid = False

    # -- internals ----------------------------------------------------------

    def _freq_table(self, device, dtype=torch.complex64) -> torch.Tensor:
        if self._freqs is None or self._freqs.device != device:
            max_pos = max(self.spec.max_frames, self.spec.latent_h, self.spec.latent_w)
            self._freqs = build_pyramidkv_freq_table(
                self.spec.head_dim, max_pos=max_pos, device=device, dtype=dtype
            )
        return self._freqs

    def _cache_for(self, attn_ctx: AttnContext, batch_size: int, block_idx: int) -> AdaptiveKVCache:
        key = id(attn_ctx.kv_cache) if attn_ctx.kv_cache is not None else 0
        # Block 0 arriving after anything else means a new clip on this stream.
        if block_idx == 0 and self._last_block.get(key) != 0:
            self._caches.pop(key, None)
        if key not in self._caches:
            self._caches[key] = AdaptiveKVCache(
                self.spec.kv_config(),
                batch_size=batch_size,
                num_heads=self.spec.num_heads,
                head_dim=self.spec.head_dim,
                layer_idx=attn_ctx.layer_idx,
                tail_len=self.spec.default_capacity,
                sink_grid_decoupling=self.spec.sink_grid_decoupling,
                sink_time_clamp_min=self.spec.sink_time_clamp_min,
                sink_time_clamp_max=self.spec.sink_time_clamp_max,
            )
        self._last_block[key] = block_idx
        return self._caches[key]

    def reset(self) -> None:
        """Drop every stream's cache. Call between clips if block indices repeat."""
        self._caches.clear()
        self._last_block.clear()

    # -- forward ------------------------------------------------------------

    def forward(self, q, k, v, attn_ctx: Optional[AttnContext] = None, attn_meta=None, **_ignored):
        if attn_ctx is None or attn_ctx.mode == KVCacheMode.DISABLED:
            return self._fallback(q, k, v, attn_meta)
        if attn_meta is None or getattr(attn_meta, "pattern", None) is None:
            raise ValueError(
                "PyramidLocalAttention needs attn_meta.pattern to locate the current "
                "block; got attn_meta=None. The bidirectional/teacher-forcing paths "
                "should run with KVCacheMode.DISABLED instead."
            )

        rope = attn_ctx.rope
        if rope is not None and rope.cached_k_rotated:
            raise ValueError(
                "PyramidLocalAttention requires pre-RoPE keys: set "
                "RopeCache.cached_k_rotated=False. With post-RoPE caching the key "
                "arriving here is already rotated and anchor re-timing is impossible."
            )
        if attn_ctx.fast_infer:
            raise ValueError(
                "PyramidLocalAttention is incompatible with fast_infer, which routes "
                "through KVCache.write_transient and assumes a dense buffer."
            )

        pattern = attn_meta.pattern
        block_idx = int(getattr(attn_meta, "q_block_offset", attn_ctx.q_block_idx))
        self._check_grid(pattern)

        b, l_new, h, d = k.shape
        cache = self._cache_for(attn_ctx, b, block_idx)

        grid_sizes = torch.tensor(
            [[pattern.block_size(block_idx), self.spec.latent_h, self.spec.latent_w]] * b,
            dtype=torch.long,
            device=k.device,
        )
        # APPEND is the once-per-chunk commit; READONLY is a denoising step that
        # must see the current block without committing it. That is exactly
        # pyramidkv's clean/noisy double pass.
        update_mode = "clean" if attn_ctx.mode == KVCacheMode.APPEND else "default"

        trace = _MEM_TRACE and attn_ctx.layer_idx == 0 and block_idx % _MEM_TRACE_EVERY == 0
        m0 = torch.cuda.memory_allocated() if trace else 0

        cache.update(
            k,
            v,
            current_start=pattern.blocks_to_tokens(block_idx),
            grid_sizes=grid_sizes,
            cache_update_mode=update_mode,
        )
        m1 = torch.cuda.memory_allocated() if trace else 0

        k_flat, v_flat, cu_seqlens_k, max_seqlen_k, pos = cache.get_flat_kv_and_pos()
        m2 = torch.cuda.memory_allocated() if trace else 0

        if attn_ctx.retained_k_observer is not None:
            attn_ctx.retained_k_observer.record(
                layer_idx=attn_ctx.layer_idx,
                batch_size=b,
                num_heads=h,
                block_idx=block_idx,
                mode=getattr(attn_ctx.mode, "value", str(attn_ctx.mode)),
                pass_name=attn_ctx.pass_name,
                stream=attn_ctx.stream_name,
                denoise_step=attn_ctx.denoise_step,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_k=max_seqlen_k,
                query_tokens=q.shape[1],
                frame_seq_length=pattern.frame_tokens,
                dense_prefix_tokens=pattern.blocks_to_tokens(block_idx + 1),
            )

        k_flat = cache.apply_rope_to_flat_k(k_flat, pos, freqs=self._freq_table(k.device))
        m3 = torch.cuda.memory_allocated() if trace else 0

        out = ragged_attention(q, k_flat, v_flat, cu_seqlens_k, max_seqlen_k)

        if trace:
            m4 = torch.cuda.memory_allocated()
            g = 2 ** 30
            print(
                f"MEMTRACE blk={block_idx:>4} mode={update_mode:<7} "
                f"kv_tok={int(cu_seqlens_k[-1]):>8} maxk={int(max_seqlen_k):>6} "
                f"| enter={m0/g:7.3f} update=+{(m1-m0)/g:7.3f} "
                f"readout=+{(m2-m1)/g:7.3f} rope=+{(m3-m2)/g:7.3f} "
                f"attn=+{(m4-m3)/g:7.3f} held={m4/g:7.3f}",
                flush=True,
            )
            if _MEM_CENSUS and update_mode == "clean":
                for label, n, gib in cuda_tensor_census():
                    print(f"MEMCENSUS blk={block_idx:>4} {gib:8.3f} GiB  n={n:>7}  {label}",
                          flush=True)
        return out

    # -- helpers ------------------------------------------------------------

    def _check_grid(self, pattern) -> None:
        if self._checked_grid:
            return
        expected = self.spec.latent_h * self.spec.latent_w
        if int(pattern.frame_tokens) != expected:
            raise ValueError(
                f"PyramidSpec grid {self.spec.latent_h}x{self.spec.latent_w} = {expected} "
                f"tokens disagrees with BlockPattern.frame_tokens={pattern.frame_tokens}. "
                "latent_h/latent_w must be the patchified grid."
            )
        self._checked_grid = True

    def _fallback(self, q, k, v, attn_meta):
        if self.fallback is None:
            from rcm.utils.attention import attention

            return attention(q, k, v)
        return self.fallback(q, k, v, attn_meta=attn_meta)
