"""
Extrapolation-only causal inference script for Wan2.1 T2V.

This keeps the shared causal stack on the original prefix-cache path and
implements bounded-memory extrapolation policies entirely at inference time.

Supported methods:
  - none:           unbounded prefix cache (baseline)
  - sliding_window: FIFO eviction of oldest blocks
  - rolling_sink:   Rolling Sink (Li et al., 2026) -- within-duration bank with
                    forward/reverse content cycling and sliding temporal indices
  - infinity_rope:  Infinity-RoPE (Yesiltepe et al., 2025) -- Block-Relativistic
                    RoPE with KV Flush and RoPE Cut
  - deep_forcing:   Deep Forcing (Yi et al., 2025) -- Deep Sink with token-level
                    Participative Compression
  - memrope:        MemRoPE (Kim et al., 2026) -- dual EMA memory tokens with
                    compact block-relative RoPE indices (pre-RoPE caching)
  - relax_forcing:  Relax Forcing (Zhao et al., 2026) -- structured Sink + History
                    + Tail with cosine similarity scoring (post-RoPE caching)

Usage (single prompt, sliding window):
    python -m rcm.inference.wan2pt1_t2v_causal_extrapolation_infer \\
        --distilled --dit_path path/to/distilled.pth \\
        --extrapolation_method sliding_window --window_blocks 6

Usage (single prompt, Infinity-RoPE):
    python -m rcm.inference.wan2pt1_t2v_causal_extrapolation_infer \\
        --distilled --dit_path path/to/distilled.pth \\
        --extrapolation_method infinity_rope --f_limit 21

Usage (multi-prompt with scene cuts, Infinity-RoPE):
    python -m rcm.inference.wan2pt1_t2v_causal_extrapolation_infer \\
        --distilled --dit_path path/to/distilled.pth \\
        --extrapolation_method infinity_rope --rope_cut_delta 21 \\
        --prompt "a man standing in a park[5s] | a man jumping[10s#] | a man sitting[10s#]"

    Multi-prompt syntax:
      "action_prompt[Ns]"     -- segment with absolute duration N seconds
      "|"                     -- separator between segments
      "#" inside brackets     -- triggers KV Flush + RoPE Cut (scene transition)
    Total video length is the sum of all segment durations (--num_frames is
    ignored).  Segments without duration or separators are single-prompt mode.
"""

import argparse
import math
import os
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple, Union

import torch
from einops import rearrange, repeat
from tqdm import tqdm

from imaginaire.lazy_config import LazyCall as L, LazyDict, instantiate
from imaginaire.utils import log
from imaginaire.utils.io import save_image_or_video

from rcm.datasets.utils import VIDEO_RES_SIZE_INFO
from rcm.networks.wan2pt1 import WanModel
from rcm.tokenizers.wan2pt1 import Wan2pt1VAEInterface
from rcm.utils.blockmask import AttnMaskSpec, BlockPattern
from rcm.utils.kv_cache import CausalInferenceState, KVCache, KVCacheMode
from rcm.utils.model_utils import init_weights_on_device, load_state_dict, load_state_dict_from_dcp
from rcm.utils.umt5 import clear_umt5_memory, get_umt5_embedding

torch._dynamo.config.suppress_errors = True

_DEFAULT_NEGATIVE_PROMPT = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
_DEFAULT_PROMPT = "A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about."

TENSOR_KWARGS = {"device": "cuda", "dtype": torch.bfloat16}

WAN2PT1_1PT3B_T2V: LazyDict = L(WanModel)(
    dim=1536,
    eps=1e-06,
    ffn_dim=8960,
    freq_dim=256,
    in_dim=16,
    model_type="t2v",
    num_heads=12,
    num_layers=30,
    out_dim=16,
    text_len=512,
)
WAN2PT1_14B_T2V: LazyDict = L(WanModel)(
    dim=5120,
    eps=1e-06,
    ffn_dim=13824,
    freq_dim=256,
    in_dim=16,
    model_type="t2v",
    num_heads=40,
    num_layers=40,
    out_dim=16,
    text_len=512,
)
DIT_CONFIGS = {"1.3B": WAN2PT1_1PT3B_T2V, "14B": WAN2PT1_14B_T2V}

RECTIFIED_FLOW_T_SCALING = 1000.0


def load_dit_weights(net, dit_path: str):
    if os.path.isdir(dit_path):
        state_dict = load_state_dict_from_dcp(dit_path)
    else:
        state_dict = load_state_dict(dit_path)
    for key in ["generator", "generator_ema"]:
        if key in state_dict.keys():
            state_dict = state_dict[key]
    for prefix in ["model.", "net.", "_fsdp_wrapped_module."]:
        state_dict = {(k[len(prefix) :] if k.startswith(prefix) else k): v for k, v in state_dict.items()}
    for k, v in state_dict.items():
        if k.endswith("patch_embedding.weight"):
            state_dict[k] = v.reshape(net.patch_embedding.weight.shape)
        if k.endswith("patch_embedding.bias"):
            state_dict[k] = v.reshape(net.patch_embedding.bias.shape)
    net.load_state_dict(state_dict, strict=True, assign=True)
    del state_dict


# ---------------------------------------------------------------------------
# Per-method configuration (union type)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SlidingWindowConfig:
    """FIFO sliding window with optional attention sink.

    Keeps the first ``sink_blocks`` blocks permanently plus the most recent
    blocks, for a total of ``window_blocks``.  When ``sink_blocks=0``
    (default), this is a pure FIFO window.
    """

    window_blocks: int = 6
    sink_blocks: int = 0

    def __post_init__(self):
        if self.window_blocks < 1:
            raise ValueError(f"window_blocks must be >= 1, got {self.window_blocks}")
        if self.sink_blocks < 0:
            raise ValueError(f"sink_blocks must be >= 0, got {self.sink_blocks}")
        if self.sink_blocks >= self.window_blocks:
            raise ValueError(f"sink_blocks ({self.sink_blocks}) must be < window_blocks ({self.window_blocks})")


@dataclass(frozen=True)
class RollingSinkConfig:
    """Rolling Sink (Li et al., 2026).

    The first ``total_blocks`` generated blocks form a within-duration "bank".
    After the bank is full, the sink region cycles through the bank content in
    alternating forward / reverse order while temporal RoPE indices slide to
    stay contiguous with the recent blocks.
    """

    total_blocks: int = 6
    sink_blocks: int = 5
    recent_blocks: int = 1

    def __post_init__(self):
        if self.recent_blocks < 1:
            raise ValueError(f"recent_blocks must be >= 1, got {self.recent_blocks}")
        if self.sink_blocks < 1:
            raise ValueError(f"sink_blocks must be >= 1, got {self.sink_blocks}")
        if self.sink_blocks + self.recent_blocks > self.total_blocks:
            raise ValueError(
                f"sink_blocks ({self.sink_blocks}) + recent_blocks ({self.recent_blocks}) " f"exceeds total_blocks ({self.total_blocks})"
            )


@dataclass(frozen=True)
class InfinityRopeConfig:
    """Infinity-RoPE (Yesiltepe et al., 2025).

    Block-Relativistic RoPE keeps the newest block at frame index ``f_limit``
    while older blocks are shifted backward.  Blocks whose shifted index falls
    below 0 undergo *semanticization* (temporal indices clamped to 0).
    """

    cache_blocks: int = 6
    sink_blocks: int = 1
    f_limit: int = 21
    flush_interval_blocks: int = 0
    rope_cut_delta: int = 0
    rope_cut_single_frame: bool = True

    def __post_init__(self):
        if self.sink_blocks >= self.cache_blocks:
            raise ValueError(f"sink_blocks ({self.sink_blocks}) must be < cache_blocks ({self.cache_blocks})")
        if self.f_limit < 2:
            raise ValueError(f"f_limit must be >= 2, got {self.f_limit}")


@dataclass(frozen=True)
class DeepForcingConfig:
    """Deep Forcing (Yi et al., 2025).

    Deep Sink preserves the first ``sink_frames`` frames.  Participative
    Compression scores every cached *token* via Q·K^T and keeps the Top-C
    tokens from the candidate region.  Scoring uses a rolling window of the
    last ``recent_frames`` queries and per-layer max fusion.
    """

    sink_frames: int = 10
    budget_frames: int = 16
    recent_frames: int = 4
    observation_layers: int = 0

    def __post_init__(self):
        if self.sink_frames + self.recent_frames > self.budget_frames:
            raise ValueError(
                f"sink_frames ({self.sink_frames}) + recent_frames ({self.recent_frames}) " f"exceeds budget_frames ({self.budget_frames})"
            )


@dataclass(frozen=True)
class MemRopeConfig:
    """MemRoPE (Kim et al., 2026).

    Dual EMA memory tokens continuously compress evicted frames into long-term
    (identity) and short-term (dynamics) streams.  Keys are stored without
    RoPE and positioned via compact block-relative indices at attention time.
    """

    sink_frames: int = 3
    local_frames: int = 4
    alpha_long: float = 0.01
    alpha_short: float = 0.1

    def __post_init__(self):
        if self.sink_frames < 0:
            raise ValueError(f"sink_frames must be >= 0, got {self.sink_frames}")
        if self.local_frames < 1:
            raise ValueError(f"local_frames must be >= 1, got {self.local_frames}")


@dataclass(frozen=True)
class RelaxForcingConfig:
    """Relax Forcing (Zhao et al., 2026).

    Structured temporal memory: Sink (global anchor) + History (sparse,
    selected by alignment/complementarity scoring) + Tail (recent window).
    Scoring uses cosine similarity rather than Q*K^T attention.
    """

    sink_frames: int = 4
    budget_frames: int = 16
    recent_frames: int = 4
    alignment_weight: float = 0.5

    def __post_init__(self):
        if self.sink_frames + self.recent_frames > self.budget_frames:
            raise ValueError(
                f"sink_frames ({self.sink_frames}) + recent_frames ({self.recent_frames}) " f"exceeds budget_frames ({self.budget_frames})"
            )
        if not 0.0 <= self.alignment_weight <= 1.0:
            raise ValueError(f"alignment_weight must be in [0, 1], got {self.alignment_weight}")


ExtrapolationConfig = Union[SlidingWindowConfig, RollingSinkConfig, InfinityRopeConfig, DeepForcingConfig, MemRopeConfig, RelaxForcingConfig, None]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_shifted_ode_t_steps(num_steps: int, sigma_max: float, timestep_shift: float, device: torch.device) -> torch.Tensor:
    sigma_max_rf = sigma_max / (sigma_max + 1)
    unshifted_sigma_max = sigma_max_rf / (timestep_shift - (timestep_shift - 1) * sigma_max_rf)
    t_steps = torch.linspace(unshifted_sigma_max, 0.0, num_steps + 1, dtype=torch.float64, device=device)
    return timestep_shift * t_steps / (1 + (timestep_shift - 1) * t_steps)


def build_few_step_t_steps(num_steps: int, sigma_max: float, mid_t: list[float], device: torch.device) -> torch.Tensor:
    mid_t = mid_t[: num_steps - 1]
    sigma_max_rf = sigma_max / (sigma_max + 1)
    return torch.tensor([sigma_max_rf, *mid_t, 0], dtype=torch.float64, device=device)


def make_block_pattern(T: int, H: int, W: int, first_chunk_t: int, chunk_t: int, spatial_patch_size: int):
    frame_tokens = H * W // spatial_patch_size
    assert (T - first_chunk_t) % chunk_t == 0, f"T={T}, first_chunk_t={first_chunk_t}, chunk_t={chunk_t}"
    num_blocks = 1 + (T - first_chunk_t) // chunk_t
    pattern = BlockPattern(frame_tokens=frame_tokens, first_chunk_frames=first_chunk_t, chunk_frames=chunk_t)
    return frame_tokens, num_blocks, pattern


def block_span(pattern: BlockPattern, block_idx: int):
    frame_start = pattern.blocks_to_frames(block_idx)
    block_size = pattern.block_size(block_idx)
    return frame_start, frame_start + block_size, block_size


# ---------------------------------------------------------------------------
# Deep Forcing: token-level attention observer
# ---------------------------------------------------------------------------


class DeepForcingTokenCollector:
    """Collects per-layer, per-token Q·K^T importance scores during APPEND.

    Implements Eq. 7 from Deep Forcing: ``phi_j = sum_r q_r^T k_j``.

    Each observed layer uses its own rolling query window (matching the
    official per-layer ``win_q`` buffer) and produces its own score vector
    for independent per-layer token selection.
    """

    def __init__(
        self, cached_len: int, recent_tokens: int = 0, layer_ids: Optional[Set[int]] = None, prior_queries: Optional[dict[int, torch.Tensor]] = None
    ):
        self.cached_len = cached_len
        self.recent_tokens = recent_tokens
        self.layer_ids = layer_ids
        self._prior_queries = prior_queries or {}
        self._layer_queries: dict[int, torch.Tensor] = {}
        self._layer_scores: dict[int, torch.Tensor] = {}

    def observe(self, layer_idx: int, q_block_idx: int, query: torch.Tensor, key: torch.Tensor, cached_len: int):
        del q_block_idx
        if cached_len <= 0:
            return
        if self.layer_ids is not None and layer_idx not in self.layer_ids:
            return

        q_detached = query.detach().float()
        self._layer_queries[layer_idx] = q_detached

        prior = self._prior_queries.get(layer_idx)
        if prior is not None and self.recent_tokens > 0:
            all_q = torch.cat([prior.to(query.device), q_detached], dim=1)
            all_q = all_q[:, -self.recent_tokens :]
        else:
            all_q = q_detached

        cached_key = key[:, :cached_len].float()
        scale = 1.0 / math.sqrt(all_q.shape[-1])
        scores = torch.einsum("bqhd,bkhd->bk", all_q, cached_key) * scale  # [B, S_cached]
        self._layer_scores[layer_idx] = scores[:, :cached_len]

    def finalize_per_layer(self) -> dict[int, torch.Tensor]:
        return self._layer_scores

    def get_layer_queries(self) -> dict[int, torch.Tensor]:
        return self._layer_queries


# ---------------------------------------------------------------------------
# ExtrapolationRuntime
# ---------------------------------------------------------------------------


class ExtrapolationRuntime:
    """Manages KV-cache eviction, RoPE re-indexing, and content cycling.

    Each extrapolation method is implemented as a group of private methods
    called by the public API (``readonly_state``, ``append_state``,
    ``finalize_after_append``).
    """

    def __init__(self, config: ExtrapolationConfig, pattern: BlockPattern, device: torch.device, num_layers: int):
        self.config = config
        self.pattern = pattern
        self.device = device
        self.num_layers = num_layers

        self._block_ids: List[int] = []
        self._block_roles: List[str] = []

        # Rolling Sink state
        self._rs_bank: Optional[List[List[Tuple[torch.Tensor, torch.Tensor]]]] = None
        self._rs_bank_block_count: int = 0
        self._rs_bank_frames: List[int] = []
        self._rs_phase: int = 0

        # Infinity-RoPE state
        self._ir_flush_on_next: bool = False
        self._ir_cut_on_next: bool = False

        # Deep Forcing state
        self._df_layer_query_buffers: dict[int, torch.Tensor] = {}
        self._df_sink_rope_base: int = 0
        self._df_comp_rope_base: int = 0
        self._active_collector: Optional[DeepForcingTokenCollector] = None

        # MemRoPE state (per-layer dual EMA memory tokens)
        self._mr_long_k: Optional[List[Optional[torch.Tensor]]] = None
        self._mr_long_v: Optional[List[Optional[torch.Tensor]]] = None
        self._mr_short_k: Optional[List[Optional[torch.Tensor]]] = None
        self._mr_short_v: Optional[List[Optional[torch.Tensor]]] = None

        # Relax Forcing state
        self._rx_sink_rope_base: int = 0
        self._rx_comp_rope_base: int = 0

    # -- public helpers ------------------------------------------------------

    @property
    def method(self) -> str:
        if self.config is None:
            return "none"
        if isinstance(self.config, SlidingWindowConfig):
            return "sliding_window"
        if isinstance(self.config, RollingSinkConfig):
            return "rolling_sink"
        if isinstance(self.config, InfinityRopeConfig):
            return "infinity_rope"
        if isinstance(self.config, DeepForcingConfig):
            return "deep_forcing"
        if isinstance(self.config, MemRopeConfig):
            return "memrope"
        if isinstance(self.config, RelaxForcingConfig):
            return "relax_forcing"
        raise ValueError(f"Unknown config type: {type(self.config)}")

    @property
    def blocks(self) -> List[Tuple[int, str]]:
        return list(zip(self._block_ids, self._block_roles))

    def reset(self):
        self._block_ids.clear()
        self._block_roles.clear()
        self._rs_bank = None
        self._rs_bank_block_count = 0
        self._rs_bank_frames = []
        self._rs_phase = 0
        self._ir_flush_on_next = False
        self._ir_cut_on_next = False
        self._df_layer_query_buffers = {}
        self._df_sink_rope_base = 0
        self._df_comp_rope_base = 0
        self._active_collector = None
        self._mr_long_k = None
        self._mr_long_v = None
        self._mr_short_k = None
        self._mr_short_v = None
        self._rx_sink_rope_base = 0
        self._rx_comp_rope_base = 0

    def cache_max_len(self, total_tokens: int) -> int:
        if self.config is None:
            return total_tokens
        ft = self.pattern.frame_tokens
        max_blk = max(self.pattern.get_block_tokens(0), self.pattern.chunk_frames * ft)
        if isinstance(self.config, SlidingWindowConfig):
            cap = self.config.window_blocks * max_blk
        elif isinstance(self.config, RollingSinkConfig):
            cap = self.config.total_blocks * max_blk
        elif isinstance(self.config, InfinityRopeConfig):
            cap = self.config.cache_blocks * max_blk
        elif isinstance(self.config, DeepForcingConfig):
            cap = self.config.budget_frames * ft
        elif isinstance(self.config, MemRopeConfig):
            mc = self.config
            cap = (mc.sink_frames + mc.local_frames + 2) * ft
        elif isinstance(self.config, RelaxForcingConfig):
            cap = self.config.budget_frames * ft
        else:
            return total_tokens
        return min(total_tokens, cap + max_blk)

    # -- state builders ------------------------------------------------------

    def readonly_state(self, kv_caches: List[KVCache], block_idx: int) -> CausalInferenceState:
        return self._build_state(kv_caches, block_idx, KVCacheMode.READONLY)

    def append_state(self, kv_caches: List[KVCache], block_idx: int) -> CausalInferenceState:
        attn_observer = None
        if isinstance(self.config, DeepForcingConfig) and self._block_ids:
            cached_len = kv_caches[0].current_len if kv_caches else 0
            layers = self._df_observation_layers()
            recent_tokens = self.config.recent_frames * self.pattern.frame_tokens
            attn_observer = DeepForcingTokenCollector(
                cached_len,
                recent_tokens=recent_tokens,
                layer_ids=layers,
                prior_queries=self._df_layer_query_buffers,
            )
            self._active_collector = attn_observer
        else:
            self._active_collector = None
        return self._build_state(kv_caches, block_idx, KVCacheMode.APPEND, attn_observer=attn_observer)

    def finalize_after_append(self, kv_caches: List[KVCache], block_idx: int):
        if isinstance(self.config, RollingSinkConfig):
            self._rs_finalize(kv_caches, block_idx)
        elif isinstance(self.config, InfinityRopeConfig):
            self._ir_finalize(kv_caches, block_idx)
        elif isinstance(self.config, DeepForcingConfig):
            self._df_finalize(kv_caches, block_idx)
        elif isinstance(self.config, SlidingWindowConfig):
            self._sw_finalize(kv_caches, block_idx)
        elif isinstance(self.config, MemRopeConfig):
            self._mr_finalize(kv_caches, block_idx)
        elif isinstance(self.config, RelaxForcingConfig):
            self._rx_finalize(kv_caches, block_idx)
        else:
            self._block_ids.append(block_idx)
            self._block_roles.append("recent")

    def request_flush(self):
        """Trigger an on-demand KV Flush at the next finalize step (Infinity-RoPE)."""
        self._ir_flush_on_next = True

    def request_scene_cut(self):
        """Trigger a KV Flush + RoPE Cut at the next finalize step (Infinity-RoPE).

        Combines KV Flush (keep sink + last frame) with a discontinuous temporal
        jump of ``rope_cut_delta`` frames, producing a cinematic scene transition.
        """
        self._ir_flush_on_next = True
        self._ir_cut_on_next = True

    # -- internal: build inference state ------------------------------------

    def _build_state(
        self,
        kv_caches: List[KVCache],
        block_idx: int,
        mode: KVCacheMode,
        attn_observer: Optional[DeepForcingTokenCollector] = None,
    ) -> CausalInferenceState:
        if isinstance(self.config, (DeepForcingConfig, RelaxForcingConfig)):
            return CausalInferenceState(
                mode=mode,
                kv_caches=kv_caches,
                pattern=self.pattern,
                block_cursor=block_idx,
                block_range=None,
                attn_observer=attn_observer,
            )
        cached_t = self._cached_t_indices(block_idx)
        current_t = self._current_t_indices(block_idx)
        return CausalInferenceState(
            mode=mode,
            kv_caches=kv_caches,
            pattern=self.pattern,
            block_cursor=block_idx,
            block_range=None,
            cached_t_indices=cached_t,
            current_t_indices=current_t,
            attn_observer=attn_observer,
        )

    # -- chunk helpers -------------------------------------------------------

    def _chunk_frames(self, block_idx: int) -> int:
        return self.pattern.block_size(block_idx)

    def _block_frame_start(self, block_idx: int) -> int:
        return self.pattern.blocks_to_frames(block_idx)

    def _block_token_len(self, block_idx: int) -> int:
        return self._chunk_frames(block_idx) * self.pattern.frame_tokens

    def _absolute_frame_ids(self, block_idx: int) -> torch.Tensor:
        fs = self._block_frame_start(block_idx)
        bf = self._chunk_frames(block_idx)
        return torch.arange(fs, fs + bf, device=self.device, dtype=torch.long)

    def _expand(self, frame_ids: torch.Tensor) -> torch.Tensor:
        return torch.repeat_interleave(frame_ids, repeats=self.pattern.frame_tokens)

    # -- temporal index dispatch ---------------------------------------------

    def _cached_t_indices(self, current_block_idx: int) -> Optional[torch.Tensor]:
        if not self._block_ids:
            return None
        parts = [self._block_t_indices(bid, current_block_idx) for bid in self._block_ids]
        return torch.cat(parts, dim=0)

    def _current_t_indices(self, current_block_idx: int) -> torch.Tensor:
        return self._block_t_indices(current_block_idx, current_block_idx)

    def _block_t_indices(self, block_idx: int, current_block_idx: int) -> torch.Tensor:
        m = self.method
        if m in ("none", "sliding_window", "deep_forcing", "relax_forcing"):
            return self._expand(self._absolute_frame_ids(block_idx))
        if m == "rolling_sink":
            return self._rs_block_t_indices(block_idx, current_block_idx)
        if m == "infinity_rope":
            return self._ir_block_t_indices(block_idx, current_block_idx)
        if m == "memrope":
            return self._mr_block_t_indices(block_idx, current_block_idx)
        raise ValueError(m)

    # ======================================================================
    # Sliding Window
    # ======================================================================

    def _sw_finalize(self, kv_caches: List[KVCache], block_idx: int):
        assert isinstance(self.config, SlidingWindowConfig)
        S = self.config.sink_blocks
        self._block_ids.append(block_idx)
        self._block_roles.append("sink" if len(self._block_ids) <= S else "recent")
        cap = self.config.window_blocks
        if len(self._block_ids) > cap:
            n_evict = len(self._block_ids) - cap
            evict_start = S
            if evict_start >= len(self._block_ids):
                return
            n_evict = min(n_evict, len(self._block_ids) - S - 1)
            if n_evict <= 0:
                return
            sink_tokens = sum(self._block_token_len(self._block_ids[i]) for i in range(evict_start))
            evict_tokens = sum(self._block_token_len(self._block_ids[evict_start + i]) for i in range(n_evict))
            keep = torch.cat(
                [
                    torch.arange(0, sink_tokens, device=self.device, dtype=torch.long),
                    torch.arange(sink_tokens + evict_tokens, kv_caches[0].current_len, device=self.device, dtype=torch.long),
                ]
            )
            for c in kv_caches:
                c.compact_(keep)
            self._block_ids = self._block_ids[:evict_start] + self._block_ids[evict_start + n_evict :]
            self._block_roles = self._block_roles[:evict_start] + self._block_roles[evict_start + n_evict :]

    # ======================================================================
    # Rolling Sink  (Li et al., 2026)
    # ======================================================================

    def _rs_finalize(self, kv_caches: List[KVCache], block_idx: int):
        """After appending block ``block_idx``, apply Rolling Sink policy.

        Phase 1 (bank fill): accumulate the first ``total_blocks`` blocks
        normally and snapshot them as the within-duration bank.

        Phase 2 (rolling): rebuild the cache from bank content (sink region in
        rolling forward/reverse order) + the most recent block(s).
        """
        cfg = self.config
        assert isinstance(cfg, RollingSinkConfig)
        K = cfg.total_blocks

        self._block_ids.append(block_idx)
        self._block_roles.append("recent")

        if len(self._block_ids) <= K:
            if len(self._block_ids) == K:
                self._rs_snapshot_bank(kv_caches)
            return

        self._rs_rebuild_cache(kv_caches, block_idx)

    def _rs_snapshot_bank(self, kv_caches: List[KVCache]):
        """Save the first K blocks' pre-RoPE KV data as the within-duration bank."""
        cfg = self.config
        assert isinstance(cfg, RollingSinkConfig)
        K = cfg.total_blocks
        self._rs_bank = []
        for cache in kv_caches:
            layer_blocks: List[Tuple[torch.Tensor, torch.Tensor]] = []
            cur = 0
            for i, bid in enumerate(self._block_ids[:K]):
                tlen = self._block_token_len(bid)
                k_block = cache.k[:, cur : cur + tlen].clone()
                v_block = cache.v[:, cur : cur + tlen].clone()
                layer_blocks.append((k_block, v_block))
                cur += tlen
            self._rs_bank.append(layer_blocks)
        self._rs_bank_block_count = K
        self._rs_bank_frames = [self._chunk_frames(bid) for bid in self._block_ids[:K]]

    def _rs_rebuild_cache(self, kv_caches: List[KVCache], latest_block_idx: int):
        """Rebuild cache with rolling sink content + most recent block."""
        cfg = self.config
        assert isinstance(cfg, RollingSinkConfig)
        assert self._rs_bank is not None
        K = cfg.total_blocks
        S = cfg.sink_blocks
        R = cfg.recent_blocks

        self._rs_phase += 1
        sink_bank_indices = self._rs_rolling_indices(S, K)

        recent_kv_per_layer: List[Tuple[torch.Tensor, torch.Tensor]] = []
        recent_block_ids = self._block_ids[-R:]
        for cache in kv_caches:
            recent_start = cache.current_len - sum(self._block_token_len(bid) for bid in recent_block_ids)
            k_recent = cache.k[:, recent_start : cache.current_len].clone()
            v_recent = cache.v[:, recent_start : cache.current_len].clone()
            recent_kv_per_layer.append((k_recent, v_recent))

        for layer_idx, cache in enumerate(kv_caches):
            cache.reset(free_buffers=False)
            bank = self._rs_bank[layer_idx]
            for bank_blk_idx in sink_bank_indices:
                is_reversed = (self._rs_phase % (2 * K)) >= K
                k_blk, v_blk = bank[bank_blk_idx]
                if is_reversed:
                    ft = self.pattern.frame_tokens
                    n_frames = k_blk.shape[1] // ft
                    if n_frames > 1:
                        k_blk = self._reverse_frames(k_blk, ft, n_frames)
                        v_blk = self._reverse_frames(v_blk, ft, n_frames)
                cache.append(k_blk, v_blk)
            k_r, v_r = recent_kv_per_layer[layer_idx]
            cache.append(k_r, v_r)

        self._block_ids = [-(i + 1) for i in range(S)] + recent_block_ids
        self._block_roles = ["sink"] * S + ["recent"] * R

    def _rs_rolling_indices(self, S: int, K: int) -> List[int]:
        """Return S bank block indices for the current rolling phase.

        Implements Eq. 11: forward on even cycles, reverse on odd cycles.
        """
        cycle = self._rs_phase % (2 * K)
        if cycle < K:
            start = self._rs_phase % K
            return [(start + i) % K for i in range(S)]
        start = (K - 1) - (self._rs_phase % K)
        return [(start - i) % K for i in range(S)]

    @staticmethod
    def _reverse_frames(t: torch.Tensor, frame_tokens: int, n_frames: int) -> torch.Tensor:
        """Reverse frame order within a block tensor [B, S, H, D]."""
        chunks = [t[:, i * frame_tokens : (i + 1) * frame_tokens] for i in range(n_frames)]
        return torch.cat(list(reversed(chunks)), dim=1)

    def _rs_sink_frames(self, pos_in_sinks: int) -> int:
        """Frame count for the sink block at position ``pos_in_sinks``."""
        sink_bank_indices = (
            self._rs_rolling_indices(
                self.config.sink_blocks,
                self.config.total_blocks,
            )
            if isinstance(self.config, RollingSinkConfig)
            else []
        )
        if pos_in_sinks < len(sink_bank_indices):
            bank_idx = sink_bank_indices[pos_in_sinks]
            if bank_idx < len(self._rs_bank_frames):
                return self._rs_bank_frames[bank_idx]
        return self.pattern.chunk_frames

    def _rs_block_t_indices(self, block_idx: int, current_block_idx: int) -> torch.Tensor:
        """Sliding indices: sink blocks positioned contiguously before recent."""
        if block_idx not in self._block_ids:
            return self._expand(self._absolute_frame_ids(block_idx))
        pos = self._block_ids.index(block_idx)
        role = self._block_roles[pos]

        if role == "recent" or block_idx == current_block_idx:
            if block_idx >= 0:
                return self._expand(self._absolute_frame_ids(block_idx))
            bf = self.pattern.chunk_frames
            fs = self._block_frame_start(current_block_idx) - bf
            return self._expand(torch.arange(fs, fs + bf, device=self.device, dtype=torch.long))

        sink_positions = [i for i, r in enumerate(self._block_roles) if r == "sink"]
        recent_ids = [bid for bid, r in zip(self._block_ids, self._block_roles) if r == "recent"]

        recent_total = sum(self._chunk_frames(bid) for bid in recent_ids if bid >= 0)
        current_frames = self._chunk_frames(current_block_idx)
        recent_end = self._block_frame_start(current_block_idx) + current_frames

        sink_frames_list = [self._rs_sink_frames(i) for i in range(len(sink_positions))]
        sink_total = sum(sink_frames_list)

        recent_start = recent_end - current_frames - recent_total
        sink_start = recent_start - sink_total

        my_sink_idx = sink_positions.index(pos)
        offset = sum(sink_frames_list[:my_sink_idx])
        bf = sink_frames_list[my_sink_idx]
        ids = torch.arange(sink_start + offset, sink_start + offset + bf, device=self.device, dtype=torch.long)
        return self._expand(ids)

    # ======================================================================
    # Infinity-RoPE  (Yesiltepe et al., 2025)
    # ======================================================================

    def _ir_finalize(self, kv_caches: List[KVCache], block_idx: int):
        cfg = self.config
        assert isinstance(cfg, InfinityRopeConfig)

        self._block_ids.append(block_idx)
        self._block_roles.append("recent")

        needs_flush = self._ir_flush_on_next
        if not needs_flush and cfg.flush_interval_blocks > 0:
            needs_flush = (block_idx + 1) % cfg.flush_interval_blocks == 0

        if needs_flush:
            self._ir_do_flush(kv_caches)
            self._ir_flush_on_next = False
            self._ir_cut_on_next = False
            return

        if len(self._block_ids) > cfg.cache_blocks:
            self._ir_evict_oldest(kv_caches)

    def _ir_do_flush(self, kv_caches: List[KVCache]):
        """KV Flush: keep only the global sink block(s) and the last generated block."""
        cfg = self.config
        assert isinstance(cfg, InfinityRopeConfig)
        S = cfg.sink_blocks

        if len(self._block_ids) <= S + 1:
            return

        sink_tokens = sum(self._block_token_len(self._block_ids[i]) for i in range(min(S, len(self._block_ids))))
        last_tokens = self._block_token_len(self._block_ids[-1])
        total = kv_caches[0].current_len

        keep_parts = [torch.arange(0, sink_tokens, device=self.device, dtype=torch.long)]
        if total - last_tokens >= sink_tokens:
            keep_parts.append(torch.arange(total - last_tokens, total, device=self.device, dtype=torch.long))
        keep = torch.cat(keep_parts)
        for c in kv_caches:
            c.compact_(keep)

        kept_ids = self._block_ids[:S] + self._block_ids[-1:]
        kept_roles = ["sink"] * S + ["recent"]
        self._block_ids = kept_ids
        self._block_roles = kept_roles

    def _ir_evict_oldest(self, kv_caches: List[KVCache]):
        """FIFO eviction of the oldest non-sink blocks (batch)."""
        cfg = self.config
        assert isinstance(cfg, InfinityRopeConfig)
        S = cfg.sink_blocks
        n_evict = len(self._block_ids) - cfg.cache_blocks
        if n_evict <= 0 or S >= len(self._block_ids):
            return

        sink_tokens = sum(self._block_token_len(self._block_ids[i]) for i in range(S))
        evict_tokens = sum(self._block_token_len(self._block_ids[S + i]) for i in range(n_evict))
        keep = torch.cat(
            [
                torch.arange(0, sink_tokens, device=self.device, dtype=torch.long),
                torch.arange(sink_tokens + evict_tokens, kv_caches[0].current_len, device=self.device, dtype=torch.long),
            ]
        )
        for c in kv_caches:
            c.compact_(keep)
        self._block_ids = self._block_ids[:S] + self._block_ids[S + n_evict :]
        self._block_roles = self._block_roles[:S] + self._block_roles[S + n_evict :]

        for i in range(min(S, len(self._block_ids))):
            self._block_roles[i] = "sink"

    def _ir_block_t_indices(self, block_idx: int, current_block_idx: int) -> torch.Tensor:
        """Block-Relativistic RoPE with semanticization and RoPE Cut.

        The current block is placed at the end of the teacher's horizon
        (frame index ``f_limit - chunk_frames`` to ``f_limit - 1``).  Older
        blocks are shifted backward; blocks that land below 0 are clamped to 0
        (semanticization -- Eq. 3 in the paper).

        When ``rope_cut_delta > 0`` and a scene cut was requested, the entire
        block is shifted forward by ``rope_cut_delta`` frames, creating a
        temporal discontinuity that signals a scene transition (paper Eq. 4).
        This works for any ``chunk_t`` including ``chunk_t=1``.
        """
        cfg = self.config
        assert isinstance(cfg, InfinityRopeConfig)
        f_limit = cfg.f_limit

        cur_frames = self._chunk_frames(current_block_idx)
        cur_end = f_limit
        cur_start = cur_end - cur_frames

        if block_idx == current_block_idx:
            apply_cut = self._ir_cut_on_next and cfg.rope_cut_delta > 0
            if apply_cut and cur_frames == 1 and not cfg.rope_cut_single_frame:
                apply_cut = False
            if apply_cut:
                cut_start = cur_start + cfg.rope_cut_delta
                ids = torch.arange(cut_start, cut_start + cur_frames, device=self.device, dtype=torch.long)
                return self._expand(ids)
            return self._expand(torch.arange(cur_start, cur_end, device=self.device, dtype=torch.long))

        frames_between = 0
        found = False
        for bid in reversed(self._block_ids):
            if bid == block_idx:
                found = True
                break
            frames_between += self._chunk_frames(bid)
        if not found:
            return self._expand(self._absolute_frame_ids(block_idx))

        bf = self._chunk_frames(block_idx)
        block_end = cur_start - frames_between
        block_start = block_end - bf

        ids = torch.arange(block_start, block_end, device=self.device, dtype=torch.long)
        ids.clamp_(min=0)
        return self._expand(ids)

    # ======================================================================
    # Deep Forcing  (Yi et al., 2025)
    # ======================================================================

    def _df_finalize(self, kv_caches: List[KVCache], block_idx: int):
        """Per-layer token-level compression with post-RoPE caching.

        Since cached keys are already rotated, per-layer ``compact_`` just
        removes entries — no RoPE re-computation needed.  After compaction,
        the Deep Sink temporal shift (paper Eq. 5-6) is applied in-place via
        ``rope_time_delta_`` to close the gap between sink and tail.
        """
        from rcm.utils.rope import rope_time_delta_

        cfg = self.config
        assert isinstance(cfg, DeepForcingConfig)
        ft = self.pattern.frame_tokens

        self._block_ids.append(block_idx)
        self._block_roles.append("recent")

        layer_scores: Optional[dict[int, torch.Tensor]] = None
        if self._active_collector is not None:
            layer_scores = self._active_collector.finalize_per_layer()
            recent_tokens = cfg.recent_frames * ft
            for lid, lq in self._active_collector.get_layer_queries().items():
                if lid in self._df_layer_query_buffers:
                    combined = torch.cat([self._df_layer_query_buffers[lid].to(lq.device), lq], dim=1)
                    self._df_layer_query_buffers[lid] = combined[:, -recent_tokens:]
                else:
                    self._df_layer_query_buffers[lid] = lq[:, -recent_tokens:]
            self._active_collector = None

        total_tokens = kv_caches[0].current_len if kv_caches else 0
        budget_tokens = cfg.budget_frames * ft
        if total_tokens <= budget_tokens:
            return

        sink_tokens = cfg.sink_frames * ft
        recent_tokens = cfg.recent_frames * ft
        candidate_tokens = total_tokens - sink_tokens - recent_tokens

        if candidate_tokens <= 0:
            return

        top_c = budget_tokens - sink_tokens - recent_tokens
        if top_c <= 0:
            keep = torch.cat(
                [
                    torch.arange(0, sink_tokens, device=self.device, dtype=torch.long),
                    torch.arange(total_tokens - recent_tokens, total_tokens, device=self.device, dtype=torch.long),
                ]
            )
            cache_end_frame = self._block_frame_start(block_idx) + self._chunk_frames(block_idx)
            tail_frame = cache_end_frame - cfg.recent_frames
            desired_sink_shift = tail_frame - cfg.sink_frames
            sink_incr = desired_sink_shift - self._df_sink_rope_base
            for cache in kv_caches:
                cache.compact_(keep)
                if sink_incr != 0 and sink_tokens > 0 and cache.k is not None:
                    head_dim = cache.k.shape[-1]
                    dim_t = head_dim - (head_dim // 6 * 2) * 2
                    rope_time_delta_(cache.k[:, :sink_tokens], sink_incr, dim_t)
            self._df_sink_rope_base = desired_sink_shift
            self._df_comp_rope_base = 0
            self._block_ids = []
            self._block_roles = []
            if sink_tokens > 0:
                self._block_ids.append(-1)
                self._block_roles.append("sink")
            if recent_tokens > 0:
                self._block_ids.append(-3)
                self._block_roles.append("recent")
            return

        cand_start = sink_tokens
        cand_end = total_tokens - recent_tokens

        sink_range = torch.arange(0, sink_tokens, device=self.device, dtype=torch.long)
        recent_range = torch.arange(cand_end, total_tokens, device=self.device, dtype=torch.long)

        n_candidate = cand_end - cand_start
        n_evicted = n_candidate - top_c

        cache_end_frame = self._block_frame_start(block_idx) + self._chunk_frames(block_idx)
        tail_frame = cache_end_frame - cfg.recent_frames
        top_c_frames = top_c // ft
        desired_comp_start = tail_frame - top_c_frames
        desired_sink_shift = desired_comp_start - cfg.sink_frames

        sink_incr = desired_sink_shift - self._df_sink_rope_base

        current_comp_base = self._df_comp_rope_base if self._df_comp_rope_base != 0 else cand_start // ft
        comp_incr = desired_comp_start - current_comp_base

        B = kv_caches[0].k.shape[0] if kv_caches and kv_caches[0].k is not None else 1

        for layer_idx, cache in enumerate(kv_caches):
            if layer_scores and layer_idx in layer_scores:
                scores = layer_scores[layer_idx]  # [B, S_cached]
                if scores.ndim == 1:
                    scores = scores.unsqueeze(0).expand(B, -1)
                if scores.shape[1] < total_tokens:
                    padded = torch.zeros(B, total_tokens, device=self.device)
                    padded[:, : scores.shape[1]] = scores
                    scores = padded
                cand_scores = scores[:, cand_start:cand_end]  # [B, n_candidate]
                k_select = min(top_c, cand_scores.shape[1])
                _, top_local = torch.topk(cand_scores, k_select, dim=1, sorted=False)  # [B, k_select]
            else:
                k_select = min(top_c, cand_end - cand_start)
                top_local = torch.arange(cand_end - cand_start - k_select, cand_end - cand_start, device=self.device, dtype=torch.long)
                top_local = top_local.unsqueeze(0).expand(B, -1)
            top_global = torch.sort(top_local + cand_start, dim=1).values  # [B, k_select]
            keep = torch.cat(
                [
                    sink_range.unsqueeze(0).expand(B, -1),
                    top_global,
                    recent_range.unsqueeze(0).expand(B, -1),
                ],
                dim=1,
            )  # [B, budget]
            cache.compact_(keep)

            if cache.k is not None and n_evicted > 0:
                head_dim = cache.k.shape[-1]
                dim_t = head_dim - (head_dim // 6 * 2) * 2

                if sink_tokens > 0 and sink_incr != 0:
                    rope_time_delta_(cache.k[:, :sink_tokens], sink_incr, dim_t)

                comp_start_pos = sink_tokens
                comp_end_pos = sink_tokens + top_c
                if comp_end_pos > comp_start_pos and comp_incr != 0:
                    rope_time_delta_(cache.k[:, comp_start_pos:comp_end_pos], comp_incr, dim_t)

        self._df_sink_rope_base = desired_sink_shift
        self._df_comp_rope_base = desired_comp_start

        new_total = budget_tokens
        self._block_ids = []
        self._block_roles = []
        sink_end = cfg.sink_frames * ft
        recent_start = new_total - cfg.recent_frames * ft
        if sink_end > 0:
            self._block_ids.append(-1)
            self._block_roles.append("sink")
        if recent_start > sink_end:
            self._block_ids.append(-2)
            self._block_roles.append("compressed")
        if recent_start < new_total:
            self._block_ids.append(-3)
            self._block_roles.append("recent")

    def _df_observation_layers(self) -> Optional[Set[int]]:
        cfg = self.config
        assert isinstance(cfg, DeepForcingConfig)
        n = cfg.observation_layers
        if n <= 0 or n >= self.num_layers:
            return None
        return set(range(self.num_layers - n, self.num_layers))

    # ======================================================================
    # MemRoPE
    # ======================================================================

    def _mr_finalize(self, kv_caches: List[KVCache], block_idx: int):
        """MemRoPE finalize: maintain sink + dual EMA memory + local window.

        Keys are stored *without* RoPE and repositioned at attention time via
        compact block-relative indices (handled by ``_mr_block_t_indices``).

        Cache layout after compaction: [sink | mem_long | mem_short | local]
        where mem_long and mem_short are single-frame-width EMA summaries.
        """
        cfg = self.config
        assert isinstance(cfg, MemRopeConfig)
        ft = self.pattern.frame_tokens

        self._block_ids.append(block_idx)
        self._block_roles.append("recent")

        total_tokens = kv_caches[0].current_len if kv_caches else 0
        sink_tokens = cfg.sink_frames * ft
        local_tokens = cfg.local_frames * ft
        mem_slots = 2 * ft

        budget = sink_tokens + mem_slots + local_tokens
        if total_tokens <= budget:
            return

        if self._mr_long_k is None:
            self._mr_long_k = [None] * self.num_layers
            self._mr_long_v = [None] * self.num_layers
            self._mr_short_k = [None] * self.num_layers
            self._mr_short_v = [None] * self.num_layers

        evict_start = sink_tokens
        evict_end = total_tokens - local_tokens
        if evict_end <= evict_start:
            return

        alpha_l = cfg.alpha_long
        alpha_s = cfg.alpha_short

        for layer_idx, cache in enumerate(kv_caches):
            if cache.k is None:
                continue
            evict_k = cache.k[:, evict_start:evict_end].float()
            evict_v = cache.v[:, evict_start:evict_end].float()

            evict_mean_k = evict_k.mean(dim=1, keepdim=True)
            evict_mean_v = evict_v.mean(dim=1, keepdim=True)

            if self._mr_long_k[layer_idx] is None:
                self._mr_long_k[layer_idx] = evict_mean_k.to(cache.k.dtype)
                self._mr_long_v[layer_idx] = evict_mean_v.to(cache.v.dtype)
            else:
                prev_k = self._mr_long_k[layer_idx].float()
                prev_v = self._mr_long_v[layer_idx].float()
                self._mr_long_k[layer_idx] = ((1 - alpha_l) * prev_k + alpha_l * evict_mean_k).to(cache.k.dtype)
                self._mr_long_v[layer_idx] = ((1 - alpha_l) * prev_v + alpha_l * evict_mean_v).to(cache.v.dtype)

            if self._mr_short_k[layer_idx] is None:
                self._mr_short_k[layer_idx] = evict_mean_k.to(cache.k.dtype)
                self._mr_short_v[layer_idx] = evict_mean_v.to(cache.v.dtype)
            else:
                prev_k = self._mr_short_k[layer_idx].float()
                prev_v = self._mr_short_v[layer_idx].float()
                self._mr_short_k[layer_idx] = ((1 - alpha_s) * prev_k + alpha_s * evict_mean_k).to(cache.k.dtype)
                self._mr_short_v[layer_idx] = ((1 - alpha_s) * prev_v + alpha_s * evict_mean_v).to(cache.v.dtype)

            B, _, H, D = cache.k.shape
            long_k = self._mr_long_k[layer_idx].expand(B, ft, H, D)
            long_v = self._mr_long_v[layer_idx].expand(B, ft, H, D)
            short_k = self._mr_short_k[layer_idx].expand(B, ft, H, D)
            short_v = self._mr_short_v[layer_idx].expand(B, ft, H, D)

            sink_k = cache.k[:, :sink_tokens]
            sink_v = cache.v[:, :sink_tokens]
            local_k = cache.k[:, total_tokens - local_tokens : total_tokens]
            local_v = cache.v[:, total_tokens - local_tokens : total_tokens]

            rebuilt_k = torch.cat([sink_k, long_k, short_k, local_k], dim=1)
            rebuilt_v = torch.cat([sink_v, long_v, short_v, local_v], dim=1)

            cache.k[:, :budget].copy_(rebuilt_k.detach())
            cache.v[:, :budget].copy_(rebuilt_v.detach())
            cache.cur = budget
            cache._chunk_lens = [budget]
            cache._cum_ends = [budget]

        self._block_ids = []
        self._block_roles = []
        if sink_tokens > 0:
            self._block_ids.append(-1)
            self._block_roles.append("sink")
        self._block_ids.append(-4)
        self._block_roles.append("mem_long")
        self._block_ids.append(-5)
        self._block_roles.append("mem_short")
        if local_tokens > 0:
            self._block_ids.append(-3)
            self._block_roles.append("recent")

    def _mr_block_t_indices(self, block_idx: int, current_block_idx: int) -> torch.Tensor:
        """Online RoPE Indexing: all positions in a bounded window [0, W).

        After compaction the layout is:
            [sink 0..S-1 | mem_long S | mem_short S+1 | local S+2..S+L+1]
        and the current block occupies [S+L+2 .. S+L+1+cur_frames].

        Before compaction (block_idx >= 0), blocks are assigned contiguous
        bounded positions starting from 0 so they never exceed the training
        horizon.  This is the paper's Online RoPE Indexing.
        """
        cfg = self.config
        assert isinstance(cfg, MemRopeConfig)
        ft = self.pattern.frame_tokens

        S = cfg.sink_frames
        L = cfg.local_frames
        has_mem = any(bid in (-4, -5) for bid in self._block_ids)

        if block_idx >= 0:
            if has_mem:
                cur_start = S + 2 + L
                bf = self._chunk_frames(block_idx)
                return self._expand(torch.arange(cur_start, cur_start + bf, device=self.device, dtype=torch.long))
            cached_frames = sum(self._chunk_frames(bid) for bid in self._block_ids if bid >= 0)
            bf = self._chunk_frames(block_idx)
            if block_idx == current_block_idx and block_idx not in self._block_ids:
                frame_offset = cached_frames
            elif block_idx in self._block_ids:
                frame_offset = 0
                for bid in self._block_ids:
                    if bid == block_idx:
                        break
                    if bid >= 0:
                        frame_offset += self._chunk_frames(bid)
            else:
                frame_offset = cached_frames
            return self._expand(torch.arange(frame_offset, frame_offset + bf, device=self.device, dtype=torch.long))

        if block_idx == -1:
            return self._expand(torch.arange(0, S, device=self.device, dtype=torch.long))

        if block_idx == -4:
            return torch.full((ft,), S, device=self.device, dtype=torch.long)

        if block_idx == -5:
            return torch.full((ft,), S + 1, device=self.device, dtype=torch.long)

        if block_idx == -3:
            start = S + 2
            return self._expand(torch.arange(start, start + L, device=self.device, dtype=torch.long))

        return self._expand(self._absolute_frame_ids(block_idx))

    # ======================================================================
    # Relax Forcing
    # ======================================================================

    def _rx_finalize(self, kv_caches: List[KVCache], block_idx: int):
        """Relax Forcing finalize: Sink + History (similarity-selected) + Tail.

        Uses cosine similarity between the mean of the tail region and each
        candidate frame to select the most relevant history frames.  A
        complementarity bonus (1 - inter-similarity) encourages diversity.

        Post-RoPE caching: keys are stored rotated.  After compaction each
        selected history frame receives a per-frame ``rope_time_delta_`` shift
        to map it from its original absolute position to a contiguous position
        immediately before the tail.  The sink region is shifted uniformly.
        """
        from rcm.utils.rope import rope_time_delta_

        cfg = self.config
        assert isinstance(cfg, RelaxForcingConfig)
        ft = self.pattern.frame_tokens

        self._block_ids.append(block_idx)
        self._block_roles.append("recent")

        total_tokens = kv_caches[0].current_len if kv_caches else 0
        budget_tokens = cfg.budget_frames * ft
        if total_tokens <= budget_tokens:
            return

        sink_tokens = cfg.sink_frames * ft
        recent_tokens = cfg.recent_frames * ft
        candidate_tokens = total_tokens - sink_tokens - recent_tokens

        if candidate_tokens <= 0:
            return

        history_budget = budget_tokens - sink_tokens - recent_tokens
        if history_budget <= 0:
            keep = torch.cat(
                [
                    torch.arange(0, sink_tokens, device=self.device, dtype=torch.long),
                    torch.arange(total_tokens - recent_tokens, total_tokens, device=self.device, dtype=torch.long),
                ]
            )
            cache_end_frame = self._block_frame_start(block_idx) + self._chunk_frames(block_idx)
            tail_frame = cache_end_frame - cfg.recent_frames
            desired_sink_shift = tail_frame - cfg.sink_frames
            sink_incr = desired_sink_shift - self._rx_sink_rope_base
            for cache in kv_caches:
                cache.compact_(keep)
                if sink_incr != 0 and sink_tokens > 0 and cache.k is not None:
                    head_dim = cache.k.shape[-1]
                    dim_t = head_dim - (head_dim // 6 * 2) * 2
                    rope_time_delta_(cache.k[:, :sink_tokens], sink_incr, dim_t)
            self._rx_sink_rope_base = desired_sink_shift
            self._rx_comp_rope_base = 0
            self._block_ids = []
            self._block_roles = []
            if sink_tokens > 0:
                self._block_ids.append(-1)
                self._block_roles.append("sink")
            if recent_tokens > 0:
                self._block_ids.append(-3)
                self._block_roles.append("recent")
            return

        cand_start = sink_tokens
        cand_end = total_tokens - recent_tokens
        n_cand_frames = (cand_end - cand_start) // ft
        history_frames = history_budget // ft

        ref_cache = kv_caches[0]
        if ref_cache.k is None:
            return

        tail_k = ref_cache.k[:, cand_end:total_tokens].float()
        tail_mean = tail_k.mean(dim=1, keepdim=True)
        tail_norm = torch.nn.functional.normalize(tail_mean.flatten(2), dim=-1)

        cand_k = ref_cache.k[:, cand_start:cand_end].float()
        B = cand_k.shape[0]
        cand_frames = cand_k.reshape(B, n_cand_frames, ft, *cand_k.shape[2:])
        cand_frame_mean = cand_frames.mean(dim=2)
        cand_flat = torch.nn.functional.normalize(cand_frame_mean.flatten(2), dim=-1)

        alignment = torch.bmm(cand_flat, tail_norm.transpose(1, 2)).squeeze(-1)

        inter_sim = torch.bmm(cand_flat, cand_flat.transpose(1, 2))
        complement = 1.0 - (inter_sim.sum(dim=-1) - 1.0) / max(n_cand_frames - 1, 1)

        w = cfg.alignment_weight
        score = w * alignment + (1.0 - w) * complement

        k_select = min(history_frames, score.shape[1])
        _, top_frame_idx = torch.topk(score, k_select, dim=1, sorted=False)
        top_frame_idx = torch.sort(top_frame_idx, dim=1).values

        top_token_offsets = top_frame_idx.unsqueeze(-1) * ft + torch.arange(ft, device=self.device).unsqueeze(0).unsqueeze(0)
        top_token_global = (top_token_offsets + cand_start).reshape(B, -1)

        sink_range = torch.arange(0, sink_tokens, device=self.device, dtype=torch.long).unsqueeze(0).expand(B, -1)
        recent_range = torch.arange(cand_end, total_tokens, device=self.device, dtype=torch.long).unsqueeze(0).expand(B, -1)
        keep = torch.cat([sink_range, top_token_global, recent_range], dim=1)

        cache_end_frame = self._block_frame_start(block_idx) + self._chunk_frames(block_idx)
        tail_frame = cache_end_frame - cfg.recent_frames
        desired_hist_start = tail_frame - k_select
        desired_sink_shift = desired_hist_start - cfg.sink_frames

        sink_incr = desired_sink_shift - self._rx_sink_rope_base

        current_hist_base = self._rx_comp_rope_base if self._rx_comp_rope_base != 0 else cand_start // ft
        orig_frame_ids = top_frame_idx + current_hist_base
        desired_frame_ids = torch.arange(desired_hist_start, desired_hist_start + k_select, device=self.device, dtype=torch.long)
        per_frame_delta = desired_frame_ids.unsqueeze(0) - orig_frame_ids

        for cache in kv_caches:
            cache.compact_(keep)
            if cache.k is not None:
                head_dim = cache.k.shape[-1]
                dim_t = head_dim - (head_dim // 6 * 2) * 2

                if sink_tokens > 0 and sink_incr != 0:
                    rope_time_delta_(cache.k[:, :sink_tokens], sink_incr, dim_t)

                hist_start_pos = sink_tokens
                for fi in range(k_select):
                    tok_start = hist_start_pos + fi * ft
                    tok_end = tok_start + ft
                    for b in range(B):
                        delta = int(per_frame_delta[b, fi].item())
                        if delta != 0:
                            rope_time_delta_(cache.k[b : b + 1, tok_start:tok_end], delta, dim_t)

        self._rx_sink_rope_base = desired_sink_shift
        self._rx_comp_rope_base = desired_hist_start

        self._block_ids = []
        self._block_roles = []
        if sink_tokens > 0:
            self._block_ids.append(-1)
            self._block_roles.append("sink")
        if k_select > 0:
            self._block_ids.append(-2)
            self._block_roles.append("history")
        if recent_tokens > 0:
            self._block_ids.append(-3)
            self._block_roles.append("recent")


# ---------------------------------------------------------------------------
# Multi-prompt scene schedule (Infinity-RoPE action control)
# ---------------------------------------------------------------------------


@dataclass
class ActionSegment:
    """One segment from the multi-prompt syntax ``"prompt[Ns]"`` or ``"prompt[Ns#]"``."""

    prompt: str
    duration_seconds: float
    scene_cut: bool = False


def parse_action_prompts(raw: str, fps: int = 16) -> List[ActionSegment]:
    """Parse Infinity-RoPE multi-prompt syntax.

    Format: ``"action1[5s] | action2[10s] | action3[10s#]"``
    - ``[Ns]``  sets the segment duration in seconds.
    - ``#`` inside brackets triggers a hard scene cut (RoPE Cut + KV Flush).
    - ``|`` separates segments.

    If the string contains no ``|`` or ``[``, it is treated as a single segment
    covering the full video.
    """
    if "|" not in raw and "[" not in raw:
        return [ActionSegment(prompt=raw.strip(), duration_seconds=0.0)]

    segments: List[ActionSegment] = []
    for part in raw.split("|"):
        part = part.strip()
        if not part:
            continue
        cut = False
        duration = 0.0
        if "[" in part and "]" in part:
            bracket_start = part.index("[")
            bracket_end = part.index("]")
            spec = part[bracket_start + 1 : bracket_end]
            prompt_text = part[:bracket_start].strip()
            if spec.endswith("#"):
                cut = True
                spec = spec[:-1]
            if spec.endswith("s"):
                spec = spec[:-1]
            try:
                duration = float(spec)
            except ValueError:
                duration = 0.0
        else:
            prompt_text = part
        segments.append(ActionSegment(prompt=prompt_text, duration_seconds=duration, scene_cut=cut))
    return segments if segments else [ActionSegment(prompt=raw.strip(), duration_seconds=0.0)]


def segments_to_num_frames(
    segments: List[ActionSegment],
    fps: int = 16,
    temporal_compression: int = 4,
    first_chunk_frames: int = 1,
    chunk_frames: int = 1,
) -> int:
    """Compute the total pixel-frame count from segment durations.

    The result is rounded up so that the latent-frame count fits the block
    pattern (``first_chunk_frames + N * chunk_frames``).
    """
    total_seconds = sum(s.duration_seconds for s in segments)
    if total_seconds <= 0:
        return 81
    raw_pixel_frames = int(total_seconds * fps)
    latent_frames = 1 + (raw_pixel_frames - 1) // temporal_compression
    remaining = latent_frames - first_chunk_frames
    if remaining < 0:
        remaining = 0
    n_tail_blocks = (remaining + chunk_frames - 1) // chunk_frames
    aligned_latent = first_chunk_frames + n_tail_blocks * chunk_frames
    pixel_frames = (aligned_latent - 1) * temporal_compression + 1
    return pixel_frames


def build_scene_block_schedule(
    segments: List[ActionSegment],
    num_blocks: int,
    first_chunk_frames: int,
    chunk_frames: int,
    fps: int = 16,
    temporal_compression: int = 4,
) -> List[Tuple[int, bool]]:
    """Map action segments to ``(segment_index, scene_cut)`` per block.

    Durations are treated as absolute time.  Each segment's block count is
    computed from its duration, rounded to the nearest block boundary.

    Returns a list of length ``num_blocks``.
    """
    if len(segments) <= 1 or all(s.duration_seconds <= 0 for s in segments):
        return [(0, False)] * num_blocks

    segment_block_counts: List[int] = []
    assigned = 0
    for i, seg in enumerate(segments):
        latent_frames = max(1, round(seg.duration_seconds * fps / temporal_compression))
        if i == 0:
            remaining = max(0, latent_frames - first_chunk_frames)
            n = 1 + (remaining + chunk_frames - 1) // chunk_frames if remaining > 0 else 1
        else:
            n = max(1, (latent_frames + chunk_frames - 1) // chunk_frames)
        if i == len(segments) - 1:
            n = max(1, num_blocks - assigned)
        else:
            n = min(n, num_blocks - assigned - (len(segments) - 1 - i))
            n = max(1, n)
        segment_block_counts.append(n)
        assigned += n

    schedule: List[Tuple[int, bool]] = []
    for seg_idx, (seg, n) in enumerate(zip(segments, segment_block_counts)):
        for j in range(n):
            is_cut = seg.scene_cut and j == 0 and seg_idx > 0
            schedule.append((seg_idx, is_cut))
            if len(schedule) >= num_blocks:
                break
        if len(schedule) >= num_blocks:
            break

    while len(schedule) < num_blocks:
        schedule.append((len(segments) - 1, False))
    return schedule[:num_blocks]


# ---------------------------------------------------------------------------
# Sampling loop
# ---------------------------------------------------------------------------


@torch.no_grad()
def causal_rollout_sampling(
    net,
    init_noise: torch.Tensor,
    t_steps: torch.Tensor,
    condition: dict,
    uncondition: dict | None,
    guidance: float,
    first_chunk_t: int,
    chunk_t: int,
    ode: bool,
    extrapolation_config: ExtrapolationConfig,
    generator: torch.Generator | None = None,
    scene_schedule: Optional[List[Tuple[int, bool]]] = None,
    conditions_per_segment: Optional[List[dict]] = None,
) -> torch.Tensor:
    """Generate video autoregressively with optional multi-prompt scene cuts.

    Args:
        scene_schedule: Per-block ``(segment_index, should_scene_cut)`` list
            from :func:`build_scene_block_schedule`.  When ``None``, single-
            prompt mode is used.
        conditions_per_segment: One ``{"crossattn_emb": ...}`` dict per
            segment.  Required when ``scene_schedule`` is not ``None``.
    """
    B, C, T, H, W = init_noise.shape
    frame_tokens, num_blocks, bp = make_block_pattern(T, H, W, first_chunk_t, chunk_t, net.get_spatial_patch_size())
    ones_B_1 = torch.ones(B, 1, device=init_noise.device, dtype=torch.float64)
    use_cfg = uncondition is not None and guidance > 1.0

    runtime_cond = ExtrapolationRuntime(extrapolation_config, pattern=bp, device=init_noise.device, num_layers=len(net.blocks))
    runtime_uncond = ExtrapolationRuntime(extrapolation_config, pattern=bp, device=init_noise.device, num_layers=len(net.blocks)) if use_cfg else None
    kv_cond = net.allocate_kv_caches(max_len=runtime_cond.cache_max_len(T * frame_tokens))
    kv_uncond = net.allocate_kv_caches(max_len=runtime_uncond.cache_max_len(T * frame_tokens)) if use_cfg else None

    if not ode:
        step_noises = [torch.randn_like(init_noise, dtype=torch.float32, generator=generator) for _ in range(len(t_steps) - 1)]
    else:
        step_noises = None

    x_blocks = []
    num_steps = len(t_steps) - 1
    block_bar = tqdm(range(num_blocks), desc="Chunks", position=0)

    for i in block_bar:
        frame_start, frame_end, block_size = block_span(bp, i)
        attn_meta = AttnMaskSpec(mode="none")

        if scene_schedule is not None and i < len(scene_schedule):
            seg_idx, should_cut = scene_schedule[i]
            if should_cut:
                runtime_cond.request_scene_cut()
                if runtime_uncond is not None:
                    runtime_uncond.request_scene_cut()
            if conditions_per_segment is not None and seg_idx < len(conditions_per_segment):
                condition = conditions_per_segment[seg_idx]

        x = init_noise[:, :, frame_start:frame_end].to(torch.float64) * t_steps[0]

        step_bar = tqdm(zip(t_steps[:-1], t_steps[1:]), total=num_steps, desc=f"  Chunk {i}/{num_blocks}", position=1, leave=False)
        for step_idx, (t_cur, t_next) in enumerate(step_bar):
            t_cur_B_block = repeat(t_cur * ones_B_1, "b 1 -> b t", t=block_size)
            inf_cond = runtime_cond.readonly_state(kv_cond, i)

            v_cond = net(
                x_B_C_T_H_W=x.to(**TENSOR_KWARGS),
                timesteps_B_T=(t_cur_B_block * RECTIFIED_FLOW_T_SCALING).to(**TENSOR_KWARGS),
                **condition,
                inference_state=inf_cond,
                attn_meta=attn_meta,
            ).float()

            if use_cfg:
                inf_uncond = runtime_uncond.readonly_state(kv_uncond, i)
                v_uncond = net(
                    x_B_C_T_H_W=x.to(**TENSOR_KWARGS),
                    timesteps_B_T=(t_cur_B_block * RECTIFIED_FLOW_T_SCALING).to(**TENSOR_KWARGS),
                    **uncondition,
                    inference_state=inf_uncond,
                    attn_meta=attn_meta,
                ).float()
                v_pred = v_uncond + guidance * (v_cond - v_uncond)
            else:
                v_pred = v_cond

            if ode:
                x = x - (t_cur - t_next) * v_pred.to(torch.float64)
            else:
                noise = step_noises[step_idx][:, :, frame_start:frame_end]
                x = (1 - t_next) * (x - t_cur * v_pred.to(torch.float64)) + t_next * noise

        zero_t = (0 * t_cur_B_block).to(**TENSOR_KWARGS)
        inf_append_cond = runtime_cond.append_state(kv_cond, i)
        net(x_B_C_T_H_W=x.to(**TENSOR_KWARGS), timesteps_B_T=zero_t, **condition, inference_state=inf_append_cond, attn_meta=attn_meta)
        runtime_cond.finalize_after_append(kv_cond, i)
        if use_cfg:
            inf_append_uncond = runtime_uncond.append_state(kv_uncond, i)
            net(x_B_C_T_H_W=x.to(**TENSOR_KWARGS), timesteps_B_T=zero_t, **uncondition, inference_state=inf_append_uncond, attn_meta=attn_meta)
            runtime_uncond.finalize_after_append(kv_uncond, i)

        x_blocks.append(x)

    block_bar.close()
    del kv_cond, kv_uncond
    torch.cuda.empty_cache()
    return torch.cat(x_blocks, dim=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_config(args) -> ExtrapolationConfig:
    m = args.extrapolation_method
    if m == "none":
        return None
    if m == "sliding_window":
        return SlidingWindowConfig(window_blocks=args.window_blocks, sink_blocks=args.sw_sink_blocks)
    if m == "rolling_sink":
        return RollingSinkConfig(
            total_blocks=args.total_blocks,
            sink_blocks=args.sink_blocks,
            recent_blocks=args.recent_blocks,
        )
    if m == "infinity_rope":
        return InfinityRopeConfig(
            cache_blocks=args.cache_blocks,
            sink_blocks=args.ir_sink_blocks,
            f_limit=args.f_limit,
            flush_interval_blocks=args.flush_interval_blocks,
            rope_cut_delta=args.rope_cut_delta,
            rope_cut_single_frame=not args.no_rope_cut_single_frame,
        )
    if m == "deep_forcing":
        return DeepForcingConfig(
            sink_frames=args.df_sink_frames,
            budget_frames=args.df_budget_frames,
            recent_frames=args.df_recent_frames,
            observation_layers=args.df_observation_layers,
        )
    if m == "memrope":
        return MemRopeConfig(
            sink_frames=args.mr_sink_frames,
            local_frames=args.mr_local_frames,
            alpha_long=args.mr_alpha_long,
            alpha_short=args.mr_alpha_short,
        )
    if m == "relax_forcing":
        return RelaxForcingConfig(
            sink_frames=args.rx_sink_frames,
            budget_frames=args.rx_budget_frames,
            recent_frames=args.rx_recent_frames,
            alignment_weight=args.rx_alignment_weight,
        )
    raise ValueError(m)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extrapolation-only causal inference script for Wan2.1 T2V")
    parser.add_argument("--model_size", choices=["1.3B", "14B"], default="1.3B")
    parser.add_argument("--distilled", action="store_true", help="Use few-step distilled sampling instead of diffusion ODE")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--num_steps", type=int, default=50, help="ODE steps (non-distilled) or 1-4 (distilled)")
    parser.add_argument("--sigma_max", type=float, default=1600)
    parser.add_argument("--guidance_scale", type=float, default=3.0, help="CFG scale (only used in ODE mode)")
    parser.add_argument("--timestep_shift", type=float, default=3.0, help="Timestep shift for diffusion ODE sampling")
    parser.add_argument("--mid_t", type=float, nargs="*", default=[15 / 16, 5 / 6, 5 / 8])
    parser.add_argument("--first_chunk_t", type=int, default=3)
    parser.add_argument("--chunk_t", type=int, default=3)
    parser.add_argument("--dit_path", type=str, required=True)
    parser.add_argument("--vae_path", type=str, default="assets/checkpoints/Wan2.1_VAE.pth")
    parser.add_argument("--text_encoder_path", type=str, default="assets/checkpoints/models_t5_umt5-xxl-enc-bf16.pth")
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--prompt", type=str, default=_DEFAULT_PROMPT)
    parser.add_argument("--negative_prompt", type=str, default=_DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--resolution", default="480p", type=str)
    parser.add_argument("--aspect_ratio", default="16:9", type=str)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_path", type=str, default="output/causal_extrapolated_video.mp4")

    parser.add_argument(
        "--extrapolation_method",
        choices=["none", "sliding_window", "rolling_sink", "infinity_rope", "deep_forcing", "memrope", "relax_forcing"],
        default="sliding_window",
    )

    g_sw = parser.add_argument_group("sliding_window")
    g_sw.add_argument("--window_blocks", type=int, default=6)
    g_sw.add_argument("--sw_sink_blocks", type=int, default=0, help="Number of initial blocks to keep as permanent sink (0=pure FIFO)")

    g_rs = parser.add_argument_group("rolling_sink")
    g_rs.add_argument("--total_blocks", type=int, default=6, help="K in paper (cache capacity in blocks)")
    g_rs.add_argument("--sink_blocks", type=int, default=5, help="S in paper (S/K=83%%)")
    g_rs.add_argument("--recent_blocks", type=int, default=1, help="K - S")

    g_ir = parser.add_argument_group("infinity_rope")
    g_ir.add_argument("--cache_blocks", type=int, default=6)
    g_ir.add_argument("--ir_sink_blocks", type=int, default=1, help="Global sink blocks")
    g_ir.add_argument("--f_limit", type=int, default=21, help="Teacher max frame horizon")
    g_ir.add_argument("--flush_interval_blocks", type=int, default=0, help="Periodic KV Flush interval (0=disabled)")
    g_ir.add_argument("--rope_cut_delta", type=int, default=21, help="Temporal jump for RoPE Cut (0=disabled)")
    g_ir.add_argument("--no_rope_cut_single_frame", action="store_true", help="Disable RoPE Cut for single-frame chunks (chunk_t=1)")

    g_df = parser.add_argument_group("deep_forcing")
    g_df.add_argument("--df_sink_frames", type=int, default=10, help="S in paper")
    g_df.add_argument("--df_budget_frames", type=int, default=16, help="N in paper")
    g_df.add_argument("--df_recent_frames", type=int, default=4, help="R in paper")
    g_df.add_argument("--df_observation_layers", type=int, default=0, help="0 = all layers (repo-faithful)")

    g_mr = parser.add_argument_group("memrope")
    g_mr.add_argument("--mr_sink_frames", type=int, default=3, help="Permanent sink frames")
    g_mr.add_argument("--mr_local_frames", type=int, default=4, help="Recent local window frames")
    g_mr.add_argument("--mr_alpha_long", type=float, default=0.01, help="EMA decay for long-term memory (identity stream)")
    g_mr.add_argument("--mr_alpha_short", type=float, default=0.1, help="EMA decay for short-term memory (dynamics stream)")

    g_rx = parser.add_argument_group("relax_forcing")
    g_rx.add_argument("--rx_sink_frames", type=int, default=4, help="Permanent sink frames")
    g_rx.add_argument("--rx_budget_frames", type=int, default=16, help="Total cache budget in frames")
    g_rx.add_argument("--rx_recent_frames", type=int, default=4, help="Recent tail frames")
    g_rx.add_argument("--rx_alignment_weight", type=float, default=0.5, help="Weight for alignment vs complementarity scoring [0,1]")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    with init_weights_on_device():
        net = instantiate(DIT_CONFIGS[args.model_size]).eval()

    load_dit_weights(net, args.dit_path)
    log.success(f"Loaded DiT from {args.dit_path}")

    net.to(**TENSOR_KWARGS).cpu()
    torch.cuda.empty_cache()

    tokenizer = Wan2pt1VAEInterface(vae_pth=args.vae_path)
    w, h = VIDEO_RES_SIZE_INFO[args.resolution][args.aspect_ratio]

    segments = parse_action_prompts(args.prompt)
    is_multi_prompt = len(segments) > 1 and any(s.duration_seconds > 0 for s in segments)

    if is_multi_prompt:
        num_frames = segments_to_num_frames(
            segments,
            fps=16,
            temporal_compression=tokenizer.temporal_compression_factor,
            first_chunk_frames=args.first_chunk_t,
            chunk_frames=args.chunk_t,
        )
        total_secs = sum(s.duration_seconds for s in segments)
        log.info(f"Multi-prompt mode: {len(segments)} segments, {total_secs:.1f}s total → {num_frames} pixel frames (--num_frames ignored)")
    else:
        num_frames = args.num_frames

    log.info("Computing text embeddings...")
    if is_multi_prompt:
        conditions_per_segment = []
        for seg in segments:
            emb = get_umt5_embedding(checkpoint_path=args.text_encoder_path, prompts=seg.prompt).to(dtype=torch.bfloat16).cuda()
            conditions_per_segment.append({"crossattn_emb": repeat(emb.to(**TENSOR_KWARGS), "b l d -> (k b) l d", k=args.num_samples)})
        condition = conditions_per_segment[0]
    else:
        text_emb = get_umt5_embedding(checkpoint_path=args.text_encoder_path, prompts=args.prompt).to(dtype=torch.bfloat16).cuda()
        condition = {"crossattn_emb": repeat(text_emb.to(**TENSOR_KWARGS), "b l d -> (k b) l d", k=args.num_samples)}
        conditions_per_segment = None

    if not args.distilled and args.guidance_scale > 1.0:
        neg_text_emb = get_umt5_embedding(checkpoint_path=args.text_encoder_path, prompts=args.negative_prompt).to(dtype=torch.bfloat16).cuda()
        uncondition = {"crossattn_emb": repeat(neg_text_emb.to(**TENSOR_KWARGS), "b l d -> (k b) l d", k=args.num_samples)}
    else:
        uncondition = None

    clear_umt5_memory()

    state_shape = [
        tokenizer.latent_ch,
        tokenizer.get_latent_num_frames(num_frames),
        h // tokenizer.spatial_compression_factor,
        w // tokenizer.spatial_compression_factor,
    ]

    generator = torch.Generator(device=TENSOR_KWARGS["device"])
    generator.manual_seed(args.seed)
    init_noise = torch.randn(args.num_samples, *state_shape, dtype=torch.float32, device=TENSOR_KWARGS["device"], generator=generator)

    extrapolation_config = _build_config(args)

    if args.distilled:
        t_steps = build_few_step_t_steps(args.num_steps, args.sigma_max, args.mid_t, init_noise.device)
        ode = False
        guidance = 1.0
    else:
        t_steps = build_shifted_ode_t_steps(args.num_steps, args.sigma_max, args.timestep_shift, init_noise.device)
        ode = True
        guidance = args.guidance_scale

    T_latent = state_shape[1]
    _, num_blocks, _ = make_block_pattern(T_latent, state_shape[2], state_shape[3], args.first_chunk_t, args.chunk_t, net.get_spatial_patch_size())

    scene_schedule = None
    if is_multi_prompt:
        scene_schedule = build_scene_block_schedule(
            segments,
            num_blocks,
            first_chunk_frames=args.first_chunk_t,
            chunk_frames=args.chunk_t,
        )

    log.info(f"Latent: T={T_latent}, H={state_shape[2]}, W={state_shape[3]}")
    log.info(f"Chunk pattern: first_chunk_t={args.first_chunk_t}, chunk_t={args.chunk_t}, num_blocks={num_blocks}")
    log.info(f"Mode: {'distilled' if args.distilled else 'ODE'}, steps={len(t_steps)-1}, guidance={guidance}")
    log.info(f"Extrapolation: {args.extrapolation_method} / {extrapolation_config}")

    net.cuda()
    samples = causal_rollout_sampling(
        net,
        init_noise,
        t_steps,
        condition,
        uncondition,
        guidance,
        first_chunk_t=args.first_chunk_t,
        chunk_t=args.chunk_t,
        ode=ode,
        extrapolation_config=extrapolation_config,
        generator=generator,
        scene_schedule=scene_schedule,
        conditions_per_segment=conditions_per_segment,
    )
    net.cpu()
    torch.cuda.empty_cache()

    video = tokenizer.decode(samples.float())
    to_show = (1.0 + video.float().cpu().unsqueeze(0).clamp(-1, 1)) / 2.0
    save_image_or_video(rearrange(to_show, "n b c t h w -> c t (n h) (b w)"), args.save_path, fps=16)
    log.success(f"Saved to {args.save_path}")
