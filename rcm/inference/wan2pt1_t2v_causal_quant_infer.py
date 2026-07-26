"""
Causal inference script for Wan2.1 T2V.

Supports both ODE sampling (teacher / non-distilled) and few-step sampling
(distilled) via the --distilled flag, using block-causal attention with
configurable chunk patterns.

This file also provides a training-free, single-file KV-cache quantization
playground. The core model code is left untouched: we swap in local cache
objects that implement the same append/get/reset/compact_ interface used by
the attention stack.

Usage (quantized KV cache):
    python -m rcm.inference.wan2pt1_t2v_causal_quant_infer \
        --distilled --dit_path path/to/distilled.pth \
        --kv_dtype_k int4 --kv_dtype_v fp8 \
        --kv_smoothing_k hadamard --kv_smoothing_v clip \
        --kv_pre_rope_keys
"""

import argparse
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from tqdm import tqdm

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None

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
_DEFAULT_PROMPT = "The camera follows behind a white vintage SUV with a black roof rack as it speeds up a steep dirt road surrounded by pine trees on a steep mountain slope, dust kicks up from it’s tires, the sunlight shines on the SUV as it speeds along the dirt road, casting a warm glow over the scene. The dirt road curves gently into the distance, with no other cars or vehicles in sight. The trees on either side of the road are redwoods, with patches of greenery scattered throughout. The car is seen from the rear following the curve with ease, making it seem as if it is on a rugged drive through the rugged terrain. The dirt road itself is surrounded by steep hills and mountains, with a clear blue sky above with wispy clouds."

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


_FP4_E2M1_CODEBOOK = torch.tensor(
    [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, -0.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
    dtype=torch.float32,
)
# _INT2_MIDRISE_CODEBOOK = torch.tensor([-2, -1, 0, 1], dtype=torch.float32)
_INT2_MIDRISE_CODEBOOK = torch.tensor([-1.5, -0.5, 0.5, 1.5], dtype=torch.float32)


@dataclass(frozen=True)
class TensorQuantSpec:
    dtype_name: str = "bf16"
    smoothing: str = "none"
    group_size: int = 32
    scale_granularity: str = "token"
    smooth_alpha: float = 0.5
    clip_percentile: float = 0.999
    qvg_clusters: int = 256
    qvg_iters: int = 4
    qvg_stages: int = 1
    qvg_share_assignments: bool = True
    qvg_warm_start: bool = True

    @property
    def is_identity(self) -> bool:
        return self.dtype_name == "bf16" and self.smoothing == "none"


@dataclass(frozen=True)
class KVQuantConfig:
    key: TensorQuantSpec
    value: TensorQuantSpec
    pre_rope_keys: bool = False
    dense_prefix_cache: bool = False
    use_triton_restore: bool = False

    @property
    def enabled(self) -> bool:
        return not (self.key.is_identity and self.value.is_identity)


@dataclass
class StoredTensor:
    shape: Tuple[int, int, int, int]
    dtype_name: str
    group_size: int
    scale_granularity: str
    payload: torch.Tensor
    scale: Optional[torch.Tensor]
    smoothing: str
    smooth_meta: Optional[Dict[str, Any]]
    packed_numel: Optional[int] = None
    quant_shape: Optional[Tuple[int, ...]] = None


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


def _effective_group_size(group_size: int, dim: int) -> int:
    if group_size <= 0 or group_size >= dim:
        return dim
    return group_size


def _quant_axis_len(shape: Tuple[int, int, int, int], granularity: str) -> int:
    B, S, H, D = shape
    del B
    if granularity == "token":
        return D
    if granularity == "channel":
        return S
    if granularity == "head":
        return S * D
    if granularity == "tensor":
        return S * H * D
    raise ValueError(f"Unsupported scale granularity: {granularity}")


def _flatten_for_quant(x: torch.Tensor, granularity: str) -> torch.Tensor:
    B, S, H, D = x.shape
    if granularity == "token":
        return x.contiguous().view(B * S * H, D)
    if granularity == "channel":
        return x.permute(0, 2, 3, 1).contiguous().view(B * H * D, S)
    if granularity == "head":
        return x.permute(0, 2, 1, 3).contiguous().view(B * H, S * D)
    if granularity == "tensor":
        return x.contiguous().view(B, S * H * D)
    raise ValueError(f"Unsupported scale granularity: {granularity}")


def _unflatten_from_quant(x_2d: torch.Tensor, shape: Tuple[int, int, int, int], granularity: str) -> torch.Tensor:
    B, S, H, D = shape
    if granularity == "token":
        return x_2d.view(B, S, H, D).contiguous()
    if granularity == "channel":
        return x_2d.view(B, H, D, S).permute(0, 3, 1, 2).contiguous()
    if granularity == "head":
        return x_2d.view(B, H, S, D).permute(0, 2, 1, 3).contiguous()
    if granularity == "tensor":
        return x_2d.view(B, S, H, D).contiguous()
    raise ValueError(f"Unsupported scale granularity: {granularity}")


def _group_last_dim(x_2d: torch.Tensor, group_size: int) -> Tuple[torch.Tensor, int]:
    axis_len = x_2d.shape[-1]
    g = _effective_group_size(group_size, axis_len)
    pad = (g - axis_len % g) % g
    if pad:
        x_2d = F.pad(x_2d, (0, pad))
    return x_2d.view(x_2d.shape[0], -1, g), axis_len


def _ungroup_last_dim(x_groups: torch.Tensor, axis_len: int) -> torch.Tensor:
    return x_groups.reshape(x_groups.shape[0], -1)[..., :axis_len].contiguous()


def _store_scale_tensor(scale: torch.Tensor, prefer_fp8: bool = False) -> torch.Tensor:
    if prefer_fp8 and hasattr(torch, "float8_e4m3fn"):
        return scale.to(torch.float8_e4m3fn)
    if prefer_fp8:
        return scale.to(torch.float16)
    return scale.to(torch.float32)


def _store_assignment_tensor(assignments: torch.Tensor, num_centroids: int) -> torch.Tensor:
    if num_centroids <= 256:
        return assignments.to(torch.uint8)
    if num_centroids <= 32767:
        return assignments.to(torch.int16)
    return assignments.to(torch.int32)


def _prepare_warm_start_centers(
    centers: Optional[torch.Tensor],
    *,
    shape: Tuple[int, ...],
    init_fallback: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if centers is None or tuple(centers.shape) != shape:
        return init_fallback
    return F.normalize(centers.to(device=device, dtype=dtype), dim=-1, eps=1e-6).contiguous()


def _pack_subbyte(values: torch.Tensor, bits: int) -> Tuple[torch.Tensor, int]:
    if bits not in (2, 4):
        raise ValueError(f"Only 2-bit and 4-bit packing are supported, got bits={bits}")
    flat = values.reshape(-1).to(torch.uint8)
    numel = flat.numel()
    values_per_byte = 8 // bits
    pad = (values_per_byte - numel % values_per_byte) % values_per_byte
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)], dim=0)
    packed = flat.new_zeros(flat.numel() // values_per_byte)
    mask = (1 << bits) - 1
    for idx in range(values_per_byte):
        packed |= (flat[idx::values_per_byte] & mask) << (idx * bits)
    return packed.contiguous(), numel


def _unpack_subbyte(packed: torch.Tensor, numel: int, shape: Tuple[int, ...], bits: int) -> torch.Tensor:
    if bits not in (2, 4):
        raise ValueError(f"Only 2-bit and 4-bit unpacking are supported, got bits={bits}")
    flat = packed.reshape(-1)
    values_per_byte = 8 // bits
    out = torch.empty(flat.numel() * values_per_byte, device=flat.device, dtype=torch.uint8)
    mask = (1 << bits) - 1
    for idx in range(values_per_byte):
        out[idx::values_per_byte] = (flat >> (idx * bits)) & mask
    out = out[:numel]
    return out.view(shape)


def _hadamard_last_dim(x: torch.Tensor) -> torch.Tensor:
    n = x.shape[-1]
    if n & (n - 1) != 0:
        raise ValueError(f"Hadamard smoothing requires power-of-two head dim, got {n}")
    y = x.float().reshape(-1, n)
    h = 1
    while h < n:
        y = y.reshape(-1, n // (2 * h), 2, h)
        a = y[:, :, 0, :]
        b = y[:, :, 1, :]
        y = torch.cat([a + b, a - b], dim=-1).reshape(-1, n)
        h *= 2
    return y.reshape(*x.shape[:-1], n) / math.sqrt(n)


def _compute_channel_perm(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # x: [B, S, H, D] -> perm/inv_perm: [B, H, D]
    stats = x.float().abs().mean(dim=1)
    perm = torch.argsort(stats, dim=-1, descending=True)
    inv_perm = torch.argsort(perm, dim=-1)
    return perm, inv_perm


def _apply_channel_perm(x: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    idx = perm[:, None, :, :].expand(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
    return torch.gather(x, dim=-1, index=idx)


def _semantic_kmeans_smoothing(
    x: torch.Tensor,
    num_centroids: int,
    num_iters: int,
    share_assignments: bool,
    warm_start: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    B, S, H, D = x.shape
    if S == 0:
        empty_assign = torch.empty(B, 0, device=x.device, dtype=torch.uint8)
        empty_centroids = x.new_zeros(B, 0, H, D)
        empty_warm = x.new_zeros(B, 0, D)
        return (
            x,
            {"centroids": empty_centroids, "assignments": empty_assign, "share_assignments": True},
            {
                "cluster_centers": empty_warm,
                "share_assignments": True,
            },
        )

    k = min(num_centroids, S)
    if k <= 1:
        centroid = x.mean(dim=1, keepdim=True)
        assignments = torch.zeros(B, S, device=x.device, dtype=torch.long)
        token_centroids = centroid.expand(B, S, H, D)
        if share_assignments:
            warm_centers = F.normalize(x.float().mean(dim=2).mean(dim=1, keepdim=True), dim=-1, eps=1e-6)
            warm_state = {"cluster_centers": warm_centers.to(torch.bfloat16), "share_assignments": True}
        else:
            warm_centers = F.normalize(x.float().permute(0, 2, 1, 3).mean(dim=2, keepdim=True), dim=-1, eps=1e-6)
            warm_state = {"cluster_centers": warm_centers.to(torch.bfloat16), "share_assignments": False}
        return (
            x - token_centroids,
            {
                "centroids": centroid.to(torch.bfloat16),
                "assignments": _store_assignment_tensor(assignments, k),
                "share_assignments": True,
            },
            warm_state,
        )

    init_idx = torch.linspace(0, S - 1, steps=k, device=x.device).round().long()

    if share_assignments:
        feat = F.normalize(x.float().mean(dim=2), dim=-1, eps=1e-6)
        centers = _prepare_warm_start_centers(
            None if warm_start is None else warm_start.get("cluster_centers"),
            shape=(B, k, D),
            init_fallback=feat[:, init_idx].contiguous(),
            dtype=feat.dtype,
            device=feat.device,
        )

        for _ in range(max(1, num_iters)):
            dist = torch.cdist(feat, centers)
            assignments = dist.argmin(dim=-1)
            onehot = F.one_hot(assignments, num_classes=k).to(feat.dtype)
            counts = onehot.sum(dim=1, keepdim=False).clamp(min=1.0)
            centers = torch.einsum("bsk,bsd->bkd", onehot, feat) / counts.unsqueeze(-1)
            centers = F.normalize(centers, dim=-1, eps=1e-6)

        onehot_x = F.one_hot(assignments, num_classes=k).to(x.dtype)
        counts_x = onehot_x.sum(dim=1, keepdim=False).clamp(min=1)
        centroids = torch.einsum("bsk,bshd->bkhd", onehot_x, x) / counts_x.unsqueeze(-1).unsqueeze(-1)
        batch_idx = torch.arange(B, device=x.device)[:, None]
        token_centroids = centroids[batch_idx, assignments]
        residual = x - token_centroids
        meta = {
            "centroids": centroids.to(torch.bfloat16),
            "assignments": _store_assignment_tensor(assignments, k),
            "share_assignments": True,
        }
        warm_state = {"cluster_centers": centers.to(torch.bfloat16), "share_assignments": True}
        return residual, meta, warm_state

    x_heads = x.float().permute(0, 2, 1, 3).contiguous()
    feat = F.normalize(x_heads.reshape(B * H, S, D), dim=-1, eps=1e-6)
    default_centers = feat[:, init_idx].contiguous()
    warm_centers = None if warm_start is None else warm_start.get("cluster_centers")
    if warm_centers is not None and tuple(warm_centers.shape) == (B, H, k, D):
        warm_centers = warm_centers.view(B * H, k, D)
    else:
        warm_centers = None
    centers = _prepare_warm_start_centers(
        warm_centers,
        shape=(B * H, k, D),
        init_fallback=default_centers,
        dtype=feat.dtype,
        device=feat.device,
    )

    for _ in range(max(1, num_iters)):
        dist = torch.cdist(feat, centers)
        assignments = dist.argmin(dim=-1)
        onehot = F.one_hot(assignments, num_classes=k).to(feat.dtype)
        counts = onehot.sum(dim=1, keepdim=False).clamp(min=1.0)
        centers = torch.einsum("nsk,nsd->nkd", onehot, feat) / counts.unsqueeze(-1)
        centers = F.normalize(centers, dim=-1, eps=1e-6)

    onehot_x = F.one_hot(assignments, num_classes=k).to(x_heads.dtype)
    counts_x = onehot_x.sum(dim=1, keepdim=False).clamp(min=1)
    centroids = torch.einsum("nsk,nsd->nkd", onehot_x, x_heads.reshape(B * H, S, D)) / counts_x.unsqueeze(-1)
    token_centroids = centroids[torch.arange(B * H, device=x.device)[:, None], assignments]
    residual = x_heads.reshape(B * H, S, D) - token_centroids
    residual = residual.view(B, H, S, D).permute(0, 2, 1, 3).contiguous()
    meta = {
        "centroids": centroids.view(B, H, k, D).to(torch.bfloat16),
        "assignments": _store_assignment_tensor(assignments.view(B, H, S), k),
        "share_assignments": False,
    }
    warm_state = {"cluster_centers": centers.view(B, H, k, D).to(torch.bfloat16), "share_assignments": False}
    return residual, meta, warm_state


def _progressive_qvg_smoothing(
    x: torch.Tensor,
    spec: TensorQuantSpec,
    warm_start: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, Dict[str, Any], Dict[str, Any]]:
    residual = x
    stages = []
    next_stage_states = []
    warm_stages = [] if warm_start is None else warm_start.get("stages", [])
    for stage_idx in range(max(1, spec.qvg_stages)):
        stage_warm = warm_stages[stage_idx] if stage_idx < len(warm_stages) else None
        residual, stage_meta, stage_state = _semantic_kmeans_smoothing(
            residual,
            num_centroids=spec.qvg_clusters,
            num_iters=spec.qvg_iters,
            share_assignments=spec.qvg_share_assignments,
            warm_start=stage_warm,
        )
        stages.append(stage_meta)
        next_stage_states.append(stage_state)
    return residual, {"stages": stages}, {"stages": next_stage_states}


def _channel_balance_smoothing(x: torch.Tensor, alpha: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    stats = x.float().abs().mean(dim=1, keepdim=True).clamp(min=1e-6)
    ref = stats.mean(dim=-1, keepdim=True).clamp(min=1e-6)
    scale = (stats / ref).pow(alpha).clamp(min=1 / 16, max=16)
    return x / scale, {"channel_scale": scale.to(torch.bfloat16)}


def _apply_smoothing_encode(
    x: torch.Tensor,
    spec: TensorQuantSpec,
    warm_start: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    if spec.smoothing == "none":
        return x, None, None

    if spec.smoothing == "clip":
        flat = x.float().abs().transpose(1, 2).reshape(x.shape[0], x.shape[2], -1)
        q = min(max(spec.clip_percentile, 0.0), 1.0)
        thr = torch.quantile(flat, q, dim=-1).view(x.shape[0], 1, x.shape[2], 1)
        return x.clamp(min=-thr, max=thr), {"threshold": thr.to(torch.bfloat16)}, None

    if spec.smoothing == "balance":
        x_out, meta = _channel_balance_smoothing(x, alpha=spec.smooth_alpha)
        return x_out, meta, None

    if spec.smoothing == "hadamard":
        return _hadamard_last_dim(x), {"dim": torch.tensor([x.shape[-1]], device=x.device, dtype=torch.int32)}, None

    if spec.smoothing == "reorder":
        perm, inv_perm = _compute_channel_perm(x)
        return _apply_channel_perm(x, perm), {"perm": perm.to(torch.int16), "inv_perm": inv_perm.to(torch.int16)}, None

    if spec.smoothing == "qvg":
        x_out, meta, next_state = _progressive_qvg_smoothing(
            x,
            spec,
            warm_start=warm_start if spec.qvg_warm_start else None,
        )
        return x_out, meta, next_state if spec.qvg_warm_start else None

    raise ValueError(f"Unknown smoothing strategy: {spec.smoothing}")


def _apply_smoothing_decode(x: torch.Tensor, spec: TensorQuantSpec, meta: Optional[Dict[str, Any]]) -> torch.Tensor:
    if spec.smoothing == "none" or meta is None:
        return x

    if spec.smoothing == "clip":
        return x

    if spec.smoothing == "balance":
        return x * meta["channel_scale"].to(device=x.device, dtype=torch.float32)

    if spec.smoothing == "hadamard":
        return _hadamard_last_dim(x)

    if spec.smoothing == "reorder":
        inv_perm = meta["inv_perm"].to(device=x.device, dtype=torch.long)
        return _apply_channel_perm(x, inv_perm)

    if spec.smoothing == "qvg":
        return _apply_qvg_stages_decode(x, reversed(meta["stages"]))

    raise ValueError(f"Unknown smoothing strategy: {spec.smoothing}")


def _apply_qvg_stage_decode(x: torch.Tensor, stage_meta: Dict[str, Any]) -> torch.Tensor:
    centroids = stage_meta["centroids"].to(device=x.device, dtype=torch.float32)
    assignments = stage_meta["assignments"].to(device=x.device, dtype=torch.long)
    if stage_meta.get("share_assignments", True):
        batch_idx = torch.arange(x.shape[0], device=x.device)[:, None]
        token_centroids = centroids[batch_idx, assignments]
        return x + token_centroids
    B, S, H, D = x.shape
    x_heads = x.permute(0, 2, 1, 3).contiguous()
    batch_idx = torch.arange(B, device=x.device)[:, None, None]
    head_idx = torch.arange(H, device=x.device)[None, :, None]
    token_centroids = centroids[batch_idx, head_idx, assignments]
    return (x_heads + token_centroids).permute(0, 2, 1, 3).contiguous()


def _apply_qvg_stages_decode(x: torch.Tensor, stage_metas) -> torch.Tensor:
    for stage_meta in stage_metas:
        x = _apply_qvg_stage_decode(x, stage_meta)
    return x


def _quantize_uniform_int(x: torch.Tensor, bits: int, group_size: int, granularity: str) -> Tuple[torch.Tensor, torch.Tensor]:
    x_2d = _flatten_for_quant(x.float(), granularity)
    x_groups, _ = _group_last_dim(x_2d, group_size)
    qmin = -(1 << (bits - 1))
    qmax = (1 << (bits - 1)) - 1
    scale = x_groups.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / max(qmax, 1)
    q = torch.round(x_groups / scale).clamp(qmin, qmax).to(torch.int8)
    return q, scale.to(torch.float32)


def _quantize_int2_midrise(x: torch.Tensor, group_size: int, granularity: str) -> Tuple[torch.Tensor, torch.Tensor]:
    x_2d = _flatten_for_quant(x.float(), granularity)
    x_groups, _ = _group_last_dim(x_2d, group_size)
    codebook = _INT2_MIDRISE_CODEBOOK.to(device=x.device)
    scale = x_groups.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / codebook.abs().max()
    normalized = (x_groups / scale).clamp(codebook.min().item(), codebook.max().item())
    idx = torch.argmin((normalized.unsqueeze(-1) - codebook.view(*(1 for _ in range(normalized.ndim)), -1)).abs(), dim=-1).to(torch.uint8)
    return idx, scale.to(torch.float32)


def _dequantize_uniform_int(
    q: torch.Tensor,
    scale: torch.Tensor,
    shape: Tuple[int, int, int, int],
    granularity: str,
) -> torch.Tensor:
    axis_len = _quant_axis_len(shape, granularity)
    x_groups = q.float() * scale.float()
    x_2d = _ungroup_last_dim(x_groups, axis_len)
    return _unflatten_from_quant(x_2d, shape, granularity)


def _dequantize_int2_midrise(
    idx: torch.Tensor,
    scale: torch.Tensor,
    shape: Tuple[int, int, int, int],
    granularity: str,
) -> torch.Tensor:
    axis_len = _quant_axis_len(shape, granularity)
    codebook = _INT2_MIDRISE_CODEBOOK.to(device=idx.device)
    x_groups = codebook[idx.long()] * scale.float()
    x_2d = _ungroup_last_dim(x_groups, axis_len)
    return _unflatten_from_quant(x_2d, shape, granularity)


def _quantize_fp8_scaled(x: torch.Tensor, group_size: int, granularity: str) -> Tuple[torch.Tensor, torch.Tensor]:
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("Requested fp8 KV cache, but torch.float8_e4m3fn is unavailable in this environment.")
    x_2d = _flatten_for_quant(x.float(), granularity)
    xg, _ = _group_last_dim(x_2d, group_size)
    fp8_max = 448.0
    scale = xg.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / fp8_max
    q = (xg / scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
    return q, scale.to(torch.float32)


def _dequantize_fp8_scaled(
    q: torch.Tensor,
    scale: torch.Tensor,
    shape: Tuple[int, int, int, int],
    granularity: str,
) -> torch.Tensor:
    axis_len = _quant_axis_len(shape, granularity)
    out = q.to(torch.float32) * scale.float()
    x_2d = _ungroup_last_dim(out, axis_len)
    return _unflatten_from_quant(x_2d, shape, granularity)


def _quantize_fp4_scaled(x: torch.Tensor, group_size: int, granularity: str) -> Tuple[torch.Tensor, torch.Tensor, int, Tuple[int, ...]]:
    x_2d = _flatten_for_quant(x.float(), granularity)
    xg, _ = _group_last_dim(x_2d, group_size)
    codebook = _FP4_E2M1_CODEBOOK.to(device=x.device)
    scale = xg.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / codebook.abs().max()
    normalized = (xg / scale).clamp(codebook.min().item(), codebook.max().item())
    idx = torch.argmin((normalized.unsqueeze(-1) - codebook.view(*(1 for _ in range(normalized.ndim)), -1)).abs(), dim=-1).to(torch.uint8)
    packed, numel = _pack_subbyte(idx, bits=4)
    return packed, _store_scale_tensor(scale, prefer_fp8=True), numel, tuple(idx.shape)


def _dequantize_fp4_scaled(
    packed: torch.Tensor,
    scale: torch.Tensor,
    packed_numel: int,
    quant_shape: Tuple[int, ...],
    shape: Tuple[int, int, int, int],
    granularity: str,
) -> torch.Tensor:
    idx = _unpack_subbyte(packed, packed_numel, quant_shape, bits=4).to(torch.long)
    codebook = _FP4_E2M1_CODEBOOK.to(device=packed.device)
    out = codebook[idx] * scale.float()
    axis_len = _quant_axis_len(shape, granularity)
    x_2d = _ungroup_last_dim(out, axis_len)
    return _unflatten_from_quant(x_2d, shape, granularity)


if triton is not None:  # pragma: no cover

    @triton.jit
    def _restore_packed_token_kernel(
        PACKED,
        SCALE,
        ASSIGNMENTS,
        CENTROIDS,
        OUT,
        ROWS,
        D,
        S_TOKENS,
        H_HEADS,
        K_CENTROIDS,
        NUM_GROUPS,
        BITS: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        HAS_QVG: tl.constexpr,
        SHARE_ASSIGNMENTS: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_row = tl.program_id(0).to(tl.int64)
        pid_col = tl.program_id(1).to(tl.int64)

        offs_d = pid_col * BLOCK_D + tl.arange(0, BLOCK_D)
        mask = (pid_row < ROWS) & (offs_d < D)

        values_per_byte = 8 // BITS
        group_idx = offs_d // GROUP_SIZE
        elem_in_group = offs_d % GROUP_SIZE
        linear_elem = (pid_row * NUM_GROUPS + group_idx).to(tl.int64) * GROUP_SIZE + elem_in_group.to(tl.int64)
        byte_idx = linear_elem // values_per_byte
        bit_shift = (linear_elem % values_per_byte) * BITS
        packed = tl.load(PACKED + byte_idx, mask=mask, other=0).to(tl.int32)
        code = (packed >> bit_shift) & ((1 << BITS) - 1)

        scale = tl.load(SCALE + pid_row * NUM_GROUPS + group_idx, mask=mask, other=0.0).to(tl.float32)
        if BITS == 4:
            value = (code.to(tl.int32) - 8).to(tl.float32) * scale
        else:
            value = (code.to(tl.float32) - 1.5) * scale

        if HAS_QVG:
            h = pid_row % H_HEADS
            tmp = pid_row // H_HEADS
            s = tmp % S_TOKENS
            b = tmp // S_TOKENS
            if SHARE_ASSIGNMENTS:
                assign = tl.load(ASSIGNMENTS + b * S_TOKENS + s).to(tl.int32)
                centroid_offset = (((b * K_CENTROIDS + assign) * H_HEADS + h) * D + offs_d).to(tl.int64)
            else:
                assign = tl.load(ASSIGNMENTS + ((b * H_HEADS + h) * S_TOKENS + s)).to(tl.int32)
                centroid_offset = ((((b * H_HEADS + h) * K_CENTROIDS + assign) * D) + offs_d).to(tl.int64)
            value += tl.load(CENTROIDS + centroid_offset, mask=mask, other=0.0).to(tl.float32)

        tl.store(OUT + pid_row * D + offs_d, value.to(OUT.type.element_ty), mask=mask)


def _triton_block_d(dim: int) -> int:
    if dim <= 32:
        return 32
    if dim <= 64:
        return 64
    if dim <= 128:
        return 128
    return 256


def _try_restore_tensor_triton(blob: StoredTensor, spec: TensorQuantSpec) -> Optional[torch.Tensor]:
    if triton is None or not blob.payload.is_cuda:
        return None
    if blob.dtype_name not in {"int2", "int4"}:
        return None
    if blob.scale_granularity != "token" or blob.quant_shape is None or len(blob.quant_shape) != 3:
        return None

    B, S, H, D = blob.shape
    rows, num_groups, packed_group = blob.quant_shape
    if rows != B * S * H or packed_group != blob.group_size:
        return None

    qvg_stage = None
    remaining_qvg_stages = None
    if spec.smoothing == "qvg":
        if blob.smooth_meta is None or not blob.smooth_meta.get("stages"):
            return None
        decode_stages = list(reversed(blob.smooth_meta["stages"]))
        qvg_stage = decode_stages[0]
        remaining_qvg_stages = decode_stages[1:]

    scale = blob.scale
    if scale is None:
        return None
    if scale.dtype == getattr(torch, "float8_e4m3fn", None):
        scale = scale.to(torch.float16)
    scale = scale.contiguous().view(rows, num_groups)

    assignments = blob.payload.new_empty(1, dtype=torch.int32)
    centroids = blob.payload.new_empty(1, dtype=torch.bfloat16)
    has_qvg = qvg_stage is not None
    share_assignments = True
    k_centroids = 1
    if has_qvg:
        assignments = qvg_stage["assignments"].to(device=blob.payload.device, dtype=torch.int32).contiguous()
        centroids = qvg_stage["centroids"].to(device=blob.payload.device).contiguous()
        share_assignments = bool(qvg_stage.get("share_assignments", True))
        k_centroids = int(centroids.shape[1] if share_assignments else centroids.shape[2])

    out = torch.empty((rows, D), device=blob.payload.device, dtype=torch.bfloat16)
    bits = 2 if blob.dtype_name == "int2" else 4
    block_d = _triton_block_d(D)
    grid = (rows, triton.cdiv(D, block_d))
    _restore_packed_token_kernel[grid](
        blob.payload,
        scale,
        assignments,
        centroids,
        out,
        rows,
        D,
        S,
        H,
        k_centroids,
        num_groups,
        BITS=bits,
        GROUP_SIZE=blob.group_size,
        HAS_QVG=has_qvg,
        SHARE_ASSIGNMENTS=share_assignments,
        BLOCK_D=block_d,
        num_warps=4 if D <= 64 else 8,
        num_stages=3,
    )
    x = out.view(B, S, H, D).to(torch.float32)
    if spec.smoothing == "qvg":
        if remaining_qvg_stages:
            x = _apply_qvg_stages_decode(x, remaining_qvg_stages)
        return x.to(torch.bfloat16).contiguous()
    x = _apply_smoothing_decode(x, spec, blob.smooth_meta)
    return x.to(torch.bfloat16).contiguous()


def _store_tensor(
    x: torch.Tensor,
    spec: TensorQuantSpec,
    qvg_warm_start: Optional[Dict[str, Any]] = None,
) -> Tuple[StoredTensor, Optional[Dict[str, Any]]]:
    x_smoothed, smooth_meta, next_qvg_warm_start = _apply_smoothing_encode(x.detach(), spec, warm_start=qvg_warm_start)
    shape = tuple(x.shape)
    group_size = _effective_group_size(spec.group_size, _quant_axis_len(shape, spec.scale_granularity))

    if spec.dtype_name == "bf16":
        return (
            StoredTensor(
                shape=shape,
                dtype_name=spec.dtype_name,
                group_size=group_size,
                scale_granularity=spec.scale_granularity,
                payload=x_smoothed.to(torch.bfloat16).contiguous(),
                scale=None,
                smoothing=spec.smoothing,
                smooth_meta=smooth_meta,
            ),
            next_qvg_warm_start,
        )

    if spec.dtype_name == "int8":
        q, scale = _quantize_uniform_int(x_smoothed, bits=8, group_size=spec.group_size, granularity=spec.scale_granularity)
        return (
            StoredTensor(
                shape=shape,
                dtype_name=spec.dtype_name,
                group_size=group_size,
                scale_granularity=spec.scale_granularity,
                payload=q.contiguous(),
                scale=_store_scale_tensor(scale, prefer_fp8=True),
                smoothing=spec.smoothing,
                smooth_meta=smooth_meta,
            ),
            next_qvg_warm_start,
        )

    if spec.dtype_name == "int2":
        idx, scale = _quantize_int2_midrise(x_smoothed, group_size=spec.group_size, granularity=spec.scale_granularity)
        packed, numel = _pack_subbyte(idx, bits=2)
        return (
            StoredTensor(
                shape=shape,
                dtype_name=spec.dtype_name,
                group_size=group_size,
                scale_granularity=spec.scale_granularity,
                payload=packed,
                scale=_store_scale_tensor(scale, prefer_fp8=True),
                smoothing=spec.smoothing,
                smooth_meta=smooth_meta,
                packed_numel=numel,
                quant_shape=tuple(idx.shape),
            ),
            next_qvg_warm_start,
        )

    if spec.dtype_name == "int4":
        q, scale = _quantize_uniform_int(x_smoothed, bits=4, group_size=spec.group_size, granularity=spec.scale_granularity)
        q_u = (q + 8).to(torch.uint8)
        packed, numel = _pack_subbyte(q_u, bits=4)
        return (
            StoredTensor(
                shape=shape,
                dtype_name=spec.dtype_name,
                group_size=group_size,
                scale_granularity=spec.scale_granularity,
                payload=packed,
                scale=_store_scale_tensor(scale, prefer_fp8=True),
                smoothing=spec.smoothing,
                smooth_meta=smooth_meta,
                packed_numel=numel,
                quant_shape=tuple(q.shape),
            ),
            next_qvg_warm_start,
        )

    if spec.dtype_name == "fp8":
        q, scale = _quantize_fp8_scaled(x_smoothed, group_size=spec.group_size, granularity=spec.scale_granularity)
        return (
            StoredTensor(
                shape=shape,
                dtype_name=spec.dtype_name,
                group_size=group_size,
                scale_granularity=spec.scale_granularity,
                payload=q.contiguous(),
                scale=_store_scale_tensor(scale, prefer_fp8=True),
                smoothing=spec.smoothing,
                smooth_meta=smooth_meta,
            ),
            next_qvg_warm_start,
        )

    if spec.dtype_name == "fp4":
        packed, scale, numel, quant_shape = _quantize_fp4_scaled(
            x_smoothed,
            group_size=spec.group_size,
            granularity=spec.scale_granularity,
        )
        return (
            StoredTensor(
                shape=shape,
                dtype_name=spec.dtype_name,
                group_size=group_size,
                scale_granularity=spec.scale_granularity,
                payload=packed,
                scale=scale,
                smoothing=spec.smoothing,
                smooth_meta=smooth_meta,
                packed_numel=numel,
                quant_shape=quant_shape,
            ),
            next_qvg_warm_start,
        )

    raise ValueError(f"Unsupported kv dtype: {spec.dtype_name}")


def _restore_tensor(blob: StoredTensor, spec: TensorQuantSpec, prefer_triton: bool = False) -> torch.Tensor:
    if prefer_triton:
        x_fast = _try_restore_tensor_triton(blob, spec)
        if x_fast is not None:
            return x_fast
    if blob.dtype_name == "bf16":
        x = blob.payload.to(torch.float32)
    elif blob.dtype_name == "int8":
        x = _dequantize_uniform_int(blob.payload.to(torch.int8), blob.scale, blob.shape, blob.scale_granularity)
    elif blob.dtype_name == "int2":
        idx = _unpack_subbyte(blob.payload, blob.packed_numel, blob.quant_shape, bits=2)
        x = _dequantize_int2_midrise(idx, blob.scale, blob.shape, blob.scale_granularity)
    elif blob.dtype_name == "int4":
        q_u = _unpack_subbyte(blob.payload, blob.packed_numel, blob.quant_shape, bits=4)
        q = q_u.to(torch.int16) - 8
        x = _dequantize_uniform_int(q.to(torch.int8), blob.scale, blob.shape, blob.scale_granularity)
    elif blob.dtype_name == "fp8":
        x = _dequantize_fp8_scaled(blob.payload, blob.scale, blob.shape, blob.scale_granularity)
    elif blob.dtype_name == "fp4":
        x = _dequantize_fp4_scaled(
            blob.payload,
            blob.scale,
            blob.packed_numel,
            blob.quant_shape,
            blob.shape,
            blob.scale_granularity,
        )
    else:
        raise ValueError(f"Unsupported stored dtype: {blob.dtype_name}")
    x = _apply_smoothing_decode(x, spec, blob.smooth_meta)
    return x.to(torch.bfloat16).contiguous()


class QuantizedKVCache:
    """
    Chunked cache that stores each appended chunk in a compact representation and
    dequantizes prefixes on demand. The hottest path in this script repeatedly
    queries the same prefix within a denoising block, so we always keep a
    single incrementally extendable dense prefix cache. Optional memoization of
    additional prefixes remains available for even faster repeated reads.
    """

    def __init__(
        self,
        max_len: int,
        key_spec: TensorQuantSpec,
        value_spec: TensorQuantSpec,
        use_dense_prefix_cache: bool = False,
        use_triton_restore: bool = False,
    ):
        self.max_len = max_len
        self.key_spec = key_spec
        self.value_spec = value_spec
        self.use_dense_prefix_cache = use_dense_prefix_cache
        self.use_triton_restore = use_triton_restore
        self.cur = 0
        self.k = None
        self.v = None
        self._chunk_lens: List[int] = []
        self._cum_ends: List[int] = []
        self._chunks: List[Tuple[StoredTensor, StoredTensor]] = []
        self._prefix_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._last_prefix_chunks = 0
        self._last_prefix: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self._key_qvg_warm_start: Optional[Dict[str, Any]] = None
        self._value_qvg_warm_start: Optional[Dict[str, Any]] = None

    def reset(self, free_buffers: bool = True):
        del free_buffers
        self.cur = 0
        self.k = None
        self.v = None
        self._chunk_lens = []
        self._cum_ends = []
        self._chunks = []
        self._prefix_cache = {}
        self._last_prefix_chunks = 0
        self._last_prefix = None
        self._key_qvg_warm_start = None
        self._value_qvg_warm_start = None

    def ensure(self, B: int, H: int, D: int, dtype: torch.dtype, device: torch.device):
        del B, H, D, dtype, device
        return

    def append(self, k: torch.Tensor, v: torch.Tensor) -> Tuple[int, int]:
        B, s, H, D = k.shape
        del B, H, D
        if self.cur + s > self.max_len:
            raise RuntimeError(f"KV cache overflow: cur={self.cur}, append={s}, max_len={self.max_len}")
        start = self.cur
        end = start + s
        blob_k, next_key_warm = _store_tensor(k, self.key_spec, qvg_warm_start=self._key_qvg_warm_start)
        blob_v, next_value_warm = _store_tensor(v, self.value_spec, qvg_warm_start=self._value_qvg_warm_start)
        self._chunks.append((blob_k, blob_v))
        self._key_qvg_warm_start = next_key_warm
        self._value_qvg_warm_start = next_value_warm
        self.cur = end
        self._chunk_lens.append(s)
        self._cum_ends.append(self.cur)
        self._prefix_cache.clear()
        return start, end

    def _materialize_prefix(self, n_chunks: int) -> Tuple[torch.Tensor, torch.Tensor]:
        key = n_chunks
        if self.use_dense_prefix_cache and key in self._prefix_cache:
            return self._prefix_cache[key]

        if self._last_prefix is not None and self._last_prefix_chunks == n_chunks:
            return self._last_prefix

        if self._last_prefix is not None and 0 < self._last_prefix_chunks < n_chunks:
            last_k, last_v = self._last_prefix
            k_suffix = [
                _restore_tensor(blob_k, self.key_spec, prefer_triton=self.use_triton_restore)
                for blob_k, _ in self._chunks[self._last_prefix_chunks : n_chunks]
            ]
            v_suffix = [
                _restore_tensor(blob_v, self.value_spec, prefer_triton=self.use_triton_restore)
                for _, blob_v in self._chunks[self._last_prefix_chunks : n_chunks]
            ]
            if k_suffix:
                k_cat = torch.cat([last_k, *k_suffix], dim=1).contiguous()
                v_cat = torch.cat([last_v, *v_suffix], dim=1).contiguous()
            else:
                k_cat, v_cat = last_k, last_v
        else:
            k_parts = [_restore_tensor(blob_k, self.key_spec, prefer_triton=self.use_triton_restore) for blob_k, _ in self._chunks[:n_chunks]]
            v_parts = [_restore_tensor(blob_v, self.value_spec, prefer_triton=self.use_triton_restore) for _, blob_v in self._chunks[:n_chunks]]
            k_cat = torch.cat(k_parts, dim=1).contiguous()
            v_cat = torch.cat(v_parts, dim=1).contiguous()

        self._last_prefix_chunks = n_chunks
        self._last_prefix = (k_cat, v_cat)
        if self.use_dense_prefix_cache:
            self._prefix_cache[key] = (k_cat, v_cat)
        return k_cat, v_cat

    def get(self, block_range: Optional[int] = None) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.cur == 0 or not self._chunks:
            return None, None
        if block_range is None:
            return self._materialize_prefix(len(self._chunks))
        if block_range < 0:
            raise ValueError(f"block_range must be >= 0, got {block_range}")
        if block_range == 0:
            return None, None
        if block_range > len(self._chunks):
            raise IndexError(f"block_range={block_range} exceeds cached chunks={len(self._chunks)}")
        return self._materialize_prefix(block_range)

    def compact_(self, keep_indices: Optional[torch.Tensor]):
        if keep_indices is None or self.cur == 0:
            return
        keep_indices = keep_indices.to(dtype=torch.long)
        if keep_indices.ndim != 1:
            raise NotImplementedError("QuantizedKVCache.compact_ currently supports only 1D keep indices in this script.")
        k_full, v_full = self.get()
        if k_full is None or v_full is None:
            return
        if keep_indices.numel() == 0:
            self.reset()
            return
        k_new = k_full.index_select(dim=1, index=keep_indices.to(device=k_full.device))
        v_new = v_full.index_select(dim=1, index=keep_indices.to(device=v_full.device))
        self.reset()
        self.append(k_new, v_new)

    @property
    def current_len(self) -> int:
        return self.cur


def _allocate_kv_caches(net, max_len: int, quant_cfg: KVQuantConfig):
    if not quant_cfg.enabled:
        return net.allocate_kv_caches(max_len=max_len)
    return [
        QuantizedKVCache(
            max_len=max_len,
            key_spec=quant_cfg.key,
            value_spec=quant_cfg.value,
            use_dense_prefix_cache=quant_cfg.dense_prefix_cache,
            use_triton_restore=quant_cfg.use_triton_restore,
        )
        for _ in range(len(net.blocks))
    ]


def _sequential_t_indices(pattern: BlockPattern, block_idx: int, device: torch.device) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    frame_tokens = pattern.frame_tokens
    frame_start = pattern.blocks_to_frames(block_idx)
    chunk_frames = pattern.block_size(block_idx)
    current_frames = torch.arange(frame_start, frame_start + chunk_frames, device=device, dtype=torch.long)
    current_t = torch.repeat_interleave(current_frames, repeats=frame_tokens)
    if frame_start == 0:
        return None, current_t
    cached_frames = torch.arange(0, frame_start, device=device, dtype=torch.long)
    cached_t = torch.repeat_interleave(cached_frames, repeats=frame_tokens)
    return cached_t, current_t


def _make_inference_state(
    mode: KVCacheMode,
    kv_caches,
    pattern: BlockPattern,
    block_cursor: int,
    use_pre_rope_keys: bool,
    device: torch.device,
):
    if not use_pre_rope_keys:
        return CausalInferenceState(mode=mode, kv_caches=kv_caches, pattern=pattern, block_cursor=block_cursor)
    cached_t, current_t = _sequential_t_indices(pattern, block_cursor, device=device)
    return CausalInferenceState(
        mode=mode,
        kv_caches=kv_caches,
        pattern=pattern,
        block_cursor=block_cursor,
        cached_t_indices=cached_t,
        current_t_indices=current_t,
    )


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
    generator: torch.Generator | None = None,
    kv_quant_cfg: KVQuantConfig | None = None,
) -> torch.Tensor:
    kv_quant_cfg = kv_quant_cfg or KVQuantConfig(TensorQuantSpec(), TensorQuantSpec(), pre_rope_keys=False)
    B, C, T, H, W = init_noise.shape
    frame_tokens, num_blocks, bp = make_block_pattern(T, H, W, first_chunk_t, chunk_t, net.get_spatial_patch_size())
    ones_B_1 = torch.ones(B, 1, device=init_noise.device, dtype=torch.float64)
    use_cfg = uncondition is not None and guidance > 1.0

    kv_cond = _allocate_kv_caches(net, max_len=T * frame_tokens, quant_cfg=kv_quant_cfg)
    kv_uncond = _allocate_kv_caches(net, max_len=T * frame_tokens, quant_cfg=kv_quant_cfg) if use_cfg else None

    if not ode:
        step_noises = [torch.randn(init_noise.shape, dtype=torch.float32, device=init_noise.device, generator=generator) for _ in range(len(t_steps) - 1)]
    else:
        step_noises = None

    x_blocks = []
    num_steps = len(t_steps) - 1
    block_bar = tqdm(range(num_blocks), desc="Chunks", position=0)

    for i in block_bar:
        frame_start, frame_end, block_size = block_span(bp, i)
        attn_meta = AttnMaskSpec(mode="block_causal", pattern=bp, q_block_offset=i)

        x = init_noise[:, :, frame_start:frame_end].to(torch.float64) * t_steps[0]

        step_bar = tqdm(zip(t_steps[:-1], t_steps[1:]), total=num_steps, desc=f"  Chunk {i}/{num_blocks}", position=1, leave=False)
        for step_idx, (t_cur, t_next) in enumerate(step_bar):
            t_cur_B_block = repeat(t_cur * ones_B_1, "b 1 -> b t", t=block_size)
            inf_cond = _make_inference_state(
                mode=KVCacheMode.READONLY,
                kv_caches=kv_cond,
                pattern=bp,
                block_cursor=i,
                use_pre_rope_keys=kv_quant_cfg.pre_rope_keys,
                device=init_noise.device,
            )

            v_cond = net(
                x_B_C_T_H_W=x.to(**TENSOR_KWARGS),
                timesteps_B_T=(t_cur_B_block * RECTIFIED_FLOW_T_SCALING).to(**TENSOR_KWARGS),
                **condition,
                inference_state=inf_cond,
                attn_meta=attn_meta,
            ).float()

            if use_cfg:
                inf_uncond = _make_inference_state(
                    mode=KVCacheMode.READONLY,
                    kv_caches=kv_uncond,
                    pattern=bp,
                    block_cursor=i,
                    use_pre_rope_keys=kv_quant_cfg.pre_rope_keys,
                    device=init_noise.device,
                )
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
        inf_append_cond = _make_inference_state(
            mode=KVCacheMode.APPEND,
            kv_caches=kv_cond,
            pattern=bp,
            block_cursor=i,
            use_pre_rope_keys=kv_quant_cfg.pre_rope_keys,
            device=init_noise.device,
        )
        net(x_B_C_T_H_W=x.to(**TENSOR_KWARGS), timesteps_B_T=zero_t, **condition, inference_state=inf_append_cond, attn_meta=attn_meta)
        if use_cfg:
            inf_append_uncond = _make_inference_state(
                mode=KVCacheMode.APPEND,
                kv_caches=kv_uncond,
                pattern=bp,
                block_cursor=i,
                use_pre_rope_keys=kv_quant_cfg.pre_rope_keys,
                device=init_noise.device,
            )
            net(x_B_C_T_H_W=x.to(**TENSOR_KWARGS), timesteps_B_T=zero_t, **uncondition, inference_state=inf_append_uncond, attn_meta=attn_meta)

        x_blocks.append(x)

    block_bar.close()
    del kv_cond, kv_uncond
    torch.cuda.empty_cache()
    return torch.cat(x_blocks, dim=2)


def _resolve_scale_granularity(dtype_name: str, granularity: str, is_key: bool) -> str:
    if granularity != "auto":
        return granularity
    if dtype_name == "bf16":
        return "token"
    return "channel" if is_key else "token"


def _build_tensor_spec(
    dtype_name: str,
    smoothing: str,
    group_size: int,
    scale_granularity: str,
    smooth_alpha: float,
    clip_percentile: float,
    qvg_clusters: int,
    qvg_iters: int,
    qvg_stages: int,
    qvg_share_assignments: bool,
    qvg_warm_start: bool,
    is_key: bool,
) -> TensorQuantSpec:
    return TensorQuantSpec(
        dtype_name=dtype_name,
        smoothing=smoothing,
        group_size=group_size,
        scale_granularity=_resolve_scale_granularity(dtype_name, scale_granularity, is_key=is_key),
        smooth_alpha=smooth_alpha,
        clip_percentile=clip_percentile,
        qvg_clusters=qvg_clusters,
        qvg_iters=qvg_iters,
        qvg_stages=qvg_stages,
        qvg_share_assignments=qvg_share_assignments,
        qvg_warm_start=qvg_warm_start,
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Causal inference script for Wan2.1 T2V")
    parser.add_argument("--model_size", choices=["1.3B", "14B"], default="1.3B")
    parser.add_argument("--distilled", action="store_true", help="Use few-step distilled sampling instead of diffusion ODE")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--num_steps", type=int, default=50, help="ODE steps (non-distilled) or 1-4 (distilled)")
    parser.add_argument("--sigma_max", type=float, default=1600)
    parser.add_argument("--guidance_scale", type=float, default=3.0, help="CFG scale (only used in ODE mode)")
    parser.add_argument("--timestep_shift", type=float, default=3.0, help="Timestep shift for diffusion ODE sampling")
    parser.add_argument("--mid_t", type=float, nargs="*", default=[15 / 16, 5 / 6, 5 / 8], help="Intermediate timesteps for distilled mode")
    parser.add_argument("--first_chunk_t", type=int, default=3, help="Number of frames in the first chunk")
    parser.add_argument("--chunk_t", type=int, default=3, help="Number of frames in subsequent chunks")
    parser.add_argument("--dit_path", type=str, required=True, help="Path to the DiT checkpoint")
    parser.add_argument("--vae_path", type=str, default="assets/checkpoints/Wan2.1_VAE.pth")
    parser.add_argument("--text_encoder_path", type=str, default="assets/checkpoints/models_t5_umt5-xxl-enc-bf16.pth")
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--prompt", type=str, default=_DEFAULT_PROMPT)
    parser.add_argument("--negative_prompt", type=str, default=_DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--resolution", default="480p", type=str)
    parser.add_argument("--aspect_ratio", default="16:9", type=str)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_path", type=str, default="output/causal_quant_video.mp4")

    parser.add_argument("--kv_dtype_k", choices=["bf16", "fp8", "fp4", "int8", "int4", "int2"], default="bf16")
    parser.add_argument("--kv_dtype_v", choices=["bf16", "fp8", "fp4", "int8", "int4", "int2"], default="bf16")
    parser.add_argument("--kv_smoothing_k", choices=["none", "clip", "balance", "hadamard", "reorder", "qvg"], default="none")
    parser.add_argument("--kv_smoothing_v", choices=["none", "clip", "balance", "hadamard", "reorder", "qvg"], default="none")
    parser.add_argument(
        "--kv_group_size_k",
        type=int,
        default=64,
        help="Quantization group size along the chosen key granularity axis (<=0 means the full axis).",
    )
    parser.add_argument(
        "--kv_group_size_v",
        type=int,
        default=64,
        help="Quantization group size along the chosen value granularity axis (<=0 means the full axis).",
    )
    parser.add_argument(
        "--kv_granularity_k",
        choices=["auto", "token", "channel", "head", "tensor"],
        default="auto",
        help="Scale granularity for keys. 'auto' follows common asymmetric KV practice.",
    )
    parser.add_argument(
        "--kv_granularity_v",
        choices=["auto", "token", "channel", "head", "tensor"],
        default="auto",
        help="Scale granularity for values. 'auto' follows common asymmetric KV practice.",
    )
    parser.add_argument("--kv_smooth_alpha", type=float, default=0.5, help="Strength for balance smoothing.")
    parser.add_argument("--kv_clip_percentile", type=float, default=0.999, help="Percentile used by clip smoothing.")
    parser.add_argument("--kv_qvg_clusters", type=int, default=256, help="Number of semantic centroids for qvg smoothing.")
    parser.add_argument("--kv_qvg_iters", type=int, default=4, help="K-means iterations for qvg smoothing.")
    parser.add_argument("--kv_qvg_stages", type=int, default=1, help="Number of progressive QVG smoothing stages.")
    parser.add_argument(
        "--kv_qvg_share_assignments",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether QVG uses one shared token assignment map across heads. Disable for per-head assignments.",
    )
    parser.add_argument(
        "--kv_qvg_warm_start",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse previous-chunk QVG centroids to warm-start the next chunk's clustering.",
    )
    parser.add_argument("--kv_pre_rope_keys", action="store_true", help="Use explicit pre-RoPE key caching for quantized KV experiments.")
    parser.add_argument(
        "--kv_dense_prefix_cache",
        action="store_true",
        help="Memoize additional dequantized prefixes beyond the always-on last-prefix reuse cache. Faster, but less faithful for KV-memory experiments.",
    )
    parser.add_argument(
        "--kv_triton_restore",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the experimental Triton fused restore path for supported int2/int4 token-granularity blobs.",
    )
    parser.add_argument("--warmup_iters", type=int, default=0, help="Number of warmup runs before timed run")
    parser.add_argument("--num_runs", type=int, default=1, help="Number of timed runs to average")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    key_spec = _build_tensor_spec(
        dtype_name=args.kv_dtype_k,
        smoothing=args.kv_smoothing_k,
        group_size=args.kv_group_size_k,
        scale_granularity=args.kv_granularity_k,
        smooth_alpha=args.kv_smooth_alpha,
        clip_percentile=args.kv_clip_percentile,
        qvg_clusters=args.kv_qvg_clusters,
        qvg_iters=args.kv_qvg_iters,
        qvg_stages=args.kv_qvg_stages,
        qvg_share_assignments=args.kv_qvg_share_assignments,
        qvg_warm_start=args.kv_qvg_warm_start,
        is_key=True,
    )
    value_spec = _build_tensor_spec(
        dtype_name=args.kv_dtype_v,
        smoothing=args.kv_smoothing_v,
        group_size=args.kv_group_size_v,
        scale_granularity=args.kv_granularity_v,
        smooth_alpha=args.kv_smooth_alpha,
        clip_percentile=args.kv_clip_percentile,
        qvg_clusters=args.kv_qvg_clusters,
        qvg_iters=args.kv_qvg_iters,
        qvg_stages=args.kv_qvg_stages,
        qvg_share_assignments=args.kv_qvg_share_assignments,
        qvg_warm_start=args.kv_qvg_warm_start,
        is_key=False,
    )
    use_pre_rope_keys = args.kv_pre_rope_keys or not key_spec.is_identity
    if use_pre_rope_keys and not args.kv_pre_rope_keys and not key_spec.is_identity:
        log.info("Auto-enabling pre-RoPE key caching because quantized/smoothed keys are more stable in the pre-RoPE path.")
    kv_quant_cfg = KVQuantConfig(
        key=key_spec,
        value=value_spec,
        pre_rope_keys=use_pre_rope_keys,
        dense_prefix_cache=args.kv_dense_prefix_cache,
        use_triton_restore=args.kv_triton_restore,
    )
    if args.kv_triton_restore and triton is None:
        log.warning("Triton is not available in this environment; falling back to the PyTorch KV restore path.")

    with init_weights_on_device():
        net = instantiate(DIT_CONFIGS[args.model_size]).eval()

    load_dit_weights(net, args.dit_path)
    log.success(f"Loaded DiT from {args.dit_path}")

    net.to(**TENSOR_KWARGS).cpu()
    torch.cuda.empty_cache()

    tokenizer = Wan2pt1VAEInterface(vae_pth=args.vae_path)
    w, h = VIDEO_RES_SIZE_INFO[args.resolution][args.aspect_ratio]

    log.info("Computing text embeddings...")
    text_emb = get_umt5_embedding(checkpoint_path=args.text_encoder_path, prompts=args.prompt).to(dtype=torch.bfloat16).cuda()
    condition = {"crossattn_emb": repeat(text_emb.to(**TENSOR_KWARGS), "b l d -> (k b) l d", k=args.num_samples)}

    if not args.distilled and args.guidance_scale > 1.0:
        neg_text_emb = get_umt5_embedding(checkpoint_path=args.text_encoder_path, prompts=args.negative_prompt).to(dtype=torch.bfloat16).cuda()
        uncondition = {"crossattn_emb": repeat(neg_text_emb.to(**TENSOR_KWARGS), "b l d -> (k b) l d", k=args.num_samples)}
    else:
        uncondition = None

    clear_umt5_memory()

    state_shape = [
        tokenizer.latent_ch,
        tokenizer.get_latent_num_frames(args.num_frames),
        h // tokenizer.spatial_compression_factor,
        w // tokenizer.spatial_compression_factor,
    ]

    generator = torch.Generator(device=TENSOR_KWARGS["device"])
    generator.manual_seed(args.seed)
    init_noise = torch.randn(args.num_samples, *state_shape, dtype=torch.float32, device=TENSOR_KWARGS["device"], generator=generator)

    if args.distilled:
        t_steps = build_few_step_t_steps(args.num_steps, args.sigma_max, args.mid_t, init_noise.device)
        ode = False
        guidance = 1.0
    else:
        t_steps = build_shifted_ode_t_steps(args.num_steps, args.sigma_max, args.timestep_shift, init_noise.device)
        ode = True
        guidance = args.guidance_scale

    T_latent = state_shape[1]
    log.info(f"Latent: T={T_latent}, H={state_shape[2]}, W={state_shape[3]}")
    log.info(f"Chunk pattern: first_chunk_t={args.first_chunk_t}, chunk_t={args.chunk_t}")
    log.info(f"Mode: {'distilled' if args.distilled else 'ODE'}, steps={len(t_steps)-1}, guidance={guidance}")
    log.info(
        "KV cache config: "
        f"K(dtype={key_spec.dtype_name}, smoothing={key_spec.smoothing}, group={key_spec.group_size}, granularity={key_spec.scale_granularity}) | "
        f"V(dtype={value_spec.dtype_name}, smoothing={value_spec.smoothing}, group={value_spec.group_size}, granularity={value_spec.scale_granularity}) | "
        f"qvg_stages={args.kv_qvg_stages} | qvg_share_assignments={args.kv_qvg_share_assignments} | "
        f"qvg_warm_start={args.kv_qvg_warm_start} | triton_restore={args.kv_triton_restore} | "
        f"pre_rope_keys={kv_quant_cfg.pre_rope_keys} | "
        f"last_prefix_reuse=True | extra_prefix_cache={kv_quant_cfg.dense_prefix_cache}"
    )

    net.cuda()

    def _run_sampling():
        g = torch.Generator(device=TENSOR_KWARGS["device"])
        g.manual_seed(args.seed)
        noise = torch.randn(args.num_samples, *state_shape, dtype=torch.float32, device=TENSOR_KWARGS["device"], generator=g)
        return causal_rollout_sampling(
            net,
            noise,
            t_steps,
            condition,
            uncondition,
            guidance,
            first_chunk_t=args.first_chunk_t,
            chunk_t=args.chunk_t,
            ode=ode,
            generator=g,
            kv_quant_cfg=kv_quant_cfg,
        )

    # --- Warmup ---
    if args.warmup_iters > 0:
        log.info(f"Running {args.warmup_iters} warmup iteration(s)...")
        for wi in range(args.warmup_iters):
            _ = _run_sampling()
            torch.cuda.synchronize()
            log.info(f"  Warmup {wi + 1}/{args.warmup_iters} done")
        torch.cuda.empty_cache()

    # --- Timed runs ---
    dit_times, vae_times = [], []
    for ri in range(args.num_runs):
        torch.cuda.synchronize()
        evt_dit_start = torch.cuda.Event(enable_timing=True)
        evt_dit_end = torch.cuda.Event(enable_timing=True)
        evt_vae_start = torch.cuda.Event(enable_timing=True)
        evt_vae_end = torch.cuda.Event(enable_timing=True)

        evt_dit_start.record()
        samples = _run_sampling()
        evt_dit_end.record()

        evt_vae_start.record()
        video = tokenizer.decode(samples.float())
        evt_vae_end.record()

        torch.cuda.synchronize()
        dit_ms = evt_dit_start.elapsed_time(evt_dit_end)
        vae_ms = evt_vae_start.elapsed_time(evt_vae_end)
        dit_times.append(dit_ms)
        vae_times.append(vae_ms)
        if args.num_runs > 1:
            log.info(f"  Run {ri + 1}/{args.num_runs}: DiT={dit_ms:.1f}ms  VAE={vae_ms:.1f}ms  Total={dit_ms + vae_ms:.1f}ms")

    net.cpu()
    torch.cuda.empty_cache()

    # --- Report ---
    avg_dit = sum(dit_times) / len(dit_times)
    avg_vae = sum(vae_times) / len(vae_times)
    avg_total = avg_dit + avg_vae
    fps = args.num_frames / (avg_total / 1000.0)

    run_label = f" ({args.num_runs}-run avg)" if args.num_runs > 1 else ""
    log.success(f"Profiling{run_label}:")
    log.success(f"  DiT sampling : {avg_dit:.1f} ms")
    log.success(f"  VAE decoding : {avg_vae:.1f} ms")
    log.success(f"  Total        : {avg_total:.1f} ms")
    log.success(f"  FPS          : {fps:.1f} ({args.num_frames} frames)")

    to_show = (1.0 + video.float().cpu().unsqueeze(0).clamp(-1, 1)) / 2.0
    save_image_or_video(rearrange(to_show, "n b c t h w -> c t (n h) (b w)"), args.save_path, fps=16)
    log.success(f"Saved to {args.save_path}")
