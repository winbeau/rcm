"""
Causal inference script for Wan2.1 T2V.

Supports both ODE sampling (teacher / non-distilled) and few-step sampling
(distilled) via the --distilled flag, using block-causal attention with
configurable chunk patterns.

Usage (ODE / teacher):
    python -m rcm.inference.wan2pt1_t2v_causal_infer \
        --dit_path assets/checkpoints/Wan2.1-T2V-14B.pth \
        --model_size 14B \
        --num_steps 50 --guidance_scale 5.0 \
        --first_chunk_t 1 --chunk_t 4

Usage (distilled / few-step):
    python -m rcm.inference.wan2pt1_t2v_causal_infer \
        --distilled --dit_path path/to/distilled.pth \
        --num_steps 4 --first_chunk_t 1 --chunk_t 1
"""

import argparse
import hashlib
import importlib.metadata
import os
import platform
import sys
from pathlib import Path

import numpy as np
import torch
from einops import rearrange, repeat
from PIL import Image
import torchvision.transforms.v2 as T
from tqdm import tqdm

from imaginaire.lazy_config import LazyCall as L, LazyDict, instantiate
from imaginaire.utils import log
from imaginaire.utils.io import save_image_or_video

from rcm.datasets.utils import VIDEO_RES_SIZE_INFO
from rcm.networks.wan2pt1 import WanModel
from rcm.tokenizers.wan2pt1 import Wan2pt1VAEInterface
from rcm.utils.blockmask import AttnMaskSpec, BlockPattern
from rcm.utils.kv_cache import CausalInferenceState, KVCacheMode
from rcm.utils.frame_attention import FrameAttentionObserver
from rcm.utils.attention_artifact import (
    SCHEMA_VERSION,
    git_revision,
    git_worktree_state,
    sha256_file,
    validate_run_metadata,
    write_json,
)
from rcm.utils.retained_k import RetainedKTelemetry

# Pyramid Forcing's per-head cache is incompatible with the fast_infer path,
# which writes through KVCache.write_transient into a dense contiguous buffer.
# Set False by --pyramid_kv_labels; PyramidLocalAttention raises if it is left on.
_FAST_INFER = True

# Allocator high-water mark measured over the DiT rollout only (see below).
_SAMPLING_PEAK_BYTES = None
from rcm.utils.model_utils import init_weights_on_device, load_state_dict, load_state_dict_from_dcp
from rcm.utils.umt5 import clear_umt5_memory, get_umt5_embedding

torch._dynamo.config.suppress_errors = True

_DEFAULT_NEGATIVE_PROMPT = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
# _DEFAULT_PROMPT = "A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about."
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


def parse_rf_time(value: str) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        return float(numerator) / float(denominator)
    return float(value)


def parse_rf_time_values(values) -> list[float] | None:
    if values is None:
        return None
    if isinstance(values, (str, int, float)):
        values = [values]
    parsed = []
    for value in values:
        if isinstance(value, (int, float)):
            parsed.append(float(value))
            continue
        for item in str(value).replace(",", " ").split():
            parsed.append(parse_rf_time(item))
    return parsed


def parse_mid_t_schedules(spec: str | None) -> list[list[float]]:
    if not spec:
        return []
    schedules = []
    for item in spec.split(";"):
        if not item.strip():
            schedules.append([])
            continue
        schedules.append(parse_rf_time_values(item))
    return schedules


def build_shifted_ode_t_steps(num_steps: int, sigma_max: float, timestep_shift: float, device: torch.device) -> torch.Tensor:
    sigma_max_rf = sigma_max / (sigma_max + 1)
    unshifted_sigma_max = sigma_max_rf / (timestep_shift - (timestep_shift - 1) * sigma_max_rf)
    t_steps = torch.linspace(unshifted_sigma_max, 0.0, num_steps + 1, dtype=torch.float64, device=device)
    return timestep_shift * t_steps / (1 + (timestep_shift - 1) * t_steps)


def build_few_step_t_steps(num_steps: int, sigma_max: float, mid_t: list[float], device: torch.device) -> torch.Tensor:
    mid_t = mid_t[: num_steps - 1]
    sigma_max_rf = sigma_max / (sigma_max + 1)
    return torch.tensor([sigma_max_rf, *mid_t, 0], dtype=torch.float64, device=device)


def build_few_step_schedules_per_chunk(
    step_counts: list[int],
    steps_per_chunk: list[int] | None,
    sigma_max: float,
    default_mid_t: list[float],
    mid_t_schedules: list[list[float]],
    device: torch.device,
) -> list[torch.Tensor]:
    if mid_t_schedules and not steps_per_chunk:
        raise ValueError("--mid_t_schedules requires --steps_per_chunk")
    if mid_t_schedules and len(mid_t_schedules) != len(steps_per_chunk):
        raise ValueError(
            f"--mid_t_schedules must have one schedule per --steps_per_chunk entry, got {len(mid_t_schedules)} and {len(steps_per_chunk)}"
        )

    schedules = []
    for idx, steps in enumerate(step_counts):
        if mid_t_schedules:
            schedule_idx = min(idx, len(mid_t_schedules) - 1)
            mid_t = mid_t_schedules[schedule_idx]
        else:
            mid_t = default_mid_t
        if len(mid_t) < steps - 1:
            raise ValueError(f"Need {steps - 1} mid_t values for {steps} steps, got {mid_t}")
        schedules.append(build_few_step_t_steps(steps, sigma_max, mid_t, device))
    return schedules


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


def expand_steps_per_chunk(steps_per_chunk: list[int] | None, num_blocks: int, default_steps: int) -> list[int]:
    if not steps_per_chunk:
        return [default_steps] * num_blocks
    step_counts = [int(step) for step in steps_per_chunk[:num_blocks]]
    if len(step_counts) < num_blocks:
        step_counts.extend([step_counts[-1]] * (num_blocks - len(step_counts)))
    if any(step < 1 for step in step_counts):
        raise ValueError(f"steps_per_chunk must contain positive integers, got {steps_per_chunk}")
    return step_counts


def last_step_context_t(t_steps: torch.Tensor, t_steps_per_chunk: list[torch.Tensor] | None) -> torch.Tensor:
    """Timestep used by the final denoising forward of the first generated chunk."""
    first_chunk_steps = t_steps_per_chunk[0] if t_steps_per_chunk else t_steps
    if len(first_chunk_steps) < 2:
        raise ValueError(f"Need at least start/end timesteps, got {first_chunk_steps}")
    return first_chunk_steps[-2]


def load_i2v_image_and_resolution(
    image_path: str,
    tokenizer: Wan2pt1VAEInterface,
    resolution: str,
    aspect_ratio: str,
    adaptive_resolution: bool,
) -> tuple[torch.Tensor, int, int]:
    log.info(f"Loading and preprocessing image from: {image_path}")
    input_image = Image.open(image_path).convert("RGB")
    if adaptive_resolution:
        log.info("Adaptive resolution mode enabled.")
        base_w, base_h = VIDEO_RES_SIZE_INFO[resolution][aspect_ratio]
        max_resolution_area = base_w * base_h
        log.info(f"Target area is based on {resolution} {aspect_ratio} (~{max_resolution_area} pixels).")

        orig_w, orig_h = input_image.size
        image_aspect_ratio = orig_h / orig_w

        ideal_w = np.sqrt(max_resolution_area / image_aspect_ratio)
        ideal_h = np.sqrt(max_resolution_area * image_aspect_ratio)

        stride = tokenizer.spatial_compression_factor * 2
        lat_h = round(ideal_h / stride)
        lat_w = round(ideal_w / stride)
        h = lat_h * stride
        w = lat_w * stride

        log.info(f"Input image aspect ratio: {image_aspect_ratio:.4f}. Adaptive resolution set to: {w}x{h}")
    else:
        log.info("Fixed resolution mode.")
        w, h = VIDEO_RES_SIZE_INFO[resolution][aspect_ratio]
        log.info(f"Resolution set to: {w}x{h}")

    log.info(f"Preprocessing image to {w}x{h}...")
    image_transforms = T.Compose(
        [
            T.ToImage(),
            T.Resize(size=(h, w), antialias=True),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    image_tensor = image_transforms(input_image).unsqueeze(0).unsqueeze(2).to(device=TENSOR_KWARGS["device"], dtype=torch.float32)
    return tokenizer.encode(image_tensor), w, h


def record_chunk_latency_event(profile: dict | None, label: str, start_event: torch.cuda.Event | None) -> torch.cuda.Event | None:
    if profile is None or start_event is None:
        return start_event
    end_event = torch.cuda.Event(enable_timing=True)
    end_event.record()
    profile.setdefault("chunk_latency_events", []).append((label, start_event, end_event))
    return end_event


@torch.no_grad()
def causal_rollout_sampling(
    net,
    init_noise: torch.Tensor,
    t_steps: torch.Tensor,
    t_steps_per_chunk: list[torch.Tensor] | None,
    steps_per_chunk: list[int] | None,
    condition: dict,
    uncondition: dict | None,
    guidance: float,
    first_chunk_t: int,
    chunk_t: int,
    ode: bool,
    generator: torch.Generator | None = None,
    context_from_last_step: bool = False,
    context_from_last_step_start_chunk: int = 0,
    profile: dict | None = None,
    attn_observer: object | None = None,
    attn_capture_step: str = "append",
    retained_k_observer: object | None = None,
) -> torch.Tensor:
    if context_from_last_step_start_chunk < 0:
        raise ValueError(f"context_from_last_step_start_chunk must be non-negative, got {context_from_last_step_start_chunk}")
    B, C, T, H, W = init_noise.shape
    frame_tokens, num_blocks, bp = make_block_pattern(T, H, W, first_chunk_t, chunk_t, net.get_spatial_patch_size())
    ones_B_1 = torch.ones(B, 1, device=init_noise.device, dtype=torch.float64)
    use_cfg = uncondition is not None and guidance > 1.0

    kv_cond = net.allocate_kv_caches(max_len=T * frame_tokens)
    kv_uncond = net.allocate_kv_caches(max_len=T * frame_tokens) if use_cfg else None

    if not ode:
        default_steps = len(t_steps) - 1
        step_counts = expand_steps_per_chunk(steps_per_chunk, num_blocks, default_steps)
        max_steps = max(step_counts)
        step_noises = [torch.randn(init_noise.shape, dtype=torch.float32, device=init_noise.device, generator=generator) for _ in range(max_steps)]
    else:
        step_counts = [len(t_steps) - 1] * num_blocks
        step_noises = None

    x_blocks = []
    block_bar = tqdm(range(num_blocks), desc="Chunks", position=0)
    chunk_latency_start = None
    if profile is not None:
        profile["chunk_latency_events"] = []
        chunk_latency_start = torch.cuda.Event(enable_timing=True)
        chunk_latency_start.record()

    for i in block_bar:
        frame_start, frame_end, block_size = block_span(bp, i)
        attn_meta = AttnMaskSpec(mode="block_causal", pattern=bp, q_block_offset=i)
        t_steps_block = t_steps_per_chunk[i] if t_steps_per_chunk else t_steps
        num_steps = len(t_steps_block) - 1
        use_last_step_context = context_from_last_step and i >= context_from_last_step_start_chunk

        x = init_noise[:, :, frame_start:frame_end].to(torch.float64) * t_steps_block[0]

        step_bar = tqdm(zip(t_steps_block[:-1], t_steps_block[1:]), total=num_steps, desc=f"  Chunk {i}/{num_blocks}", position=1, leave=False)
        for step_idx, (t_cur, t_next) in enumerate(step_bar):
            t_cur_B_1 = (t_cur * ones_B_1 * RECTIFIED_FLOW_T_SCALING).to(**TENSOR_KWARGS)
            is_last = step_idx == num_steps - 1
            kv_mode = KVCacheMode.APPEND if (use_last_step_context and is_last) else KVCacheMode.READONLY
            # "denoise0" reproduces what AdaHead's extractor actually recorded: its
            # aggregator keys on the key length growing, and every step of a chunk
            # shares one key length, so only step 0 -- the noisiest query -- landed.
            observe_here = attn_observer if (attn_capture_step == "denoise0" and step_idx == 0) else None
            inf_cond = CausalInferenceState(
                mode=kv_mode,
                kv_caches=kv_cond,
                pattern=bp,
                block_cursor=i,
                fast_infer=_FAST_INFER,
                attn_observer=observe_here,
                retained_k_observer=retained_k_observer,
                pass_name="denoise0" if step_idx == 0 else "denoise",
                denoise_step=step_idx,
                stream_name="cond",
            )

            v_cond = net(
                x_B_C_T_H_W=x.to(**TENSOR_KWARGS),
                timesteps_B_T=t_cur_B_1,
                **condition,
                inference_state=inf_cond,
                attn_meta=attn_meta,
            ).float()

            if use_cfg:
                inf_uncond = CausalInferenceState(
                    mode=kv_mode,
                    kv_caches=kv_uncond,
                    pattern=bp,
                    block_cursor=i,
                    fast_infer=_FAST_INFER,
                    retained_k_observer=retained_k_observer,
                    pass_name="denoise0" if step_idx == 0 else "denoise",
                    denoise_step=step_idx,
                    stream_name="uncond",
                )
                v_uncond = net(
                    x_B_C_T_H_W=x.to(**TENSOR_KWARGS),
                    timesteps_B_T=t_cur_B_1,
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

        if not use_last_step_context:
            zero_t = torch.zeros(B, 1, **TENSOR_KWARGS)
            # The observer is attached only to this pass: it runs once per chunk
            # at t=0 on the denoised latent, so it sees clean context and yields
            # exactly one observation per chunk (the denoising forwards above
            # would fire once per step on progressively noisier inputs).
            inf_append_cond = CausalInferenceState(
                mode=KVCacheMode.APPEND,
                kv_caches=kv_cond,
                pattern=bp,
                block_cursor=i,
                fast_infer=_FAST_INFER,
                attn_observer=attn_observer if attn_capture_step == "append" else None,
                retained_k_observer=retained_k_observer,
                pass_name="append",
                denoise_step=-1,
                stream_name="cond",
            )
            net(x_B_C_T_H_W=x.to(**TENSOR_KWARGS), timesteps_B_T=zero_t, **condition, inference_state=inf_append_cond, attn_meta=attn_meta)
            if use_cfg:
                inf_append_uncond = CausalInferenceState(
                    mode=KVCacheMode.APPEND,
                    kv_caches=kv_uncond,
                    pattern=bp,
                    block_cursor=i,
                    fast_infer=_FAST_INFER,
                    retained_k_observer=retained_k_observer,
                    pass_name="append",
                    denoise_step=-1,
                    stream_name="uncond",
                )
                net(x_B_C_T_H_W=x.to(**TENSOR_KWARGS), timesteps_B_T=zero_t, **uncondition, inference_state=inf_append_uncond, attn_meta=attn_meta)

        x_blocks.append(x)
        generated_chunks = len(x_blocks)
        if generated_chunks == 1:
            chunk_latency_start = record_chunk_latency_event(profile, "first_chunk", chunk_latency_start)
        elif generated_chunks == 2:
            chunk_latency_start = record_chunk_latency_event(profile, "second_chunk", chunk_latency_start)

    block_bar.close()
    del kv_cond, kv_uncond
    torch.cuda.empty_cache()
    return torch.cat(x_blocks, dim=2)


@torch.no_grad()
def causal_i2v_rollout_sampling(
    net,
    image_latent: torch.Tensor,
    init_noise: torch.Tensor,
    t_steps: torch.Tensor,
    t_steps_per_chunk: list[torch.Tensor] | None,
    steps_per_chunk: list[int] | None,
    condition: dict,
    uncondition: dict | None,
    guidance: float,
    chunk_t: int,
    ode: bool,
    generator: torch.Generator | None = None,
    context_from_last_step: bool = False,
    context_from_last_step_start_chunk: int = 0,
    profile: dict | None = None,
    retained_k_observer: object | None = None,
) -> torch.Tensor:
    if context_from_last_step_start_chunk < 0:
        raise ValueError(f"context_from_last_step_start_chunk must be non-negative, got {context_from_last_step_start_chunk}")
    B, C, _, H, W = image_latent.shape
    T_remaining = init_noise.shape[2]
    if T_remaining % chunk_t != 0:
        raise ValueError(f"T_remaining={T_remaining} must be divisible by chunk_t={chunk_t}")

    first_chunk_t = 1
    T_total = 1 + T_remaining
    frame_tokens, num_blocks, bp = make_block_pattern(T_total, H, W, first_chunk_t, chunk_t, net.get_spatial_patch_size())
    ones_B_1 = torch.ones(B, 1, device=image_latent.device, dtype=torch.float64)
    use_cfg = uncondition is not None and guidance > 1.0

    kv_cond = net.allocate_kv_caches(max_len=T_total * frame_tokens)
    kv_uncond = net.allocate_kv_caches(max_len=T_total * frame_tokens) if use_cfg else None

    if not ode:
        default_steps = len(t_steps) - 1
        step_counts = expand_steps_per_chunk(steps_per_chunk, num_blocks - 1, default_steps)
        max_steps = max(step_counts)
        noise_full = torch.cat([image_latent.new_zeros(B, C, 1, H, W), init_noise], dim=2)
        step_noises = [torch.randn(noise_full.shape, dtype=torch.float32, device=noise_full.device, generator=generator) for _ in range(max_steps)]
    else:
        step_counts = [len(t_steps) - 1] * (num_blocks - 1)
        step_noises = None

    chunk_latency_start = None
    if profile is not None:
        profile["chunk_latency_events"] = []
        chunk_latency_start = torch.cuda.Event(enable_timing=True)
        chunk_latency_start.record()

    attn_meta_0 = AttnMaskSpec(mode="block_causal", pattern=bp, q_block_offset=0)
    prefill_t = torch.zeros((), dtype=torch.float64, device=image_latent.device)
    image_context = image_latent.to(torch.float64)
    if context_from_last_step and context_from_last_step_start_chunk == 0:
        prefill_t = last_step_context_t(t_steps, t_steps_per_chunk).to(device=image_latent.device, dtype=torch.float64)
        image_noise = torch.randn(image_latent.shape, dtype=torch.float32, device=image_latent.device, generator=generator)
        image_context = (1 - prefill_t) * image_context + prefill_t * image_noise.to(torch.float64)
    prefill_t_B_1 = (prefill_t * ones_B_1 * RECTIFIED_FLOW_T_SCALING).to(**TENSOR_KWARGS)

    inf_cond = CausalInferenceState(
        mode=KVCacheMode.APPEND,
        kv_caches=kv_cond,
        pattern=bp,
        block_cursor=0,
        fast_infer=_FAST_INFER,
        retained_k_observer=retained_k_observer,
        pass_name="append",
        denoise_step=-1,
        stream_name="cond",
    )
    net(x_B_C_T_H_W=image_context.to(**TENSOR_KWARGS), timesteps_B_T=prefill_t_B_1, **condition, inference_state=inf_cond, attn_meta=attn_meta_0)
    if use_cfg:
        inf_uncond = CausalInferenceState(
            mode=KVCacheMode.APPEND,
            kv_caches=kv_uncond,
            pattern=bp,
            block_cursor=0,
            fast_infer=_FAST_INFER,
            retained_k_observer=retained_k_observer,
            pass_name="append",
            denoise_step=-1,
            stream_name="uncond",
        )
        net(
            x_B_C_T_H_W=image_context.to(**TENSOR_KWARGS),
            timesteps_B_T=prefill_t_B_1,
            **uncondition,
            inference_state=inf_uncond,
            attn_meta=attn_meta_0,
        )

    x_blocks = [image_latent.to(torch.float64)]
    block_bar = tqdm(range(1, num_blocks), desc="I2V chunks", position=0)
    for i in block_bar:
        frame_start, frame_end, block_size = block_span(bp, i)
        noise_start = frame_start - 1
        attn_meta = AttnMaskSpec(mode="block_causal", pattern=bp, q_block_offset=i)
        t_steps_block = t_steps_per_chunk[i - 1] if t_steps_per_chunk else t_steps
        num_steps = len(t_steps_block) - 1
        use_last_step_context = context_from_last_step and i >= context_from_last_step_start_chunk

        x = init_noise[:, :, noise_start : noise_start + block_size].to(torch.float64) * t_steps_block[0]

        step_bar = tqdm(zip(t_steps_block[:-1], t_steps_block[1:]), total=num_steps, desc=f"  Chunk {i}/{num_blocks}", position=1, leave=False)
        for step_idx, (t_cur, t_next) in enumerate(step_bar):
            t_cur_B_1 = (t_cur * ones_B_1 * RECTIFIED_FLOW_T_SCALING).to(**TENSOR_KWARGS)
            is_last = step_idx == num_steps - 1
            kv_mode = KVCacheMode.APPEND if (use_last_step_context and is_last) else KVCacheMode.READONLY
            inf_cond = CausalInferenceState(
                mode=kv_mode,
                kv_caches=kv_cond,
                pattern=bp,
                block_cursor=i,
                fast_infer=_FAST_INFER,
                retained_k_observer=retained_k_observer,
                pass_name="denoise0" if step_idx == 0 else "denoise",
                denoise_step=step_idx,
                stream_name="cond",
            )

            v_cond = net(
                x_B_C_T_H_W=x.to(**TENSOR_KWARGS),
                timesteps_B_T=t_cur_B_1,
                **condition,
                inference_state=inf_cond,
                attn_meta=attn_meta,
            ).float()

            if use_cfg:
                inf_uncond = CausalInferenceState(
                    mode=kv_mode,
                    kv_caches=kv_uncond,
                    pattern=bp,
                    block_cursor=i,
                    fast_infer=_FAST_INFER,
                    retained_k_observer=retained_k_observer,
                    pass_name="denoise0" if step_idx == 0 else "denoise",
                    denoise_step=step_idx,
                    stream_name="uncond",
                )
                v_uncond = net(
                    x_B_C_T_H_W=x.to(**TENSOR_KWARGS),
                    timesteps_B_T=t_cur_B_1,
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

        if not use_last_step_context:
            zero_t = torch.zeros(B, 1, **TENSOR_KWARGS)
            inf_append_cond = CausalInferenceState(
                mode=KVCacheMode.APPEND,
                kv_caches=kv_cond,
                pattern=bp,
                block_cursor=i,
                fast_infer=_FAST_INFER,
                retained_k_observer=retained_k_observer,
                pass_name="append",
                denoise_step=-1,
                stream_name="cond",
            )
            net(x_B_C_T_H_W=x.to(**TENSOR_KWARGS), timesteps_B_T=zero_t, **condition, inference_state=inf_append_cond, attn_meta=attn_meta)
            if use_cfg:
                inf_append_uncond = CausalInferenceState(
                    mode=KVCacheMode.APPEND,
                    kv_caches=kv_uncond,
                    pattern=bp,
                    block_cursor=i,
                    fast_infer=_FAST_INFER,
                    retained_k_observer=retained_k_observer,
                    pass_name="append",
                    denoise_step=-1,
                    stream_name="uncond",
                )
                net(x_B_C_T_H_W=x.to(**TENSOR_KWARGS), timesteps_B_T=zero_t, **uncondition, inference_state=inf_append_uncond, attn_meta=attn_meta)

        x_blocks.append(x)
        generated_chunks = len(x_blocks) - 1
        if generated_chunks == 1:
            chunk_latency_start = record_chunk_latency_event(profile, "first_chunk", chunk_latency_start)
        elif generated_chunks == 2:
            chunk_latency_start = record_chunk_latency_event(profile, "second_chunk", chunk_latency_start)

    block_bar.close()
    del kv_cond, kv_uncond
    torch.cuda.empty_cache()
    return torch.cat(x_blocks, dim=2)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Causal inference script for Wan2.1 T2V")
    parser.add_argument("--model_size", choices=["1.3B", "14B"], default="1.3B")
    parser.add_argument("--distilled", action="store_true", help="Use few-step distilled sampling instead of diffusion ODE")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--num_steps", type=int, default=50, help="ODE steps (non-distilled) or 1-4 (distilled)")
    parser.add_argument("--sigma_max", type=float, default=1600)
    parser.add_argument("--guidance_scale", type=float, default=3.0, help="CFG scale (only used in ODE mode)")
    parser.add_argument("--timestep_shift", type=float, default=3.0, help="Timestep shift for diffusion ODE sampling")
    parser.add_argument("--mid_t", type=str, nargs="*", default=[15 / 16, 5 / 6, 5 / 8], help="Intermediate timesteps for distilled mode")
    parser.add_argument(
        "--steps_per_chunk", type=int, nargs="*", default=None, help="Distilled causal steps per chunk; repeats the last value, e.g. 4 2"
    )
    parser.add_argument(
        "--mid_t_schedules",
        type=str,
        default="",
        help="Per-entry midpoint schedules aligned with --steps_per_chunk, e.g. '0.9,0.75,0.5;5/6'",
    )
    parser.add_argument("--first_chunk_t", type=int, default=1, help="Number of frames in the first chunk")
    parser.add_argument("--chunk_t", type=int, default=1, help="Number of frames in subsequent chunks")
    parser.add_argument("--dit_path", type=str, required=True, help="Path to the DiT checkpoint")
    parser.add_argument(
        "--dit_sha256",
        type=str,
        default=os.environ.get("RCM_DIT_SHA256", ""),
        help="Precomputed checkpoint SHA-256 for capture metadata; avoids rehashing a large checkpoint per run",
    )
    parser.add_argument("--vae_path", type=str, default="assets/checkpoints/Wan2.1_VAE.pth")
    parser.add_argument("--text_encoder_path", type=str, default="assets/checkpoints/models_t5_umt5-xxl-enc-bf16.pth")
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--prompt", type=str, default=_DEFAULT_PROMPT)
    parser.add_argument("--negative_prompt", type=str, default=_DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--image_path", type=str, default="", help="Optional conditioning image path for I2V; output still has --num_frames frames")
    parser.add_argument("--resolution", default="480p", type=str)
    parser.add_argument("--aspect_ratio", default="16:9", type=str)
    parser.add_argument(
        "--adaptive_resolution",
        action="store_true",
        help="For I2V, adapt output resolution to the input image aspect ratio while preserving the target pixel area.",
    )
    parser.add_argument(
        "--pyramid_kv_labels",
        type=str,
        default="",
        help="Path to a 30x12 Anchor/Wave/Veil label CSV. Enables the head-aware "
        "KV cache (Pyramid Forcing) and turns off fast_infer, which is "
        "incompatible with a per-head ragged cache.",
    )
    parser.add_argument(
        "--pyramid_retained_k_output",
        type=str,
        default="",
        help="Optional JSONL path for actual per-head retained-K telemetry during PyramidKV inference",
    )
    parser.add_argument("--pyramid_recent_frames", type=int, default=4)
    parser.add_argument("--pyramid_osc_sink_frames", type=int, default=1)
    parser.add_argument("--pyramid_stable_sink_frames", type=int, default=3)
    parser.add_argument("--pyramid_cyclic_period", type=int, default=6)
    parser.add_argument("--pyramid_stride_interval", type=int, default=6)
    parser.add_argument(
        "--pyramid_sink_time_clamp",
        type=int,
        nargs=2,
        default=None,
        metavar=("MIN", "MAX"),
        help="Dynamic-RoPE window for re-timed anchors. The paper's 18/21 is "
        "pinned to Self-Forcing's 21-frame training range; rCM's "
        "VideoRopePosition3DEmb uses len_t=32, so the default here is derived "
        "from that instead of copied. Omit to disable clamping.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_path", type=str, default="output/causal_generated_video.mp4")
    parser.add_argument("--context_from_last_step", action="store_true", help="Cache KVs from last denoising step instead of extra t=0 forward pass")
    parser.add_argument("--context_from_last_step_start_chunk", type=int, default=0, help="First causal chunk index to use --context_from_last_step")
    parser.add_argument("--warmup_iters", type=int, default=0, help="Number of warmup runs before timed run")
    parser.add_argument("--num_runs", type=int, default=1, help="Number of timed runs to average")
    # --- Frame-level attention extraction ---
    parser.add_argument(
        "--extract_attn_layers",
        type=str,
        default="",
        help="Comma-separated DiT layer indices to capture frame-level attention for, or 'all'. Empty disables extraction.",
    )
    parser.add_argument("--attn_output_dir", type=str, default="cache/attn", help="Directory for the per-layer attention .pt artifacts")
    parser.add_argument(
        "--attn_method",
        choices=["pooled", "naive"],
        default="pooled",
        help="pooled = mean-pool each frame's tokens then contract (exact and ~1e6x cheaper); naive = materialize token-level scores",
    )
    parser.add_argument("--attn_verify", action="store_true", help="Check the pooled fast path against the naive reference on one observation")
    parser.add_argument(
        "--attn_capture_step",
        choices=["append", "denoise0"],
        default="append",
        help=(
            "Which forward to observe. 'append' = the once-per-chunk KV-append pass at t=0 (clean query, clean key). "
            "'denoise0' = the first denoising step (noisy query, clean key), which is what AdaHead's extractor "
            "actually recorded -- use it when comparing logit magnitudes against the Self-Forcing artifacts."
        ),
    )
    parser.add_argument("--attn_tag", type=str, default="", help="Optional suffix for the artifact filenames")
    parser.add_argument(
        "--attn_layout",
        choices=["flat", "runs"],
        default="flat",
        help=(
            "Artifact layout. 'flat' writes <dir>/layer<N><tag>.pt. 'runs' writes <dir>/run_<NNN>/layer<N>.pt, "
            "which is what jupyter-plot's spectral_analysis_utils.compute_head_period_homology_256 globs -- it "
            "requires that exact naming and ignores --attn_tag."
        ),
    )
    parser.add_argument(
        "--attn_run_index",
        type=int,
        default=0,
        help="Run index for --attn_layout runs, i.e. the NNN in run_<NNN>. Use it to shard a prompt set across GPUs.",
    )

    return parser.parse_args()


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


if __name__ == "__main__":
    args = parse_arguments()

    if args.extract_attn_layers and args.pyramid_kv_labels:
        raise ValueError("--extract_attn_layers requires dense full-horizon attention; do not combine it with --pyramid_kv_labels")
    if args.attn_verify and not args.extract_attn_layers:
        raise ValueError("--attn_verify requires --extract_attn_layers")
    if args.extract_attn_layers and args.num_samples != 1:
        raise ValueError("attention extraction requires --num_samples 1 so samples are never averaged")
    if args.extract_attn_layers and (args.warmup_iters != 0 or args.num_runs != 1):
        raise ValueError("attention extraction requires --warmup_iters 0 --num_runs 1")
    if args.pyramid_retained_k_output and not args.pyramid_kv_labels:
        raise ValueError("--pyramid_retained_k_output requires --pyramid_kv_labels")
    if args.pyramid_retained_k_output and (
        args.warmup_iters != 0 or args.num_runs != 1
    ):
        raise ValueError(
            "retained-K telemetry is per run; use --warmup_iters 0 --num_runs 1"
        )

    attention_checkpoint_sha256 = None
    if args.extract_attn_layers:
        if args.dit_sha256:
            attention_checkpoint_sha256 = args.dit_sha256.strip().lower()
        elif os.path.isfile(args.dit_path):
            log.info("Hashing DiT checkpoint for attention artifact provenance...")
            attention_checkpoint_sha256 = sha256_file(args.dit_path)
        else:
            raise ValueError(
                "attention extraction from a checkpoint directory requires "
                "--dit_sha256 or RCM_DIT_SHA256"
            )
        if len(attention_checkpoint_sha256) != 64 or any(
            char not in "0123456789abcdef"
            for char in attention_checkpoint_sha256
        ):
            raise ValueError("--dit_sha256 must be a 64-character hexadecimal digest")

    # --- Load model ---
    with init_weights_on_device():
        net = instantiate(DIT_CONFIGS[args.model_size]).eval()

    load_dit_weights(net, args.dit_path)
    log.success(f"Loaded DiT from {args.dit_path}")

    if args.pyramid_kv_labels:
        from rcm.utils.pyramid_attention import PyramidSpec

        _FAST_INFER = False
        _w, _h = VIDEO_RES_SIZE_INFO[args.resolution][args.aspect_ratio]
        _patch = net.patch_size  # (1, 2, 2)
        latent_h = _h // 8 // _patch[1]
        latent_w = _w // 8 // _patch[2]
        clamp_min, clamp_max = (args.pyramid_sink_time_clamp or (0, 0))
        pyramid_spec = PyramidSpec(
            num_layers=len(net.blocks),
            num_heads=net.num_heads,
            head_dim=net.dim // net.num_heads,
            latent_h=latent_h,
            latent_w=latent_w,
            labels_csv=args.pyramid_kv_labels,
            # Only sizes the RoPE frequency table ([max_pos, 64] complex64, so
            # 256 rows is ~128 KB); overshooting costs nothing.
            max_frames=max(256, args.num_frames),
            recent_frames=args.pyramid_recent_frames,
            osc_sink_frames=args.pyramid_osc_sink_frames,
            stable_sink_frames=args.pyramid_stable_sink_frames,
            cyclic_period=args.pyramid_cyclic_period,
            stride_interval=args.pyramid_stride_interval,
            # Clamping is OFF by default. The paper's 18/21 encodes Self-Forcing's
            # 21-latent-frame training range; rCM natively rolls out 72+ frames, so
            # copying those numbers would re-time anchors into a window that has no
            # meaning here. Whether rCM benefits from any clamp is an open question
            # for G3, not something to guess at now.
            sink_time_clamp_min=clamp_min,
            sink_time_clamp_max=clamp_max,
            sink_grid_decoupling=bool(args.pyramid_sink_time_clamp),
        )
        net.enable_pyramid_kv(pyramid_spec)
        log.success(
            f"Pyramid Forcing KV cache enabled from {args.pyramid_kv_labels}; "
            f"grid {latent_h}x{latent_w}={latent_h * latent_w} tokens/frame, "
            f"policies {pyramid_spec.composition_summary()}, fast_infer off"
        )

    net.to(**TENSOR_KWARGS).cpu()
    torch.cuda.empty_cache()

    # --- Tokenizer ---
    tokenizer = Wan2pt1VAEInterface(vae_pth=args.vae_path)

    # --- Resolution / optional I2V image ---
    is_i2v = bool(args.image_path)
    if is_i2v:
        image_latent, w, h = load_i2v_image_and_resolution(
            args.image_path,
            tokenizer,
            args.resolution,
            args.aspect_ratio,
            args.adaptive_resolution,
        )
    else:
        w, h = VIDEO_RES_SIZE_INFO[args.resolution][args.aspect_ratio]
        image_latent = None

    # --- Text embeddings ---
    log.info(f"Computing text embeddings...")
    text_emb = get_umt5_embedding(checkpoint_path=args.text_encoder_path, prompts=args.prompt).to(dtype=torch.bfloat16).cuda()

    condition = {"crossattn_emb": repeat(text_emb.to(**TENSOR_KWARGS), "b l d -> (k b) l d", k=args.num_samples)}

    if not args.distilled and args.guidance_scale > 1.0:
        neg_text_emb = get_umt5_embedding(checkpoint_path=args.text_encoder_path, prompts=args.negative_prompt).to(dtype=torch.bfloat16).cuda()
        uncondition = {"crossattn_emb": repeat(neg_text_emb.to(**TENSOR_KWARGS), "b l d -> (k b) l d", k=args.num_samples)}
    else:
        uncondition = None

    clear_umt5_memory()

    # --- Build timesteps ---
    latent_frames = tokenizer.get_latent_num_frames(args.num_frames)
    latent_h = h // tokenizer.spatial_compression_factor
    latent_w = w // tokenizer.spatial_compression_factor
    state_shape = [tokenizer.latent_ch, latent_frames, latent_h, latent_w]
    noise_shape = state_shape
    if is_i2v:
        if latent_frames <= 1:
            raise ValueError(f"I2V requires more than one latent frame, got num_frames={args.num_frames}")
        if (latent_frames - 1) % args.chunk_t != 0:
            raise ValueError(f"I2V latent remaining frames ({latent_frames - 1}) must be divisible by chunk_t={args.chunk_t}")
        image_latent = repeat(image_latent.to(**TENSOR_KWARGS), "1 c t h w -> b c t h w", b=args.num_samples)
        noise_shape = [tokenizer.latent_ch, latent_frames - 1, latent_h, latent_w]

    generator = torch.Generator(device=TENSOR_KWARGS["device"])
    generator.manual_seed(args.seed)

    if args.distilled:
        mid_t = parse_rf_time_values(args.mid_t)
        t_steps = build_few_step_t_steps(args.num_steps, args.sigma_max, mid_t, TENSOR_KWARGS["device"])
        mid_t_schedules = parse_mid_t_schedules(args.mid_t_schedules)
        if is_i2v:
            num_schedule_blocks = (latent_frames - 1) // args.chunk_t
        else:
            _, num_schedule_blocks, _ = make_block_pattern(
                latent_frames, latent_h, latent_w, args.first_chunk_t, args.chunk_t, net.get_spatial_patch_size()
            )
        step_counts = expand_steps_per_chunk(args.steps_per_chunk, num_schedule_blocks, args.num_steps)
        t_steps_per_chunk = build_few_step_schedules_per_chunk(
            step_counts, args.steps_per_chunk, args.sigma_max, mid_t, mid_t_schedules, TENSOR_KWARGS["device"]
        )
        ode = False
        guidance = 1.0
    else:
        if args.steps_per_chunk:
            raise ValueError("--steps_per_chunk is only supported with --distilled")
        t_steps = build_shifted_ode_t_steps(args.num_steps, args.sigma_max, args.timestep_shift, TENSOR_KWARGS["device"])
        t_steps_per_chunk = None
        step_counts = None
        ode = True
        guidance = args.guidance_scale

    T_latent = latent_frames
    log.info(f"Latent: T={T_latent}, H={state_shape[2]}, W={state_shape[3]}")
    if is_i2v:
        log.info(f"I2V image: {args.image_path}")
        log.info(f"Chunk pattern: first_chunk_t=1 (image), chunk_t={args.chunk_t}")
    else:
        log.info(f"Chunk pattern: first_chunk_t={args.first_chunk_t}, chunk_t={args.chunk_t}")
    log.info(f"Mode: {'distilled' if args.distilled else 'ODE'}, steps={len(t_steps)-1}, guidance={guidance}")
    if step_counts is not None and args.steps_per_chunk:
        log.info(f"Per-{'generated-' if is_i2v else ''}chunk distilled steps: {step_counts}")

    # --- Sample ---
    net.cuda()

    retained_k_observer = RetainedKTelemetry() if args.pyramid_retained_k_output else None

    # --- Frame-level attention observer ---
    attn_observer = None
    if args.extract_attn_layers:
        if is_i2v:
            raise ValueError("--extract_attn_layers is only wired for the T2V causal path")
        if args.warmup_iters != 0 or args.num_runs != 1:
            raise ValueError("attention extraction needs exactly one sampling pass: use --warmup_iters 0 --num_runs 1")
        if args.attn_capture_step == "append" and args.context_from_last_step:
            raise ValueError(
                "--context_from_last_step replaces the KV-append pass, so --attn_capture_step append would never "
                "fire and would silently write an all-zero matrix. Use --attn_capture_step denoise0 instead."
            )
        num_layers = len(net.blocks)
        if args.extract_attn_layers.strip() == "all":
            layer_indices = list(range(num_layers))
        else:
            layer_indices = [int(v) for v in args.extract_attn_layers.split(",") if v.strip() != ""]
        for layer in layer_indices:
            if not 0 <= layer < num_layers:
                raise ValueError(f"layer index {layer} out of range for a {num_layers}-layer model")
        frame_tokens_ext = state_shape[2] * state_shape[3] // net.get_spatial_patch_size()
        attn_observer = FrameAttentionObserver(
            layer_indices=layer_indices,
            frame_tokens=frame_tokens_ext,
            num_heads=net.num_heads,
            num_frames=T_latent,
            first_chunk_frames=args.first_chunk_t,
            chunk_frames=args.chunk_t,
            method=args.attn_method,
            verify_once=args.attn_verify,
        )
        log.info(
            f"Attention extraction: layers={layer_indices} heads={net.num_heads} "
            f"frames={T_latent} frame_tokens={frame_tokens_ext} method={args.attn_method}"
        )
        if args.attn_layout == "runs":
            attn_dir = Path(args.attn_output_dir) / f"run_{args.attn_run_index:03d}"
        else:
            attn_dir = Path(args.attn_output_dir)
        attn_dir.mkdir(parents=True, exist_ok=True)
        tag = "" if args.attn_layout == "runs" else (f"_{args.attn_tag}" if args.attn_tag else "")
        output_paths = {
            layer_index: attn_dir / f"layer{layer_index}{tag}.pt"
            for layer_index in sorted(layer_indices)
        }
        manifest_path = attn_dir / "run.json"
        existing = [path for path in [*output_paths.values(), manifest_path] if path.exists()]
        if existing:
            raise FileExistsError(
                "refusing to overwrite existing attention artifacts: "
                + ", ".join(str(path) for path in existing)
            )

    def _run_sampling(profile: dict | None = None):
        g = torch.Generator(device=TENSOR_KWARGS["device"])
        g.manual_seed(args.seed)
        noise = torch.randn(args.num_samples, *noise_shape, dtype=torch.float32, device=TENSOR_KWARGS["device"], generator=g)
        if is_i2v:
            return causal_i2v_rollout_sampling(
                net,
                image_latent,
                noise,
                t_steps,
                t_steps_per_chunk,
                args.steps_per_chunk,
                condition,
                uncondition,
                guidance,
                chunk_t=args.chunk_t,
                ode=ode,
                generator=g,
                context_from_last_step=args.context_from_last_step,
                context_from_last_step_start_chunk=args.context_from_last_step_start_chunk,
                profile=profile,
                retained_k_observer=retained_k_observer,
            )
        return causal_rollout_sampling(
            net,
            noise,
            t_steps,
            t_steps_per_chunk,
            args.steps_per_chunk,
            condition,
            uncondition,
            guidance,
            first_chunk_t=args.first_chunk_t,
            chunk_t=args.chunk_t,
            ode=ode,
            generator=g,
            context_from_last_step=args.context_from_last_step,
            context_from_last_step_start_chunk=args.context_from_last_step_start_chunk,
            profile=profile,
            attn_observer=attn_observer,
            attn_capture_step=args.attn_capture_step,
            retained_k_observer=retained_k_observer,
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
    first_chunk_times, second_chunk_times = [], []
    for ri in range(args.num_runs):
        torch.cuda.synchronize()
        evt_dit_start = torch.cuda.Event(enable_timing=True)
        evt_dit_end = torch.cuda.Event(enable_timing=True)
        evt_vae_start = torch.cuda.Event(enable_timing=True)
        evt_vae_end = torch.cuda.Event(enable_timing=True)

        # Reset just before the rollout so the umT5 load (freed above, but still
        # the process high-water mark) does not mask a KV cache smaller than it.
        torch.cuda.reset_peak_memory_stats()
        evt_dit_start.record()
        profile = {}
        samples = _run_sampling(profile=profile)
        evt_dit_end.record()
        _SAMPLING_PEAK_BYTES = max(_SAMPLING_PEAK_BYTES or 0, torch.cuda.max_memory_allocated())

        evt_vae_start.record()
        video = tokenizer.decode(samples.float())
        evt_vae_end.record()

        torch.cuda.synchronize()
        dit_ms = evt_dit_start.elapsed_time(evt_dit_end)
        vae_ms = evt_vae_start.elapsed_time(evt_vae_end)
        chunk_times = {label: start.elapsed_time(end) for label, start, end in profile.get("chunk_latency_events", [])}
        dit_times.append(dit_ms)
        vae_times.append(vae_ms)
        if "first_chunk" in chunk_times:
            first_chunk_times.append(chunk_times["first_chunk"])
        if "second_chunk" in chunk_times:
            second_chunk_times.append(chunk_times["second_chunk"])
        if args.num_runs > 1:
            first_msg = f"  FirstChunk={chunk_times['first_chunk']:.1f}ms" if "first_chunk" in chunk_times else ""
            second_msg = f"  SecondChunk={chunk_times['second_chunk']:.1f}ms" if "second_chunk" in chunk_times else ""
            log.info(f"  Run {ri + 1}/{args.num_runs}: DiT={dit_ms:.1f}ms  VAE={vae_ms:.1f}ms  Total={dit_ms + vae_ms:.1f}ms{first_msg}{second_msg}")

    net.cpu()
    torch.cuda.empty_cache()

    # --- Report ---
    avg_dit = sum(dit_times) / len(dit_times)
    avg_vae = sum(vae_times) / len(vae_times)
    avg_first_chunk = sum(first_chunk_times) / len(first_chunk_times) if first_chunk_times else None
    avg_second_chunk = sum(second_chunk_times) / len(second_chunk_times) if second_chunk_times else None
    avg_total = avg_dit + avg_vae
    fps = args.num_frames / (avg_total / 1000.0)

    run_label = f" ({args.num_runs}-run avg)" if args.num_runs > 1 else ""
    log.success(f"Profiling{run_label}:")
    log.success(f"  DiT sampling : {avg_dit:.1f} ms")
    if avg_first_chunk is not None:
        log.success(f"  First chunk / first-frame DiT latency : {avg_first_chunk:.1f} ms")
    if avg_second_chunk is not None:
        log.success(f"  Second chunk latency              : {avg_second_chunk:.1f} ms")
    log.success(f"  VAE decoding : {avg_vae:.1f} ms")
    log.success(f"  Total        : {avg_total:.1f} ms")
    log.success(f"  FPS          : {fps:.1f} ({args.num_frames} frames)")
    # Reported because a KV-cache policy's headline claim is memory, and
    # nvidia-smi sampling can miss the peak.
    #
    # Two numbers, because the whole-process high-water mark is misleading on
    # short clips: loading the umT5 encoder peaks around 23 GiB and is freed
    # before sampling, so any cache smaller than that is invisible under it.
    # The sampling-window figure resets the counter just before the rollout and
    # is the one that actually reflects the KV cache.
    log.success(f"  Peak CUDA mem: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB "
                f"whole process (reserved {torch.cuda.max_memory_reserved() / 2**30:.2f} GiB)")
    if _SAMPLING_PEAK_BYTES is not None:
        log.success(f"  Peak sampling: {_SAMPLING_PEAK_BYTES / 2**30:.2f} GiB "
                    f"(DiT rollout only, umT5 load excluded)")

    to_show = (1.0 + video.float().cpu().unsqueeze(0).clamp(-1, 1)) / 2.0
    save_image_or_video(rearrange(to_show, "n b c t h w -> c t (n h) (b w)"), args.save_path, fps=16)
    log.success(f"Saved to {args.save_path}")

    if retained_k_observer is not None:
        if not retained_k_observer.records:
            raise RuntimeError(
                "PyramidKV retained-K telemetry was requested but no records were captured"
            )
        retained_k_observer.write_jsonl(args.pyramid_retained_k_output)
        log.success(
            f"Saved {len(retained_k_observer.records)} retained-K records to "
            f"{args.pyramid_retained_k_output}"
        )

    # --- Persist frame-level attention ---
    if attn_observer is not None:
        if (T_latent - args.first_chunk_t) % args.chunk_t != 0:
            raise RuntimeError(
                "latent-frame geometry does not divide exactly into the configured chunks"
            )
        expected_blocks = 1 + (T_latent - args.first_chunk_t) // args.chunk_t
        if not attn_observer.observed_blocks:
            raise RuntimeError(
                f"the observer never fired with --attn_capture_step {args.attn_capture_step}; "
                "refusing to write an all-zero matrix"
            )
        coverage = attn_observer.coverage(expected_blocks)
        if args.attn_layout == "runs" and T_latent < 69:
            log.warning(
                "--attn_layout runs targets the spectral analysis, which slices "
                "last_block_frame_attention[0:69]; this plumbing run is too short "
                "for classification"
            )
        block_sizes = attn_observer.block_sizes(expected_blocks)
        last_block_q_start = sum(block_sizes[:-1])
        last_block_q_frames = list(range(last_block_q_start, T_latent))
        capture_protocol = f"rcm{T_latent}-{args.attn_capture_step}-v1"
        capture_description = (
            "kv-append (t=0, clean query x clean key), once per chunk"
            if args.attn_capture_step == "append"
            else "denoise step 0 (noisy query x clean key), once per chunk"
        )
        # The spectral consumer globs a bare layer<N>.pt, so 'runs' drops the tag.
        artifacts = []
        for layer_index, out_path in output_paths.items():
            full_frame_attn = attn_observer.full[layer_index]
            last_block_frame_attn = full_frame_attn[:, last_block_q_start:T_latent, :].mean(dim=1)
            torch.save(
                {
                    # --- schema consumed by the analysis notebooks ---
                    "schema_version": SCHEMA_VERSION,
                    "layer_index": layer_index,
                    "full_frame_attention": full_frame_attn.to(torch.float16),
                    "last_block_frame_attention": last_block_frame_attn.to(torch.float16),
                    "is_logits": True,
                    "prompt": args.prompt,
                    "num_frames": T_latent,
                    "frame_seq_length": attn_observer.frame_tokens,
                    "num_frame_per_block": args.chunk_t,
                    "num_heads": attn_observer.num_heads,
                    "block_sizes": block_sizes,
                    "query_frames": list(range(T_latent)),
                    "key_frames": list(range(T_latent)),
                    "last_block_query_frames": last_block_q_frames,
                    "extraction_method": f"rcm-observer-{args.attn_method}",
                    "chunk_frames": args.chunk_t,
                    # --- rCM-specific provenance ---
                    "capture_protocol": capture_protocol,
                    "model": f"Causal-rCM Wan2.1 T2V {args.model_size}",
                    "dit_path": args.dit_path,
                    "dit_sha256": attention_checkpoint_sha256,
                    "first_chunk_t": args.first_chunk_t,
                    "chunk_t": args.chunk_t,
                    "num_steps": args.num_steps,
                    "seed": args.seed,
                    "sample_count": args.num_samples,
                    "resolution": args.resolution,
                    "aspect_ratio": args.aspect_ratio,
                    "capture_pass": capture_description,
                    "attn_capture_step": args.attn_capture_step,
                    "coverage": coverage["layers"][str(layer_index)],
                },
                out_path,
            )
            artifacts.append(
                {
                    "layer_index": layer_index,
                    "path": out_path.name,
                    "sha256": sha256_file(out_path),
                    "shape": [attn_observer.num_heads, T_latent, T_latent],
                    "dtype": "float16",
                }
            )

        repo_root = Path(__file__).resolve().parents[2]
        try:
            flash_attn_version = importlib.metadata.version("flash-attn")
        except importlib.metadata.PackageNotFoundError:
            flash_attn_version = None
        run_metadata = {
            "schema_version": SCHEMA_VERSION,
            "capture_protocol": capture_protocol,
            "model": "Causal-rCM Wan2.1 T2V",
            "model_size": args.model_size,
            "num_layers": len(net.blocks),
            "num_heads": net.num_heads,
            "layer_indices": sorted(attn_observer.full),
            "prompt": args.prompt,
            "prompt_sha256": hashlib.sha256(args.prompt.encode("utf-8")).hexdigest(),
            "seed": args.seed,
            "sample_count": args.num_samples,
            "pixel_geometry": {"width": w, "height": h},
            "latent_geometry": {
                "frames": T_latent,
                "height": state_shape[2],
                "width": state_shape[3],
                "patchified_height": state_shape[2] // net.patch_size[1],
                "patchified_width": state_shape[3] // net.patch_size[2],
            },
            "frame_seq_length": attn_observer.frame_tokens,
            "num_frames_latent": T_latent,
            "first_chunk_frames": args.first_chunk_t,
            "chunk_frames": args.chunk_t,
            "capture_pass": args.attn_capture_step,
            "capture_description": capture_description,
            "full_horizon": True,
            "dtype": {
                "model": str(TENSOR_KWARGS["dtype"]).removeprefix("torch."),
                "artifact": "float16",
            },
            "command_argv": [sys.executable, *sys.argv],
            "checkpoint": {
                "path": str(Path(args.dit_path).resolve()),
                "sha256": attention_checkpoint_sha256,
            },
            "repository": {
                "path": str(repo_root),
                "revision": git_revision(repo_root),
                "worktree": git_worktree_state(repo_root),
            },
            "runtime": {
                "python": platform.python_version(),
                "python_executable": sys.executable,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
                "flash_attn": flash_attn_version,
                "gpu": torch.cuda.get_device_name(torch.cuda.current_device()),
            },
            "verification": {
                "enabled": args.attn_verify,
                "stats": attn_observer.verification_stats,
            },
            "coverage": coverage,
            "artifacts": artifacts,
        }
        validate_run_metadata(run_metadata)
        write_json(manifest_path, run_metadata)
        log.success(
            f"Saved {len(artifacts)} attention artifact(s) and {manifest_path}"
        )
