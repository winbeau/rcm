# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import collections
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple, Literal

import attrs
import numpy as np
import torch
import torch._dynamo
import torch.distributed.checkpoint as dcp
from einops import rearrange, repeat
from megatron.core import parallel_state
from torch import Tensor
from torch.distributed._composable.fsdp import FSDPModule, fully_shard, MixedPrecisionPolicy
from torch.distributed._tensor.api import DTensor
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict, set_model_state_dict
from torch.nn.modules.module import _IncompatibleKeys
from imaginaire.config import ObjectStoreConfig
from imaginaire.lazy_config import LazyCall as L
from imaginaire.lazy_config import LazyDict
from imaginaire.lazy_config import instantiate as lazy_instantiate
from imaginaire.model import ImaginaireModel
from imaginaire.utils import log, misc
from imaginaire.utils.easy_io import easy_io
from imaginaire.utils.ema import FastEmaModelUpdater
from rcm.conditioner import DataType, TextCondition
from rcm.utils.optim_instantiate_dtensor import get_base_scheduler
from rcm.utils.checkpointer import non_strict_load_model
from rcm.utils.model_utils import load_state_dict
from rcm.utils.context_parallel import broadcast
from rcm.utils.dtensor_helper import DTensorFastEmaModelUpdater, broadcast_dtensor_model_states
from rcm.utils.fsdp_helper import hsdp_device_mesh
from rcm.utils.jvp_helper import TensorWithT
from rcm.utils.misc import count_params
from rcm.utils.torch_future import clip_grad_norm_
from rcm.utils.denoiser_scaling import RectifiedFlow_TrigFlowWrapper
from rcm.utils.timestep_utils import LogNormal, UniformShift, shift_rf_time, sample_fopp_rf_chunk_times
from rcm.utils.blockmask import BlockPattern, AttnMaskSpec
from rcm.utils.kv_cache import KVCacheMode, CausalInferenceState
from rcm.configs.defaults.ema import EMAConfig
from rcm.samplers.euler import FlowEulerSampler
from rcm.samplers.unipc import FlowUniPCMultistepSampler

IS_PROCESSED_KEY = "is_processed"


@attrs.define(slots=False)
class T2VCausalConfig:

    tokenizer: LazyDict = None
    conditioner: LazyDict = None
    net: LazyDict = None
    net_causal_teacher: LazyDict = None
    net_bidirectional_teacher: LazyDict = None
    net_fake_score: LazyDict = None
    optimizer_fake_score: LazyDict = None
    causal_teacher_ckpt: str = ""
    bidirectional_teacher_ckpt: str = ""
    fake_score_ckpt: str = ""
    tangent_warmup: int = 0
    bidirectional_guidance: float = 5.0
    causal_guidance: float = 3.0
    grad_clip: bool = False
    sigma_max: float = 1600

    ema: EMAConfig = EMAConfig()
    checkpoint: ObjectStoreConfig = ObjectStoreConfig()
    p_G: LazyDict = L(LogNormal)(p_mean=-0.8, p_std=1.6)
    p_D: LazyDict = L(UniformShift)(shift=5.0)
    student_update_freq: int = 5
    fsdp_shard_size: int = 1
    sigma_data: float = 1.0
    precision: str = "bfloat16"
    input_data_key: str = "videos"
    input_latent_key: str = "latents"
    input_caption_key: str = "prompts"
    loss_scale: float = 100.0
    loss_scale_dmd: float = 1.0
    loss_scale_fake_score: float = 1.0
    max_simulation_steps_fake: int = 4
    neg_embed_path: str = ""
    sample_timestep_shift: float = 3

    state_ch: int = 16
    state_t: int = 21  # Number of latent frames
    resolution: str = "480p"
    rectified_flow_t_scaling_factor: float = 1000.0

    text_encoder_class: str = "umT5"
    text_encoder_path: str = ""

    # chunks: [first_chunk_t, chunk_t, chunk_t, ...]
    first_chunk_t: int = 1
    chunk_t: int = 1
    student_ckpt: str = ""
    # df: diffusion-forcing; tf: teacher-forcing; sf: self-forcing
    # Multi-Step:
    # - "df"/"tf": diffusion-forcing/teacher-forcing-based causal diffusion training on video data
    # Few-Step:
    # - "tf_dcm": teacher-forcing-based discrete-time consistency model, causal teacher
    # - "tf_scm": teacher-forcing-based continuous-time consistency model, causal teacher
    # - "sf_dmd": self-forcing-based distribution-matching distillation, bidirectional teacher
    # - "tf_scm_sf_dmd": joint TF-SCM + SF-DMD training, causal teacher + bidirection teacher
    training_type: Literal["df", "tf", "tf_dcm", "tf_scm", "tf_dscm", "sf_dmd", "tf_scm_sf_dmd"] = "tf"
    replayed_training: bool = False
    # uniform + timeshift 5
    backward_timesteps: list = [15 / 16, 5 / 6, 5 / 8]
    shift3_backward: bool = False
    dcm_total_steps: int = 48
    dcm_skipping_interval_steps: int = 1

    df_timestep_strategy: Literal["random", "same", "fopp"] = "random"
    loss_weighting: Literal["none", "gaussian_bell"] = "none"

    # "Custom Step Schedule" feature
    # Optional maximum rollout steps per autoregressive chunk for self-forcing.
    # Empty preserves the old global cycle through 1..max_simulation_steps_fake.
    # A short non-increasing list repeats its last value. The first chunk sets
    # the global cycle, later chunks use min(global_steps, chunk_max_steps).
    sf_simulation_steps_per_chunk: list = []
    # Optional midpoint schedules aligned with sf_simulation_steps_per_chunk.
    # Example: sf_simulation_steps_per_chunk=[4, 2] and
    # sf_backward_timestep_schedules=[[15 / 16, 5 / 6, 5 / 8], [5 / 6]]
    # means chunk 0 uses the 4-step schedule and later chunks use the 2-step
    # schedule. Missing schedules fall back to backward_timesteps prefixes.
    sf_backward_timestep_schedules: list = []

    # "Noisy Context" feature
    # Cache KV states from the last denoising step instead of running an extra
    # forward pass on clean (t=0) data.  Reduces train/inference mismatch and
    # saves one forward pass per chunk during inference.
    context_from_last_step: bool = False
    # First causal chunk index that uses context_from_last_step. Earlier chunks
    # still append clean t=0 K/V even when context_from_last_step is enabled.
    context_from_last_step_start_chunk: int = 0


class T2VCausalModel(ImaginaireModel):

    def __init__(self, config: T2VCausalConfig):
        super().__init__()

        self.config = config
        if self.config.shift3_backward:
            self.config.backward_timesteps = [0.9, 0.75, 0.5]

        self.precision = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[config.precision]
        self.tensor_kwargs = {"device": "cuda", "dtype": self.precision}
        log.warning(f"DiffusionModel: precision {self.precision}")

        # 1. setup up diffusion processing and scaling~(pre-condition), sampler
        self.p_G = lazy_instantiate(config.p_G)
        self.p_D = lazy_instantiate(config.p_D)
        self.scaling = RectifiedFlow_TrigFlowWrapper(config.sigma_data, config.rectified_flow_t_scaling_factor)
        if config.neg_embed_path:
            self.neg_embed = easy_io.load(config.neg_embed_path)
        else:
            self.neg_embed = None

        # 2. tokenizer
        with misc.timer("DiffusionModel: set_up_tokenizer"):
            self.tokenizer = lazy_instantiate(config.tokenizer)
            assert self.tokenizer.latent_ch == config.state_ch, f"latent_ch {self.tokenizer.latent_ch} != state_shape {config.state_ch}"

        # 3. create fsdp mesh if needed
        if config.fsdp_shard_size > 1:
            log.info(f"FSDP size: {config.fsdp_shard_size}")
            self.fsdp_device_mesh = hsdp_device_mesh(sharding_group_size=config.fsdp_shard_size)
        else:
            self.fsdp_device_mesh = None

        # 4. diffusion neural networks part
        self.set_up_model()

        # 5. training states
        if parallel_state.is_initialized():
            self.data_parallel_size = parallel_state.get_data_parallel_world_size()
        else:
            self.data_parallel_size = 1

        torch._dynamo.config.suppress_errors = True
        torch._inductor.config.wrap_inductor_compiled_regions = True

    def build_net(self, net_dict: LazyDict):
        init_device = "meta"
        with misc.timer("Creating PyTorch model"):
            with torch.device(init_device):
                net = lazy_instantiate(net_dict)

            if self.fsdp_device_mesh:
                net.fully_shard(mesh=self.fsdp_device_mesh, mp_policy=MixedPrecisionPolicy(reduce_dtype=torch.float32))
                net = fully_shard(
                    net, mesh=self.fsdp_device_mesh, mp_policy=MixedPrecisionPolicy(reduce_dtype=torch.float32), reshard_after_forward=True
                )

            with misc.timer("meta to cuda and broadcast model states"):
                net.to_empty(device="cuda")
                net.init_weights()

            if self.fsdp_device_mesh:
                broadcast_dtensor_model_states(net, self.fsdp_device_mesh)
                for name, param in net.named_parameters():
                    assert isinstance(param, DTensor), f"param should be DTensor, {name} got {type(param)}"
        return net

    @staticmethod
    def _preprocess_raw_state_dict(state_dict, net):
        """Strip common wrapper prefixes and reshape patch embeddings."""
        for key in ["generator", "generator_ema"]:
            if key in state_dict:
                state_dict = state_dict[key]
        for prefix in ["model.", "net.", "_fsdp_wrapped_module."]:
            state_dict = {(k[len(prefix) :] if k.startswith(prefix) else k): v for k, v in state_dict.items()}
        if hasattr(net, "patch_embedding"):
            for k, v in state_dict.items():
                if k.endswith("patch_embedding.weight"):
                    state_dict[k] = v.reshape(net.patch_embedding.weight.shape)
                if k.endswith("patch_embedding.bias"):
                    state_dict[k] = v.reshape(net.patch_embedding.bias.shape)
        return state_dict

    @staticmethod
    def _is_dcp_dir(path: str) -> bool:
        return os.path.isdir(path) and os.path.exists(os.path.join(path, ".metadata"))

    def load_ckpt_to_net(self, net, ckpt_path):
        if not ckpt_path:
            return

        if self._is_dcp_dir(ckpt_path):
            self._load_ckpt_dcp(net, ckpt_path)
        else:
            self._load_ckpt_raw(net, ckpt_path)

    def _load_ckpt_dcp(self, net, ckpt_path):
        """Load from a distributed checkpoint directory."""
        storage_reader = FileSystemReader(ckpt_path)
        _state_dict = get_model_state_dict(net)

        metadata = storage_reader.read_metadata()
        checkpoint_keys = metadata.state_dict_metadata.keys()

        model_keys = set(_state_dict.keys())

        prefix = "net_ema" if any(k.startswith("net_ema.") for k in checkpoint_keys) else "net"

        prefixed_model_keys = {f"{prefix}.{k}" for k in model_keys}

        missing_keys = prefixed_model_keys - checkpoint_keys
        unexpected_keys = checkpoint_keys - prefixed_model_keys

        if missing_keys:
            log.warning(f"Missing keys in checkpoint: {missing_keys}")
        if unexpected_keys:
            log.warning(f"Unexpected keys in checkpoint: {unexpected_keys}")
        if not missing_keys and not unexpected_keys:
            log.info("All keys matched successfully.")

        _new_state_dict = collections.OrderedDict()
        for k in _state_dict.keys():
            if "_extra_state" in k:
                log.warning(k)
            _new_state_dict[f"{prefix}.{k}"] = _state_dict[k]
        dcp.load(_new_state_dict, storage_reader=storage_reader, planner=DefaultLoadPlanner(allow_partial_load=True))
        for k in _state_dict.keys():
            _state_dict[k] = _new_state_dict[f"{prefix}.{k}"]

        log.info(set_model_state_dict(net, _state_dict, options=StateDictOptions(strict=False)))
        del _state_dict, _new_state_dict

    def _load_ckpt_raw(self, net, ckpt_path):
        """Load from a raw .pt/.pth/.safetensors."""
        raw_sd = load_state_dict(ckpt_path)
        raw_sd = self._preprocess_raw_state_dict(raw_sd, net)
        log.info(f"Loading raw checkpoint from {ckpt_path} ({len(raw_sd)} keys)")
        log.info(set_model_state_dict(net, raw_sd, options=StateDictOptions(strict=False, full_state_dict=True)))
        del raw_sd

    @misc.timer("DiffusionModel: set_up_model")
    def set_up_model(self):
        config = self.config
        with misc.timer("Creating PyTorch model and ema if enabled"):
            self.conditioner = lazy_instantiate(config.conditioner)
            assert sum(p.numel() for p in self.conditioner.parameters() if p.requires_grad) == 0, "conditioner should not have learnable parameters"
            self.net = self.build_net(config.net)
            self.net_causal_teacher = self.build_net(config.net_causal_teacher) if config.net_causal_teacher else None
            self.net_bidirectional_teacher = self.build_net(config.net_bidirectional_teacher) if config.net_bidirectional_teacher else None
            self.net_fake_score = self.build_net(config.net_fake_score) if config.net_fake_score else None
            if config.net_fake_score:
                assert config.loss_scale_dmd > 0
            self.load_ckpt_to_net(self.net, config.student_ckpt)
            if self.net_causal_teacher:
                self.load_ckpt_to_net(self.net_causal_teacher, config.causal_teacher_ckpt)
                self.net_causal_teacher.requires_grad_(False)
            if self.net_bidirectional_teacher:
                self.load_ckpt_to_net(self.net_bidirectional_teacher, config.bidirectional_teacher_ckpt)
                self.net_bidirectional_teacher.requires_grad_(False)
            if self.net_fake_score:
                if config.fake_score_ckpt:
                    self.load_ckpt_to_net(self.net_fake_score, config.fake_score_ckpt)
                elif self.net_bidirectional_teacher:
                    self.net_fake_score.load_state_dict(self.net_bidirectional_teacher.state_dict())
            self._param_count = count_params(self.net, verbose=False)

            # Enable/disable CP once; all CP comm/split/gather happens inside net.forward now.
            cp_group = self.get_context_parallel_group()
            if cp_group is not None and cp_group.size() > 1:
                self.net.enable_context_parallel(cp_group)
                if self.net_causal_teacher:
                    self.net_causal_teacher.enable_context_parallel(cp_group)
                if self.net_bidirectional_teacher:
                    self.net_bidirectional_teacher.enable_context_parallel(cp_group)
                if self.net_fake_score:
                    self.net_fake_score.enable_context_parallel(cp_group)
            else:
                self.net.disable_context_parallel()
                if self.net_causal_teacher:
                    self.net_causal_teacher.disable_context_parallel()
                if self.net_bidirectional_teacher:
                    self.net_bidirectional_teacher.disable_context_parallel()
                if self.net_fake_score:
                    self.net_fake_score.disable_context_parallel()

            if config.ema.enabled:
                self.net_ema = self.build_net(config.net)
                self.net_ema.requires_grad_(False)

                if self.fsdp_device_mesh:
                    self.net_ema_worker = DTensorFastEmaModelUpdater()
                else:
                    self.net_ema_worker = FastEmaModelUpdater()

                s = config.ema.rate
                self.ema_exp_coefficient = np.roots([1, 7, 16 - s**-2, 12 - s**-2]).real.max()

                self.net_ema_worker.copy_to(src_model=self.net, tgt_model=self.net_ema)
        torch.cuda.empty_cache()

    def init_optimizer_scheduler(self, optimizer_config: LazyDict, scheduler_config: LazyDict):
        """Creates the optimizer and scheduler for the model."""
        # instantiate the net optimizer
        net_optimizer = lazy_instantiate(optimizer_config, model=self.net)
        self.optimizer_dict = {"net": net_optimizer}

        # instantiate the net scheduler
        net_scheduler = get_base_scheduler(net_optimizer, self, scheduler_config)
        self.scheduler_dict = {"net": net_scheduler}

        if self.net_fake_score:
            # instantiate the optimizer and lr scheduler for fake_score
            fake_score_optimizer = lazy_instantiate(self.config.optimizer_fake_score, model=self.net_fake_score)
            fake_score_scheduler = get_base_scheduler(fake_score_optimizer, self, scheduler_config)
            self.optimizer_dict["fake_score"] = fake_score_optimizer
            self.scheduler_dict["fake_score"] = fake_score_scheduler

    def is_student_phase(self, iteration: int):
        return (
            self.net_fake_score is None
            or iteration < self.config.tangent_warmup
            or (iteration - self.config.tangent_warmup) % self.config.student_update_freq == 0
        )

    def get_effective_iteration(self, iteration: int):
        return (
            iteration
            if self.net_fake_score is None or iteration < self.config.tangent_warmup
            else self.config.tangent_warmup + (iteration - self.config.tangent_warmup) // self.config.student_update_freq
        )

    def get_effective_iteration_fake(self, iteration: int):
        return iteration - self.get_effective_iteration(iteration) - 1

    # ------------------------ training hooks ------------------------
    def on_before_zero_grad(self, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler, iteration: int) -> None:
        """
        update the net_ema
        """

        if self.config.ema.enabled and self.is_student_phase(iteration):
            # calculate beta for EMA update
            ema_beta = self.ema_beta(self.get_effective_iteration(iteration))
            self.net_ema_worker.update_average(self.net, self.net_ema, beta=ema_beta)

    def on_train_start(self, memory_format: torch.memory_format = torch.preserve_format) -> None:
        if self.config.ema.enabled:
            self.net_ema.to(dtype=torch.float32)
        if hasattr(self.tokenizer, "reset_dtype"):
            self.tokenizer.reset_dtype()
        self.net = self.net.to(memory_format=memory_format, **self.tensor_kwargs)
        if self.net_causal_teacher:
            self.net_causal_teacher = self.net_causal_teacher.to(memory_format=memory_format, **self.tensor_kwargs)
        if self.net_bidirectional_teacher:
            self.net_bidirectional_teacher = self.net_bidirectional_teacher.to(memory_format=memory_format, **self.tensor_kwargs)
        if self.net_fake_score:
            self.net_fake_score = self.net_fake_score.to(memory_format=memory_format, **self.tensor_kwargs)

    # ------------------------ training ------------------------

    def get_optimizers(self, iteration: int) -> list[torch.optim.Optimizer]:
        """
        Get the optimizers for the current iteration
        Args:
            iteration (int): The current training iteration

        """
        if self.is_student_phase(iteration):
            return [self.optimizer_dict["net"]]
        else:
            return [self.optimizer_dict["fake_score"]]

    def get_lr_schedulers(self, iteration: int) -> list[torch.optim.lr_scheduler.LRScheduler]:
        """
        Get the lr schedulers for the current iteration
        Args:
            iteration (int): The current training iteration

        """
        if self.is_student_phase(iteration):
            return [self.scheduler_dict["net"]]
        else:
            return [self.scheduler_dict["fake_score"]]

    def _require_causal_teacher(self):
        teacher = self.net_causal_teacher
        assert teacher is not None, "Causal teacher is required for this training mode."
        return teacher

    def _require_bidirectional_teacher(self):
        teacher = self.net_bidirectional_teacher
        assert teacher is not None, "Bidirectional teacher is required for this training mode."
        return teacher

    def _sample_rf_time(self, sampler, time_shape: Any) -> torch.Tensor:
        assert isinstance(time_shape, (int, tuple, list, torch.Size)), f"Unsupported time shape type: {type(time_shape)}"
        sampled = sampler(shape=time_shape, device="cuda", dtype=torch.float64)
        domain = getattr(sampler, "output_domain", "rf")
        assert domain == "rf", f"Expected RF-domain timestep sampler, got {domain}"
        return sampled.clamp(min=0.0, max=1.0)

    def draw_training_time_G(self, time_shape: Any) -> torch.Tensor:
        return self._sample_rf_time(self.p_G, time_shape)

    def draw_training_time_D(self, time_shape: Any) -> torch.Tensor:
        return self._sample_rf_time(self.p_D, time_shape)

    def _loss_weight(self, t: torch.Tensor) -> torch.Tensor:
        """Compute per-frame loss weight w(t) based on the configured strategy.
        t is in RF domain [0, 1]. Returns a weight tensor broadcastable to (B, T)."""
        if self.config.loss_weighting == "none":
            return 1.0
        if self.config.loss_weighting == "gaussian_bell":
            return torch.exp(-2 * (t - 0.5) ** 2) - torch.exp(torch.tensor(-0.5, device=t.device, dtype=t.dtype))
        raise ValueError(f"Unknown loss_weighting: {self.config.loss_weighting}")

    def _use_context_from_last_step(self, block_idx: int) -> bool:
        if self.config.context_from_last_step_start_chunk < 0:
            raise ValueError(f"context_from_last_step_start_chunk must be non-negative, got {self.config.context_from_last_step_start_chunk}")
        return self.config.context_from_last_step and block_idx >= self.config.context_from_last_step_start_chunk

    def _max_simulation_steps_per_chunk(self, num_blocks: int) -> list[int]:
        if self.config.sf_simulation_steps_per_chunk:
            return self._expand_simulation_max_steps_per_chunk(num_blocks)
        return [self.config.max_simulation_steps_fake] * num_blocks

    def _backward_timestep_schedules_per_chunk(self, num_blocks: int) -> list[list[float]]:
        max_step_counts = self._max_simulation_steps_per_chunk(num_blocks)
        if self.config.sf_backward_timestep_schedules:
            if not self.config.sf_simulation_steps_per_chunk:
                raise ValueError("sf_backward_timestep_schedules requires sf_simulation_steps_per_chunk")
            if len(self.config.sf_backward_timestep_schedules) != len(self.config.sf_simulation_steps_per_chunk):
                raise ValueError(
                    "sf_backward_timestep_schedules must align with sf_simulation_steps_per_chunk: "
                    f"got {len(self.config.sf_backward_timestep_schedules)} schedules for "
                    f"{len(self.config.sf_simulation_steps_per_chunk)} chunk step entries"
                )
            schedules = []
            for max_steps, schedule in zip(self.config.sf_simulation_steps_per_chunk, self.config.sf_backward_timestep_schedules):
                if len(schedule) != int(max_steps) - 1:
                    raise ValueError(f"Schedule {schedule} has length {len(schedule)}, expected {int(max_steps) - 1} for {max_steps} steps")
                schedules.append([float(t) for t in schedule])
            if len(schedules) < num_blocks:
                schedules.extend([schedules[-1]] * (num_blocks - len(schedules)))
            return schedules[:num_blocks]
        return [[float(t) for t in self.config.backward_timesteps[: max_steps - 1]] for max_steps in max_step_counts]

    def _context_time_for_max_schedule(self, max_schedule: list[float]) -> float:
        return 1.0 if not max_schedule else max_schedule[-1]

    def _context_from_last_step_time(self, x_B_C_T_H_W_size, iteration: Optional[int]) -> torch.Tensor:
        B, _, _, _, _, _, num_blocks, block_pattern = self._make_block_pattern(x_B_C_T_H_W_size)
        start_chunk = min(self.config.context_from_last_step_start_chunk, num_blocks)
        final_time_B_chunks = torch.zeros(B, num_blocks, device="cuda", dtype=torch.float64)
        if self.config.training_type == "tf_scm_sf_dmd" and iteration is not None:
            max_schedules = self._backward_timestep_schedules_per_chunk(num_blocks)
            for block_idx, max_schedule in enumerate(max_schedules):
                if block_idx < start_chunk:
                    continue
                final_time_B_chunks[:, block_idx] = self._context_time_for_max_schedule(max_schedule)
        else:
            final_time_B_chunks[:, start_chunk:] = self.config.backward_timesteps[-1]
        return self._expand_chunk_times(final_time_B_chunks, block_pattern)

    def _noisy_ctx(self, x0_B_C_T_H_W: torch.Tensor, time_B_T: torch.Tensor, iteration: Optional[int] = None):
        """Return (ctx_B_C_T_H_W, ctx_time_B_T) with noise added to GT context for the student when enabled."""
        if self.config.context_from_last_step:
            t_ctx = self._context_from_last_step_time(x0_B_C_T_H_W.size(), iteration)
            ctx_noise_B_C_T_H_W = torch.randn_like(x0_B_C_T_H_W)
            if isinstance(t_ctx, torch.Tensor):
                t_ctx_data_B_T = t_ctx.to(device=x0_B_C_T_H_W.device, dtype=x0_B_C_T_H_W.dtype)
                t_ctx_B_1_T_1_1 = rearrange(t_ctx_data_B_T, "b t -> b 1 t 1 1")
                ctx_B_C_T_H_W = (1 - t_ctx_B_1_T_1_1) * x0_B_C_T_H_W + t_ctx_B_1_T_1_1 * ctx_noise_B_C_T_H_W
                ctx_time_B_T = t_ctx.to(dtype=time_B_T.dtype)
            else:
                ctx_B_C_T_H_W = (1 - t_ctx) * x0_B_C_T_H_W + t_ctx * ctx_noise_B_C_T_H_W
                ctx_time_B_T = torch.full_like(time_B_T, t_ctx)
        else:
            ctx_B_C_T_H_W = x0_B_C_T_H_W
            ctx_time_B_T = torch.zeros_like(time_B_T)
        return ctx_B_C_T_H_W, ctx_time_B_T

    def _tf_step(self, data_batch, iteration):
        log.info(f"TF iteration {iteration}")
        _, x0_B_C_T_H_W, condition, uncondition = self.get_data_and_condition(data_batch)
        time_B_1 = self.draw_training_time_G((x0_B_C_T_H_W.shape[0], 1))
        epsilon_B_C_T_H_W = torch.randn(x0_B_C_T_H_W.size(), device="cuda")
        x0_B_C_T_H_W, time_B_1, epsilon_B_C_T_H_W, condition, uncondition = self.sync(
            x0_B_C_T_H_W, time_B_1, epsilon_B_C_T_H_W, condition, uncondition
        )
        B, C, T, H, W, _, num_blocks, block_pattern = self._make_block_pattern(x0_B_C_T_H_W.size(), model=self.net)
        attn_mask_spec = AttnMaskSpec(mode="teacher_forcing", pattern=block_pattern, clean_blocks=num_blocks)
        time_B_T = repeat(time_B_1, "b 1 -> b t", t=T)
        time_B_1_T_1_1 = rearrange(time_B_T, "b t -> b 1 t 1 1")
        xt_B_C_T_H_W = (1 - time_B_1_T_1_1) * x0_B_C_T_H_W + time_B_1_T_1_1 * epsilon_B_C_T_H_W
        ctx_B_C_T_H_W, ctx_time_B_T = self._noisy_ctx(x0_B_C_T_H_W, time_B_T, iteration)
        full_x_B_C_2T_H_W = torch.cat([ctx_B_C_T_H_W, xt_B_C_T_H_W], dim=2)
        full_time_B_2T = torch.cat([ctx_time_B_T, time_B_T], dim=1)
        v_pred_B_C_T_H_W = self.net(
            full_x_B_C_2T_H_W.to(**self.tensor_kwargs),
            (full_time_B_2T * self.config.rectified_flow_t_scaling_factor).to(**self.tensor_kwargs),
            **condition.to_dict(),
            attn_meta=attn_mask_spec,
        ).float()[:, :, T:, :, :]
        v_target = epsilon_B_C_T_H_W - x0_B_C_T_H_W
        w_B_T = self._loss_weight(time_B_T)
        loss = self.config.loss_scale * (w_B_T * ((v_pred_B_C_T_H_W - v_target) ** 2).mean(dim=(1, 3, 4))).sum(dim=1)
        return {}, loss

    @staticmethod
    def _expand_chunk_times(time_B_chunks: torch.Tensor, block_pattern: BlockPattern) -> torch.Tensor:
        num_blocks = time_B_chunks.shape[1]
        expanded = []
        for block_idx in range(num_blocks):
            chunk_t = block_pattern.block_size(block_idx)
            expanded.append(repeat(time_B_chunks[:, block_idx], "b -> b t", t=chunk_t))
        return torch.cat(expanded, dim=1)

    def _draw_df_chunk_times(self, B: int, num_blocks: int, block_pattern: BlockPattern) -> torch.Tensor:
        strategy = self.config.df_timestep_strategy
        if strategy == "random":
            time_B_chunks = self.draw_training_time_G((B, num_blocks))
        elif strategy == "same":
            time_B_1 = self.draw_training_time_G((B, 1))
            time_B_chunks = time_B_1.expand(B, num_blocks)
        elif strategy == "fopp":
            time_B_chunks = sample_fopp_rf_chunk_times(sampler=self.p_G, batch_size=B, num_chunks=num_blocks, device="cuda", dtype=torch.float64)
        else:
            raise ValueError(f"Unknown df_timestep_strategy: {strategy}")
        return self._expand_chunk_times(time_B_chunks, block_pattern)

    def _df_step(self, data_batch, iteration):
        log.info(f"DF iteration {iteration}")
        _, x0_B_C_T_H_W, condition, _ = self.get_data_and_condition(data_batch)
        epsilon_B_C_T_H_W = torch.randn(x0_B_C_T_H_W.size(), device="cuda")
        x0_B_C_T_H_W, epsilon_B_C_T_H_W, condition = self.sync(x0_B_C_T_H_W, epsilon_B_C_T_H_W, condition)
        B, C, T, H, W, _, num_blocks, block_pattern = self._make_block_pattern(x0_B_C_T_H_W.size())
        attn_mask_spec = AttnMaskSpec(mode="block_causal", pattern=block_pattern, q_block_offset=0)

        time_B_T = self._draw_df_chunk_times(B, num_blocks, block_pattern)
        assert time_B_T.shape == (B, T)
        time_B_1_T_1_1 = rearrange(time_B_T, "b t -> b 1 t 1 1")
        xt_B_C_T_H_W = (1 - time_B_1_T_1_1) * x0_B_C_T_H_W + time_B_1_T_1_1 * epsilon_B_C_T_H_W

        v_pred_B_C_T_H_W = self.net(
            xt_B_C_T_H_W.to(**self.tensor_kwargs),
            (time_B_T * self.config.rectified_flow_t_scaling_factor).to(**self.tensor_kwargs),
            **condition.to_dict(),
            attn_meta=attn_mask_spec,
        ).float()
        v_target = epsilon_B_C_T_H_W - x0_B_C_T_H_W
        w_B_T = self._loss_weight(time_B_T)
        loss = self.config.loss_scale * (w_B_T * ((v_pred_B_C_T_H_W - v_target) ** 2).mean(dim=(1, 3, 4))).sum(dim=1)
        return {}, loss

    def _self_forcing_simulation_steps(self, iteration: int, fake_score_update: bool = False, num_blocks: Optional[int] = None) -> int | list[int]:
        effective_iteration = self.get_effective_iteration_fake(iteration) if fake_score_update else self.get_effective_iteration(iteration)
        if self.config.sf_simulation_steps_per_chunk:
            if num_blocks is None:
                return list(self.config.sf_simulation_steps_per_chunk)
            max_step_counts = self._expand_simulation_max_steps_per_chunk(num_blocks)
            global_steps = effective_iteration % max_step_counts[0] + 1
            return [min(global_steps, max_steps) for max_steps in max_step_counts]
        return effective_iteration % self.config.max_simulation_steps_fake + 1

    def _expand_simulation_steps_per_chunk(self, n_steps: int | list[int], num_blocks: int) -> list[int]:
        if isinstance(n_steps, int):
            step_counts = [n_steps] * num_blocks
        else:
            if not n_steps:
                raise ValueError("n_steps list must be non-empty")
            step_counts = [int(step) for step in n_steps[:num_blocks]]
            if len(step_counts) < num_blocks:
                step_counts.extend([step_counts[-1]] * (num_blocks - len(step_counts)))

        if any(step < 1 for step in step_counts):
            raise ValueError(f"Simulation step counts must be positive, got {step_counts}")
        return step_counts

    def _expand_simulation_max_steps_per_chunk(self, num_blocks: int) -> list[int]:
        max_step_counts = self._expand_simulation_steps_per_chunk(self.config.sf_simulation_steps_per_chunk, num_blocks)
        if any(prev < cur for prev, cur in zip(max_step_counts, max_step_counts[1:])):
            raise ValueError(f"sf_simulation_steps_per_chunk must be non-increasing after expansion, got {max_step_counts}")
        return max_step_counts

    def _build_backward_time_trajectory(self, batch_size: int, n_steps: int, mid_timesteps: list[float]) -> list[torch.Tensor]:
        G_time_B_1 = torch.ones(batch_size, 1, device="cuda")
        t_traj = [G_time_B_1]
        for i in range(n_steps - 1):
            backward_t = mid_timesteps[i]
            G_time_B_1 = backward_t * torch.ones(batch_size, 1, device="cuda")
            t_traj.append(G_time_B_1)
        return t_traj

    def backward_simulation(self, condition, x_B_C_T_H_W_size, n_steps, with_grad: bool = False):
        B, C, T, H, W, frame_tokens, num_blocks, block_pattern = self._make_block_pattern(x_B_C_T_H_W_size, model=self.net)
        step_counts = self._expand_simulation_steps_per_chunk(n_steps, num_blocks)
        max_step_counts = self._max_simulation_steps_per_chunk(num_blocks)
        max_schedules = self._backward_timestep_schedules_per_chunk(num_blocks)
        max_steps = max(step_counts)
        t_trajs = [
            self._build_backward_time_trajectory(B, steps, max_schedule[: steps - 1]) for steps, max_schedule in zip(step_counts, max_schedules)
        ]
        # Self-forcing replay treats rollout prefixes as generated context; keep
        # those caches detached to avoid backpropagating through earlier rollout
        # blocks.
        kv_caches = self.net.allocate_kv_caches(max_len=T * frame_tokens)
        noises = []
        for i in range(max_steps):
            noises.append(self.sync(torch.randn(x_B_C_T_H_W_size, device="cuda")))
        G_xt_blocks, G_time_blocks, G_x0_blocks = [], [], []
        for block_idx in range(num_blocks):
            n_steps_block = step_counts[block_idx]
            t_traj_block = t_trajs[block_idx]
            frame_start, frame_end, block_size = self._block_span(block_pattern, block_idx)
            attn_mask_spec = AttnMaskSpec(mode="block_causal", pattern=block_pattern, q_block_offset=block_idx)
            cache_last = self._use_context_from_last_step(block_idx)
            max_steps_block = max_step_counts[block_idx]
            max_schedule_block = max_schedules[block_idx]
            cache_rollout_state = cache_last and n_steps_block == max_steps_block
            x_append_B_C_block_H_W = None
            t_append_B_1 = None
            inference_state_readonly = CausalInferenceState(
                mode=KVCacheMode.READONLY, kv_caches=kv_caches, pattern=block_pattern, block_cursor=block_idx
            )
            inference_state_append = CausalInferenceState(mode=KVCacheMode.APPEND, kv_caches=kv_caches, pattern=block_pattern, block_cursor=block_idx)
            for step, t_cur_B_1 in enumerate(t_traj_block):
                noise_B_C_block_H_W = noises[step][:, :, frame_start:frame_end]
                t_cur_B_block = repeat(t_cur_B_1, "b 1 -> b t", t=block_size)
                t_cur_B_1_block_1_1 = rearrange(t_cur_B_block, "b t -> b 1 t 1 1")
                x_B_C_block_H_W = (
                    noise_B_C_block_H_W if step == 0 else (1 - t_cur_B_1_block_1_1) * x_B_C_block_H_W + t_cur_B_1_block_1_1 * noise_B_C_block_H_W
                )
                is_last_step = step == n_steps_block - 1
                if not with_grad and is_last_step:
                    G_xt_blocks.append(x_B_C_block_H_W)
                    G_time_blocks.append(t_cur_B_block)
                if cache_rollout_state and is_last_step:
                    x_append_B_C_block_H_W = x_B_C_block_H_W.detach()
                    t_append_B_1 = t_cur_B_1.detach()
                context_fn = torch.enable_grad if with_grad and is_last_step else torch.no_grad
                with context_fn():
                    v_pred_B_C_block_H_W = self.net(
                        x_B_C_block_H_W.to(**self.tensor_kwargs),
                        (t_cur_B_1 * self.config.rectified_flow_t_scaling_factor).to(**self.tensor_kwargs),
                        **condition.to_dict(),
                        inference_state=inference_state_readonly,
                        attn_meta=attn_mask_spec,
                    ).float()
                    x_B_C_block_H_W = x_B_C_block_H_W - t_cur_B_1_block_1_1 * v_pred_B_C_block_H_W
            G_x0_blocks.append(x_B_C_block_H_W)
            if cache_rollout_state:
                assert x_append_B_C_block_H_W is not None and t_append_B_1 is not None
                with torch.no_grad():
                    self.net(
                        x_append_B_C_block_H_W.to(**self.tensor_kwargs),
                        (t_append_B_1 * self.config.rectified_flow_t_scaling_factor).to(**self.tensor_kwargs),
                        **condition.to_dict(),
                        inference_state=inference_state_append,
                        attn_meta=attn_mask_spec,
                    )
            elif cache_last:
                t_ctx = self._context_time_for_max_schedule(max_schedule_block)
                t_ctx_B_1 = torch.full((B, 1), t_ctx, device="cuda")
                t_ctx_B_block = repeat(t_ctx_B_1, "b 1 -> b t", t=block_size)
                t_ctx_B_1_block_1_1 = rearrange(t_ctx_B_block, "b t -> b 1 t 1 1")
                with torch.no_grad():
                    ctx_noise_B_C_block_H_W = self.sync(torch.randn_like(x_B_C_block_H_W))
                    x_ctx_B_C_block_H_W = (1 - t_ctx_B_1_block_1_1) * x_B_C_block_H_W.detach() + t_ctx_B_1_block_1_1 * ctx_noise_B_C_block_H_W
                    self.net(
                        x_ctx_B_C_block_H_W.to(**self.tensor_kwargs),
                        (t_ctx_B_1 * self.config.rectified_flow_t_scaling_factor).to(**self.tensor_kwargs),
                        **condition.to_dict(),
                        inference_state=inference_state_append,
                        attn_meta=attn_mask_spec,
                    )
            else:
                with torch.no_grad():
                    self.net(
                        x_B_C_block_H_W.to(**self.tensor_kwargs),
                        (t_cur_B_1 * 0).to(**self.tensor_kwargs),
                        **condition.to_dict(),
                        inference_state=inference_state_append,
                        attn_meta=attn_mask_spec,
                    )
        G_x0_B_C_T_H_W = torch.cat(G_x0_blocks, dim=2)
        if with_grad:
            return G_x0_B_C_T_H_W
        G_xt_B_C_T_H_W = torch.cat(G_xt_blocks, dim=2)
        G_time_B_T = torch.cat(G_time_blocks, dim=1)
        return G_x0_B_C_T_H_W, (G_xt_B_C_T_H_W, G_time_B_T, kv_caches)

    def _sf_step(self, data_batch, iteration):
        log.info(f"SF iteration {iteration}")
        teacher_net = self._require_bidirectional_teacher()
        _, x0_B_C_T_H_W, condition, uncondition = self.get_data_and_condition(data_batch)
        x0_B_C_T_H_W, condition, uncondition = self.sync(x0_B_C_T_H_W, condition, uncondition)
        _, _, _, _, _, _, num_blocks, _ = self._make_block_pattern(x0_B_C_T_H_W.size())
        num_simulation_steps_fake = self._self_forcing_simulation_steps(iteration, num_blocks=num_blocks)
        G_x0_theta_B_C_T_H_W = self.backward_simulation(condition, x0_B_C_T_H_W.size(), num_simulation_steps_fake, with_grad=True)
        D_time_B_1 = self.draw_training_time_D((x0_B_C_T_H_W.shape[0], 1))
        epsilon_B_C_T_H_W = torch.randn(x0_B_C_T_H_W.size(), device="cuda")
        D_time_B_1, epsilon_B_C_T_H_W = self.sync(D_time_B_1, epsilon_B_C_T_H_W)
        D_time_B_1_1_1_1 = rearrange(D_time_B_1, "b 1 -> b 1 1 1 1")
        D_xt_theta_B_C_T_H_W = (1 - D_time_B_1_1_1_1) * G_x0_theta_B_C_T_H_W + D_time_B_1_1_1_1 * epsilon_B_C_T_H_W

        with torch.no_grad():
            v_pred_fake_B_C_T_H_W = self.net_fake_score(
                x_B_C_T_H_W=D_xt_theta_B_C_T_H_W.to(**self.tensor_kwargs),
                timesteps_B_T=(D_time_B_1 * self.config.rectified_flow_t_scaling_factor).to(**self.tensor_kwargs),
                **condition.to_dict(),
            ).float()
            x0_theta_fake_B_C_T_H_W = D_xt_theta_B_C_T_H_W - D_time_B_1_1_1_1 * v_pred_fake_B_C_T_H_W
        with torch.no_grad():
            v_pred_teacher_B_C_T_H_W = self._predict_cfg(
                teacher_net, D_xt_theta_B_C_T_H_W, D_time_B_1, condition, uncondition, self.config.bidirectional_guidance
            )
            x0_theta_teacher_B_C_T_H_W = D_xt_theta_B_C_T_H_W - D_time_B_1_1_1_1 * v_pred_teacher_B_C_T_H_W
        with torch.no_grad():
            weight_factor = (
                torch.abs(G_x0_theta_B_C_T_H_W.double() - x0_theta_teacher_B_C_T_H_W.double()).mean(dim=[1, 2, 3, 4], keepdim=True).clip(min=0.00001)
            )
        grad_B_C_T_H_W = (x0_theta_fake_B_C_T_H_W.double() - x0_theta_teacher_B_C_T_H_W.double()) / weight_factor
        loss_dmd = (G_x0_theta_B_C_T_H_W.double() - (G_x0_theta_B_C_T_H_W.double() - grad_B_C_T_H_W).detach()) ** 2
        loss_dmd[torch.isnan(loss_dmd).flatten(start_dim=1).any(dim=1)] = 0
        loss_dmd = loss_dmd.sum(dim=(1, 2, 3, 4))
        loss = self.config.loss_scale_dmd * loss_dmd
        return {}, loss

    def _make_replayed_sf_ctx(self, data_batch, iteration):
        log.info(f"Replayed SF iteration {iteration}")
        teacher_net = self._require_bidirectional_teacher()
        _, x0_B_C_T_H_W, condition, uncondition = self.get_data_and_condition(data_batch)
        x0_B_C_T_H_W, condition, uncondition = self.sync(x0_B_C_T_H_W, condition, uncondition)
        _, _, _, _, _, _, num_blocks, _ = self._make_block_pattern(x0_B_C_T_H_W.size())
        num_simulation_steps_fake = self._self_forcing_simulation_steps(iteration, num_blocks=num_blocks)
        G_x0_theta_B_C_T_H_W, (G_xt_theta_B_C_T_H_W, G_time_B_T, kv_caches) = self.backward_simulation(
            condition, x0_B_C_T_H_W.size(), num_simulation_steps_fake, with_grad=False
        )
        D_time_B_1 = self.draw_training_time_D((x0_B_C_T_H_W.shape[0], 1))
        epsilon_B_C_T_H_W = torch.randn(x0_B_C_T_H_W.size(), device="cuda")
        D_time_B_1, epsilon_B_C_T_H_W = self.sync(D_time_B_1, epsilon_B_C_T_H_W)
        D_time_B_1_1_1_1 = rearrange(D_time_B_1, "b 1 -> b 1 1 1 1")
        D_xt_theta_B_C_T_H_W = (1 - D_time_B_1_1_1_1) * G_x0_theta_B_C_T_H_W + D_time_B_1_1_1_1 * epsilon_B_C_T_H_W

        with torch.no_grad():
            v_pred_fake_B_C_T_H_W = self.net_fake_score(
                x_B_C_T_H_W=D_xt_theta_B_C_T_H_W.to(**self.tensor_kwargs),
                timesteps_B_T=(D_time_B_1 * self.config.rectified_flow_t_scaling_factor).to(**self.tensor_kwargs),
                **condition.to_dict(),
            ).float()
            x0_theta_fake_B_C_T_H_W = D_xt_theta_B_C_T_H_W - D_time_B_1_1_1_1 * v_pred_fake_B_C_T_H_W
        with torch.no_grad():
            v_pred_teacher_B_C_T_H_W = self._predict_cfg(
                teacher_net, D_xt_theta_B_C_T_H_W, D_time_B_1, condition, uncondition, self.config.bidirectional_guidance
            )
            x0_theta_teacher_B_C_T_H_W = D_xt_theta_B_C_T_H_W - D_time_B_1_1_1_1 * v_pred_teacher_B_C_T_H_W
        with torch.no_grad():
            weight_factor = (
                torch.abs(G_x0_theta_B_C_T_H_W.double() - x0_theta_teacher_B_C_T_H_W.double()).mean(dim=[1, 2, 3, 4], keepdim=True).clip(min=0.00001)
            )
        grad_B_C_T_H_W = (x0_theta_fake_B_C_T_H_W.double() - x0_theta_teacher_B_C_T_H_W.double()) / weight_factor
        target_B_C_T_H_W = (G_x0_theta_B_C_T_H_W.double() - grad_B_C_T_H_W).detach()
        return num_blocks, (G_xt_theta_B_C_T_H_W, G_time_B_T, kv_caches, target_B_C_T_H_W, condition)

    def _replayed_sf_step(self, index, ctx, iteration):
        log.debug(f"Replayed SF index {index}")
        G_xt_theta_B_C_T_H_W, G_time_B_T, kv_caches, target_B_C_T_H_W, condition = ctx
        B, C, T, H, W, _, _, block_pattern = self._make_block_pattern(G_xt_theta_B_C_T_H_W.size())
        frame_start, frame_end, block_size = self._block_span(block_pattern, index)
        G_time_B_block = G_time_B_T[:, frame_start:frame_end]
        G_time_B_1 = G_time_B_block[:, :1]
        G_time_B_1_block_1_1 = rearrange(G_time_B_block, "b t -> b 1 t 1 1")
        G_xt_theta_B_C_block_H_W = G_xt_theta_B_C_T_H_W[:, :, frame_start:frame_end]
        attn_mask_spec = AttnMaskSpec(mode="block_causal", pattern=block_pattern, q_block_offset=index)
        inference_state = CausalInferenceState(mode=KVCacheMode.READONLY, kv_caches=kv_caches, pattern=block_pattern, block_cursor=index)
        v_pred_B_C_block_H_W = self.net(
            G_xt_theta_B_C_block_H_W.to(**self.tensor_kwargs),
            (G_time_B_1 * self.config.rectified_flow_t_scaling_factor).to(**self.tensor_kwargs),
            **condition.to_dict(),
            inference_state=inference_state,
            attn_meta=attn_mask_spec,
        ).float()
        x0_pred_B_C_block_H_W = G_xt_theta_B_C_block_H_W - G_time_B_1_block_1_1 * v_pred_B_C_block_H_W
        loss_dmd = (x0_pred_B_C_block_H_W - target_B_C_T_H_W[:, :, frame_start:frame_end]) ** 2
        loss_dmd[torch.isnan(loss_dmd).flatten(start_dim=1).any(dim=1)] = 0
        loss_dmd = loss_dmd.sum(dim=(1, 2, 3, 4))
        loss = self.config.loss_scale_dmd * loss_dmd
        return {}, loss

    def training_step_critic(self, data_batch, iteration) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        log.info(f"Critic iteration {iteration}")
        _, x0_B_C_T_H_W, condition, uncondition = self.get_data_and_condition(data_batch)
        x0_B_C_T_H_W, condition, uncondition = self.sync(x0_B_C_T_H_W, condition, uncondition)
        _, _, _, _, _, _, num_blocks, _ = self._make_block_pattern(x0_B_C_T_H_W.size())
        num_simulation_steps_fake = self._self_forcing_simulation_steps(iteration, fake_score_update=True, num_blocks=num_blocks)
        G_x0_theta_B_C_T_H_W, _ = self.backward_simulation(condition, x0_B_C_T_H_W.size(), num_simulation_steps_fake, with_grad=False)

        D_time_B_1 = self.draw_training_time_D((x0_B_C_T_H_W.shape[0], 1))
        D_epsilon_B_C_T_H_W = torch.randn(x0_B_C_T_H_W.size(), device="cuda")
        D_time_B_1, D_epsilon_B_C_T_H_W = self.sync(D_time_B_1, D_epsilon_B_C_T_H_W)
        D_time_B_1_1_1_1 = rearrange(D_time_B_1, "b 1 -> b 1 1 1 1")
        D_xt_theta_B_C_T_H_W = (1 - D_time_B_1_1_1_1) * G_x0_theta_B_C_T_H_W + D_time_B_1_1_1_1 * D_epsilon_B_C_T_H_W
        v_pred_fake_B_C_T_H_W = self.net_fake_score(
            x_B_C_T_H_W=D_xt_theta_B_C_T_H_W.to(**self.tensor_kwargs),
            timesteps_B_T=(D_time_B_1 * self.config.rectified_flow_t_scaling_factor).to(**self.tensor_kwargs),
            **condition.to_dict(),
        ).float()
        loss = ((v_pred_fake_B_C_T_H_W - (D_epsilon_B_C_T_H_W - G_x0_theta_B_C_T_H_W)) ** 2).mean(dim=(1, 2, 3, 4))
        return {}, loss

    def _tf_dcm_step(self, data_batch, iteration):
        log.info(f"TF dCM iteration {iteration}")
        teacher_net = self._require_causal_teacher()
        _, x0_B_C_T_H_W, condition, uncondition = self.get_data_and_condition(data_batch)
        epsilon_B_C_T_H_W = torch.randn(x0_B_C_T_H_W.size(), device="cuda")
        x0_B_C_T_H_W, epsilon_B_C_T_H_W, condition, uncondition = self.sync(x0_B_C_T_H_W, epsilon_B_C_T_H_W, condition, uncondition)
        B, C, T, H, W, _, num_blocks, block_pattern = self._make_block_pattern(x0_B_C_T_H_W.size())
        attn_mask_spec = AttnMaskSpec(mode="teacher_forcing", pattern=block_pattern, clean_blocks=num_blocks)

        du = 1.0 / self.config.dcm_total_steps
        u_B_1 = torch.rand(B, 1, device="cuda") * (1.0 - self.config.dcm_skipping_interval_steps * du)
        u_B_1 = self.sync(u_B_1)

        t_list = []
        for k in range(self.config.dcm_skipping_interval_steps + 1):
            s_k_B_1 = 1.0 - (u_B_1 + k * du)
            t_k_B_1 = shift_rf_time(s_k_B_1, self.config.sample_timestep_shift)
            t_list.append(t_k_B_1)

        t0_B_1, tN_B_1 = t_list[0], t_list[-1]
        t0_B_T, tN_B_T = repeat(t0_B_1, "b 1 -> b t", t=T), repeat(tN_B_1, "b 1 -> b t", t=T)

        xt_B_C_T_H_W = (1 - t0_B_1[:, :, None, None, None]) * x0_B_C_T_H_W + t0_B_1[:, :, None, None, None] * epsilon_B_C_T_H_W

        ctx_B_C_T_H_W, ctx_time_B_T = self._noisy_ctx(x0_B_C_T_H_W, t0_B_T, iteration)
        student_full_x_B_C_2T_H_W = torch.cat([ctx_B_C_T_H_W, xt_B_C_T_H_W], dim=2)
        student_full_time_B_2T = torch.cat([ctx_time_B_T, t0_B_T], dim=1)
        v_pred_t0_B_C_T_H_W = self.net(
            student_full_x_B_C_2T_H_W.to(**self.tensor_kwargs),
            (student_full_time_B_2T * self.config.rectified_flow_t_scaling_factor).to(**self.tensor_kwargs),
            **condition.to_dict(),
            attn_meta=attn_mask_spec,
        ).float()[:, :, T:, :, :]
        x0_pred_B_C_T_H_W = xt_B_C_T_H_W - t0_B_1[:, :, None, None, None] * v_pred_t0_B_C_T_H_W

        with torch.no_grad():
            xk_B_C_T_H_W = xt_B_C_T_H_W
            for k in range(self.config.dcm_skipping_interval_steps):
                tk_B_1 = t_list[k]
                tk1_B_1 = t_list[k + 1]
                dt_B_1 = tk_B_1 - tk1_B_1

                full_x_B_C_2T_H_W = torch.cat([x0_B_C_T_H_W, xk_B_C_T_H_W], dim=2)
                tk_B_T = repeat(tk_B_1, "b 1 -> b t", t=T)
                full_time_B_2T = torch.cat([torch.zeros_like(tk_B_T), tk_B_T], dim=1)
                v_pred_teacher_B_C_T_H_W = self._predict_cfg(
                    teacher_net, full_x_B_C_2T_H_W, full_time_B_2T, condition, uncondition, self.config.causal_guidance, attn_meta=attn_mask_spec
                ).float()[:, :, T:, :, :]

                xk_B_C_T_H_W = xk_B_C_T_H_W - dt_B_1[:, :, None, None, None] * v_pred_teacher_B_C_T_H_W

            ctx_tN_B_C_T_H_W, ctx_time_tN_B_T = self._noisy_ctx(x0_B_C_T_H_W, tN_B_T, iteration)
            student_full_x_tN_B_C_2T_H_W = torch.cat([ctx_tN_B_C_T_H_W, xk_B_C_T_H_W], dim=2)
            student_full_time_tN_B_2T = torch.cat([ctx_time_tN_B_T, tN_B_T], dim=1)
            v_pred_tN_B_C_T_H_W = self.net(
                student_full_x_tN_B_C_2T_H_W.to(**self.tensor_kwargs),
                (student_full_time_tN_B_2T * self.config.rectified_flow_t_scaling_factor).to(**self.tensor_kwargs),
                **condition.to_dict(),
                attn_meta=attn_mask_spec,
            ).float()[:, :, T:, :, :]
            x0_pred_prev_B_C_T_H_W = xk_B_C_T_H_W - tN_B_1[:, :, None, None, None] * v_pred_tN_B_C_T_H_W
        loss_cm = ((x0_pred_B_C_T_H_W - x0_pred_prev_B_C_T_H_W) ** 2).mean(dim=(1, 3, 4)).sum(dim=1)
        loss = self.config.loss_scale * loss_cm
        return {}, loss

    def student_v_withT(self, x_B_C_T_H_W_withT: TensorWithT, time_B_T_withT: TensorWithT, condition, attn_meta=None, inference_state=None):
        x_B_C_T_H_W, t_x_B_C_T_H_W = x_B_C_T_H_W_withT
        time_B_T, t_time_B_T = time_B_T_withT

        v_pred_B_C_T_H_W, t_v_pred_B_C_T_H_W = self.net(
            x_B_C_T_H_W=(x_B_C_T_H_W.to(**self.tensor_kwargs), t_x_B_C_T_H_W.to(**self.tensor_kwargs)),
            timesteps_B_T=(
                (time_B_T * self.config.rectified_flow_t_scaling_factor).to(**self.tensor_kwargs),
                (t_time_B_T * self.config.rectified_flow_t_scaling_factor).to(**self.tensor_kwargs),
            ),
            **condition.to_dict(),
            attn_meta=attn_meta,
            inference_state=inference_state,
            withT=True,
        )
        v_pred_B_C_T_H_W, t_v_pred_B_C_T_H_W = v_pred_B_C_T_H_W.float(), t_v_pred_B_C_T_H_W.float()
        return (v_pred_B_C_T_H_W, t_v_pred_B_C_T_H_W.detach())

    def _tf_scm_step(self, data_batch, iteration):
        log.info(f"TF sCM iteration {iteration}")
        teacher_net = self._require_causal_teacher()
        _, x0_B_C_T_H_W, condition, uncondition = self.get_data_and_condition(data_batch)
        time_B_1 = self.draw_training_time_G((x0_B_C_T_H_W.shape[0], 1))
        epsilon_B_C_T_H_W = torch.randn(x0_B_C_T_H_W.size(), device="cuda")
        x0_B_C_T_H_W, time_B_1, epsilon_B_C_T_H_W, condition, uncondition = self.sync(
            x0_B_C_T_H_W, time_B_1, epsilon_B_C_T_H_W, condition, uncondition
        )
        B, C, T, H, W, _, num_blocks, block_pattern = self._make_block_pattern(x0_B_C_T_H_W.size(), model=self.net)
        attn_mask_spec = AttnMaskSpec(mode="teacher_forcing", pattern=block_pattern, clean_blocks=num_blocks)

        time_B_T = repeat(time_B_1, "b 1 -> b t", t=T)
        time_B_1_T_1_1 = rearrange(time_B_T, "b t -> b 1 t 1 1")
        xt_B_C_T_H_W = (1 - time_B_1_T_1_1) * x0_B_C_T_H_W + time_B_1_T_1_1 * epsilon_B_C_T_H_W
        full_x_B_C_2T_H_W = torch.cat([x0_B_C_T_H_W, xt_B_C_T_H_W], dim=2)
        full_time_B_2T = torch.cat([torch.zeros_like(time_B_T), time_B_T], dim=1)

        ctx_B_C_T_H_W, ctx_time_B_T = self._noisy_ctx(x0_B_C_T_H_W, time_B_T, iteration)
        student_full_x_B_C_2T_H_W = torch.cat([ctx_B_C_T_H_W, xt_B_C_T_H_W], dim=2)
        student_full_time_B_2T = torch.cat([ctx_time_B_T, time_B_T], dim=1)

        with torch.no_grad():
            v_teacher_B_C_T_H_W = self._predict_cfg(
                teacher_net, full_x_B_C_2T_H_W, full_time_B_2T, condition, uncondition, self.config.causal_guidance, attn_meta=attn_mask_spec
            ).float()[:, :, T:, :, :]

        F_teacher_B_C_T_H_W = v_teacher_B_C_T_H_W
        t_xt_B_C_T_H_W = F_teacher_B_C_T_H_W
        t_time_B_T = torch.ones_like(time_B_T)
        t_student_full_x_B_C_2T_H_W = torch.cat([torch.zeros_like(x0_B_C_T_H_W), t_xt_B_C_T_H_W], dim=2)
        t_student_full_time_B_2T = torch.cat([torch.zeros_like(time_B_T), t_time_B_T], dim=1)

        with torch.no_grad():
            _, t_v_theta_B_C_2T_H_W = self.student_v_withT(
                (student_full_x_B_C_2T_H_W, t_student_full_x_B_C_2T_H_W),
                (student_full_time_B_2T, t_student_full_time_B_2T),
                condition,
                attn_meta=attn_mask_spec,
            )
            t_v_theta_B_C_T_H_W = t_v_theta_B_C_2T_H_W[:, :, T:, :, :]

        F_theta_B_C_T_H_W = self.net(
            student_full_x_B_C_2T_H_W.to(**self.tensor_kwargs),
            (student_full_time_B_2T * self.config.rectified_flow_t_scaling_factor).to(**self.tensor_kwargs),
            **condition.to_dict(),
            attn_meta=attn_mask_spec,
        ).float()[:, :, T:, :, :]
        F_theta_B_C_T_H_W_sg = F_theta_B_C_T_H_W.clone().detach()

        warmup_ratio = 1.0 if self.config.tangent_warmup == 0 else min(1.0, iteration / self.config.tangent_warmup)
        g = -(F_theta_B_C_T_H_W_sg - F_teacher_B_C_T_H_W) - warmup_ratio * time_B_1_T_1_1 * t_v_theta_B_C_T_H_W

        with torch.no_grad():
            nan_mask_g = torch.isnan(g).flatten(start_dim=1).any(dim=1).view(B, 1, 1, 1, 1).expand_as(g)
            nan_mask_F = torch.isnan(F_theta_B_C_T_H_W).flatten(start_dim=1).any(dim=1).view(B, 1, 1, 1, 1).expand_as(F_theta_B_C_T_H_W)
        nan_mask = nan_mask_g | nan_mask_F

        g[nan_mask] = 0
        F_theta_B_C_T_H_W = torch.where(nan_mask, torch.tensor(0.0, device=F_theta_B_C_T_H_W.device), F_theta_B_C_T_H_W)
        F_theta_B_C_T_H_W_sg[nan_mask] = 0

        g = g.double() / (g.double().norm(p=2, dim=(1, 2, 3, 4), keepdim=True) + 0.1)

        loss_scm = ((F_theta_B_C_T_H_W - F_theta_B_C_T_H_W_sg - g) ** 2).sum(dim=(1, 2, 3, 4))
        loss = self.config.loss_scale * loss_scm
        return {}, loss

    def training_step_closures(self, data_batch, iteration: int):
        training_type = self.config.training_type

        # Teacher-forcing training
        if training_type == "tf":
            yield training_type, lambda: self._tf_step(data_batch, iteration), True
            return

        # Diffusion-forcing training
        if training_type == "df":
            yield "df", lambda: self._df_step(data_batch, iteration), True
            return

        # Teacher-forcing distillation
        if training_type == "tf_dcm":
            self._require_causal_teacher()
            yield training_type, lambda: self._tf_dcm_step(data_batch, iteration), True
            return

        if training_type == "tf_scm":
            self._require_causal_teacher()
            yield "tf_scm", lambda: self._tf_scm_step(data_batch, iteration), True
            return

        if training_type == "tf_dscm":
            self._require_causal_teacher()
            yield "tf_dcm", lambda: self._tf_dcm_step(data_batch, iteration), False
            yield "tf_scm", lambda: self._tf_scm_step(data_batch, iteration), True
            return

        # Self-forcing and joint TF-SCM + SF-DMD phase.
        assert training_type in {"sf_dmd", "tf_scm_sf_dmd"}, f"Unsupported training_type: {training_type}"
        if self.is_student_phase(iteration):
            emit_tf_scm = self.config.loss_scale > 0 and training_type == "tf_scm_sf_dmd"
            emit_sf_dmd = self.net_fake_score and iteration >= self.config.tangent_warmup and self.config.loss_scale_dmd > 0

            if emit_tf_scm:
                self._require_causal_teacher()
                yield "tf_scm", lambda: self._tf_scm_step(data_batch, iteration), not emit_sf_dmd

            if emit_sf_dmd:
                self._require_bidirectional_teacher()
                if self.config.replayed_training:
                    num_replayed_chunks, ctx = self._make_replayed_sf_ctx(data_batch, iteration)
                    for i in range(num_replayed_chunks):
                        yield f"sf_dmd_replayed_{i}", lambda i=i: self._replayed_sf_step(i, ctx, iteration), i == num_replayed_chunks - 1
                else:
                    yield "sf_dmd", lambda: self._sf_step(data_batch, iteration), True
        else:
            yield "critic", lambda: self.training_step_critic(data_batch, iteration), True

    @torch.no_grad()
    def forward(self, xt, t, condition: TextCondition):
        pass

    # ------------------------ Sampling ------------------------

    @torch.no_grad()
    def _build_shifted_ode_t_steps(self, num_steps: int, device: torch.device) -> torch.Tensor:
        timestep_shift = self.config.sample_timestep_shift
        sigma_max_rf = self.config.sigma_max / (self.config.sigma_max + 1)
        unshifted_sigma_max = sigma_max_rf / (timestep_shift - (timestep_shift - 1) * sigma_max_rf)
        t_steps = torch.linspace(unshifted_sigma_max, 0.0, num_steps + 1, dtype=torch.float64, device=device)
        return timestep_shift * t_steps / (1 + (timestep_shift - 1) * t_steps)

    @torch.no_grad()
    def _build_few_step_t_steps(self, num_steps: int, device: torch.device, mid_t: List[float] | None = None) -> torch.Tensor:
        if mid_t is None:
            mid_t = self.config.backward_timesteps[: num_steps - 1]
        sigma_max_rf = self.config.sigma_max / (self.config.sigma_max + 1)
        return torch.tensor([sigma_max_rf, *mid_t, 0], dtype=torch.float64, device=device)

    def _make_block_pattern(self, x_B_C_T_H_W_size, model=None):
        if model is None:
            model = self.net
        B, C, T, H, W = x_B_C_T_H_W_size
        if self.config.first_chunk_t < 0:
            raise ValueError(f"first_chunk_t must be non-negative, got {self.config.first_chunk_t}")
        if self.config.chunk_t <= 0:
            if self.config.first_chunk_t == 0 and self.config.chunk_t == 0:
                raise ValueError(
                    "Causal chunk pattern is unset: this base experiment is a template with "
                    "first_chunk_t=0 and chunk_t=0. Launch a concrete chunk variant such as "
                    "*_c1-1, *_c3-3, or *_c1-4, or override model.config.first_chunk_t and "
                    "model.config.chunk_t explicitly."
                )
            raise ValueError(f"chunk_t must be positive, got {self.config.chunk_t}")
        if self.config.first_chunk_t > T:
            raise ValueError(f"first_chunk_t={self.config.first_chunk_t} exceeds latent frames T={T}")
        frame_tokens = H * W // model.get_spatial_patch_size()
        if (T - self.config.first_chunk_t) % self.config.chunk_t != 0:
            raise ValueError(f"T={T}, first_chunk_t={self.config.first_chunk_t}, chunk_t={self.config.chunk_t}")
        num_blocks = 1 + (T - self.config.first_chunk_t) // self.config.chunk_t
        block_pattern = BlockPattern(frame_tokens=frame_tokens, first_chunk_frames=self.config.first_chunk_t, chunk_frames=self.config.chunk_t)
        return B, C, T, H, W, frame_tokens, num_blocks, block_pattern

    @staticmethod
    def _block_span(block_pattern: BlockPattern, block_idx: int):
        frame_start = block_pattern.blocks_to_frames(block_idx)
        block_size = block_pattern.block_size(block_idx)
        frame_end = frame_start + block_size
        return frame_start, frame_end, block_size

    def _predict_cfg(
        self,
        model,
        x_B_C_T_H_W: torch.Tensor,
        time_B_T: torch.Tensor,
        condition: TextCondition,
        uncondition: Optional[TextCondition],
        guidance: float,
        inference_state_cond=None,
        inference_state_uncond=None,
        attn_meta=None,
    ) -> torch.Tensor:
        v_cond_B_C_T_H_W = model(
            x_B_C_T_H_W=x_B_C_T_H_W.to(**self.tensor_kwargs),
            timesteps_B_T=(time_B_T * self.config.rectified_flow_t_scaling_factor).to(**self.tensor_kwargs),
            **condition.to_dict(),
            inference_state=inference_state_cond,
            attn_meta=attn_meta,
        ).float()
        if uncondition is None or guidance <= 1.0:
            return v_cond_B_C_T_H_W
        v_uncond_B_C_T_H_W = model(
            x_B_C_T_H_W=x_B_C_T_H_W.to(**self.tensor_kwargs),
            timesteps_B_T=(time_B_T * self.config.rectified_flow_t_scaling_factor).to(**self.tensor_kwargs),
            **uncondition.to_dict(),
            inference_state=inference_state_uncond,
            attn_meta=attn_meta,
        ).float()
        return v_uncond_B_C_T_H_W + guidance * (v_cond_B_C_T_H_W - v_uncond_B_C_T_H_W)

    @torch.no_grad()
    def _causal_rollout_sampling(
        self,
        model,
        init_noise: torch.Tensor,
        t_steps: torch.Tensor,
        t_steps_per_chunk: Optional[list[torch.Tensor]],
        condition: TextCondition,
        generator: torch.Generator,
        uncondition: Optional[TextCondition] = None,
        guidance: float = 1.0,
        ode: bool = False,
        sequential: bool = False,
    ) -> torch.Tensor:
        B = init_noise.shape[0]
        cp_group = self.get_context_parallel_group()
        cp_size = 1 if cp_group is None else cp_group.size()
        if sequential and B > 1 and cp_size == 1:
            return self._causal_rollout_sampling_sequential(
                model, init_noise, t_steps, t_steps_per_chunk, condition, generator, uncondition, guidance, ode
            )
        return self._causal_rollout_sampling_batched(model, init_noise, t_steps, t_steps_per_chunk, condition, generator, uncondition, guidance, ode)

    @torch.no_grad()
    def _causal_rollout_sampling_sequential(
        self,
        model,
        init_noise: torch.Tensor,
        t_steps: torch.Tensor,
        t_steps_per_chunk: Optional[list[torch.Tensor]],
        condition: TextCondition,
        generator: torch.Generator,
        uncondition: Optional[TextCondition] = None,
        guidance: float = 1.0,
        ode: bool = False,
    ) -> torch.Tensor:
        """Sample one-by-one to avoid large KV cache allocation for big batches.
        All noises are pre-sampled with the full batch so results match batched sampling.
        A single set of B=1 KV caches is allocated once and reused between samples."""
        B, C, T, H, W = init_noise.shape
        _, _, _, _, _, frame_tokens, _, _ = self._make_block_pattern(init_noise.size(), model=model)
        if not ode:
            step_noises = []
            max_denoise_steps = max(len(ts) for ts in t_steps_per_chunk) - 1 if t_steps_per_chunk else len(t_steps) - 1
            for _ in range(max_denoise_steps):
                noise_full = torch.randn(B, C, T, H, W, dtype=torch.float32, device=self.tensor_kwargs["device"], generator=generator)
                step_noises.append(self.sync(noise_full))
        else:
            step_noises = None

        use_cfg = uncondition is not None and guidance > 1.0
        kv_cond = model.allocate_kv_caches(max_len=T * frame_tokens)
        kv_uncond = model.allocate_kv_caches(max_len=T * frame_tokens) if use_cfg else None

        results = []
        for b in range(B):
            for c in kv_cond:
                c.reset(free_buffers=False)
            if kv_uncond is not None:
                for c in kv_uncond:
                    c.reset(free_buffers=False)
            noise_b = init_noise[b : b + 1]
            cond_b = TextCondition(crossattn_emb=condition.crossattn_emb[b : b + 1], data_type=condition.data_type)
            uncond_b = None
            if uncondition is not None:
                uncond_b = TextCondition(crossattn_emb=uncondition.crossattn_emb[b : b + 1], data_type=uncondition.data_type)
            step_noises_b = None if step_noises is None else [n[b : b + 1] for n in step_noises]
            sample_b = self._causal_rollout_sampling_batched(
                model,
                noise_b,
                t_steps,
                t_steps_per_chunk,
                cond_b,
                generator=None,
                uncondition=uncond_b,
                guidance=guidance,
                ode=ode,
                step_noises=step_noises_b,
                kv_cond=kv_cond,
                kv_uncond=kv_uncond,
            )
            results.append(sample_b.cpu())
        del kv_cond, kv_uncond
        torch.cuda.empty_cache()
        return torch.cat(results, dim=0).to(device=init_noise.device)

    @torch.no_grad()
    def _causal_rollout_sampling_batched(
        self,
        model,
        init_noise: torch.Tensor,
        t_steps: torch.Tensor,
        t_steps_per_chunk: Optional[list[torch.Tensor]],
        condition: TextCondition,
        generator: Optional[torch.Generator],
        uncondition: Optional[TextCondition] = None,
        guidance: float = 1.0,
        ode: bool = False,
        step_noises: Optional[list] = None,
        kv_cond: Optional[list] = None,
        kv_uncond: Optional[list] = None,
    ) -> torch.Tensor:
        B, C, T, H, W, frame_tokens, num_blocks, block_pattern = self._make_block_pattern(init_noise.size(), model=model)
        ones_B_1 = torch.ones(B, 1, device=init_noise.device, dtype=torch.float64)

        owns_kv = kv_cond is None
        if kv_cond is None:
            kv_cond = model.allocate_kv_caches(max_len=T * frame_tokens)
        if kv_uncond is None and uncondition is not None and guidance > 1.0:
            kv_uncond = model.allocate_kv_caches(max_len=T * frame_tokens)
        if not ode and step_noises is None:
            step_noises = []
            max_denoise_steps = max(len(ts) for ts in t_steps_per_chunk) - 1 if t_steps_per_chunk else len(t_steps) - 1
            for _ in range(max_denoise_steps):
                noise_full = torch.randn(B, C, T, H, W, dtype=torch.float32, device=self.tensor_kwargs["device"], generator=generator)
                step_noises.append(self.sync(noise_full))
        x_blocks = []
        for i in range(num_blocks):
            cache_last = self._use_context_from_last_step(i)
            frame_start, frame_end, block_size = self._block_span(block_pattern, i)
            attn_mask_spec = AttnMaskSpec(mode="block_causal", pattern=block_pattern, q_block_offset=i)
            t_steps_block = t_steps_per_chunk[i] if t_steps_per_chunk else t_steps
            num_denoise_steps = len(t_steps_block) - 1

            x_B_C_block_H_W = init_noise[:, :, frame_start:frame_end].to(torch.float64) * t_steps_block[0]
            for step_idx, (t_cur, t_next) in enumerate(zip(t_steps_block[:-1], t_steps_block[1:])):
                t_cur_B_1 = t_cur * ones_B_1
                is_last = step_idx == num_denoise_steps - 1
                kv_mode = KVCacheMode.APPEND if (cache_last and is_last) else KVCacheMode.READONLY
                inf_cond = CausalInferenceState(mode=kv_mode, kv_caches=kv_cond, pattern=block_pattern, block_cursor=i)
                inf_uncond = None
                if kv_uncond is not None:
                    inf_uncond = CausalInferenceState(mode=kv_mode, kv_caches=kv_uncond, pattern=block_pattern, block_cursor=i)
                v_pred_B_C_block_H_W = self._predict_cfg(
                    model,
                    x_B_C_block_H_W,
                    t_cur_B_1,
                    condition,
                    uncondition,
                    guidance,
                    inference_state_cond=inf_cond,
                    inference_state_uncond=inf_uncond,
                    attn_meta=attn_mask_spec,
                )

                if ode:
                    x_B_C_block_H_W = x_B_C_block_H_W - (t_cur - t_next) * v_pred_B_C_block_H_W.to(torch.float64)
                else:
                    noise_B_C_block_H_W = step_noises[step_idx][:, :, frame_start:frame_end]
                    x_B_C_block_H_W = (1 - t_next) * (x_B_C_block_H_W - t_cur * v_pred_B_C_block_H_W.to(torch.float64)) + t_next * noise_B_C_block_H_W

            if not cache_last:
                zero_B_1 = torch.zeros_like(t_cur_B_1)
                with torch.no_grad():
                    inf_append = CausalInferenceState(mode=KVCacheMode.APPEND, kv_caches=kv_cond, pattern=block_pattern, block_cursor=i)
                    model(
                        x_B_C_block_H_W.to(**self.tensor_kwargs),
                        zero_B_1.to(**self.tensor_kwargs),
                        **condition.to_dict(),
                        inference_state=inf_append,
                        attn_meta=attn_mask_spec,
                    )
                    if kv_uncond is not None:
                        inf_append_uncond = CausalInferenceState(mode=KVCacheMode.APPEND, kv_caches=kv_uncond, pattern=block_pattern, block_cursor=i)
                        model(
                            x_B_C_block_H_W.to(**self.tensor_kwargs),
                            zero_B_1.to(**self.tensor_kwargs),
                            **uncondition.to_dict(),
                            inference_state=inf_append_uncond,
                            attn_meta=attn_mask_spec,
                        )
            x_blocks.append(x_B_C_block_H_W)
        if owns_kv:
            del kv_cond, kv_uncond
            torch.cuda.empty_cache()
        return torch.cat(x_blocks, dim=2)

    def _select_teacher_for_sampling(self):
        if self.config.training_type in ["sf_dmd", "tf_scm_sf_dmd"]:
            return self.net_bidirectional_teacher
        if self.config.training_type in {"tf_dcm", "tf_scm", "tf_dscm"}:
            return self.net_causal_teacher
        return self.net_causal_teacher or self.net_bidirectional_teacher

    @torch.no_grad()
    def generate_samples_from_batch(
        self,
        data_batch: Dict,
        seed: int = 1,
        state_shape: Tuple | None = None,
        n_sample: int | None = None,
        num_steps: int = 4,
        init_noise: torch.Tensor = None,
        mid_t: List[float] | None = None,
    ) -> torch.Tensor:
        input_key = self.config.input_data_key

        if n_sample is None:
            n_sample = data_batch[input_key].shape[0]
        if state_shape is None:
            _T, _H, _W = data_batch[input_key].shape[-3:]
            state_shape = [
                self.config.state_ch,
                self.tokenizer.get_latent_num_frames(_T),
                _H // self.tokenizer.spatial_compression_factor,
                _W // self.tokenizer.spatial_compression_factor,
            ]

        _, _, condition, uncondition = self.get_data_and_condition(data_batch)

        generator = torch.Generator(device=self.tensor_kwargs["device"])
        generator.manual_seed(seed)

        if init_noise is None:
            init_noise = torch.randn(
                n_sample,
                *state_shape,
                dtype=torch.float32,
                device=self.tensor_kwargs["device"],
                generator=generator,
            )
        init_noise, condition, uncondition = self.sync(init_noise, condition, uncondition)

        if self.config.training_type in {"tf", "df"}:
            return torch.zeros(
                n_sample,
                *state_shape,
                dtype=torch.float32,
                device=self.tensor_kwargs["device"],
            )

        t_steps = self._build_few_step_t_steps(num_steps=num_steps, device=init_noise.device, mid_t=mid_t)
        t_steps_per_chunk = None
        if self.config.sf_simulation_steps_per_chunk:
            _, _, _, _, _, _, num_blocks, _ = self._make_block_pattern(init_noise.size(), model=self.net)
            max_step_counts = self._max_simulation_steps_per_chunk(num_blocks)
            max_schedules = self._backward_timestep_schedules_per_chunk(num_blocks)
            t_steps_per_chunk = [
                self._build_few_step_t_steps(num_steps=max_steps, device=init_noise.device, mid_t=max_schedule)
                for max_steps, max_schedule in zip(max_step_counts, max_schedules)
            ]

        x = self._causal_rollout_sampling(
            self.net,
            init_noise=init_noise,
            t_steps=t_steps,
            t_steps_per_chunk=t_steps_per_chunk,
            condition=condition,
            generator=generator,
            sequential=True,
        )
        samples = x.float()
        return torch.nan_to_num(samples)

    @torch.no_grad()
    def generate_samples_from_batch_teacher(
        self,
        data_batch: Dict,
        seed: int = 1,
        state_shape: Tuple | None = None,
        n_sample: int | None = None,
        init_noise: torch.Tensor = None,
        num_steps: int = 50,
        sampler="UniPC",
    ) -> torch.Tensor:
        """
        Generate samples from the batch. Based on given batch, it will automatically determine whether to generate image or video samples.
        Args:
            data_batch (dict): raw data batch draw from the training data loader.
            iteration (int): Current iteration number.
            guidance (float): guidance weights
            seed (int): random seed
            state_shape (tuple): shape of the state, default to data batch if not provided
            n_sample (int): number of samples to generate
            num_steps (int): number of steps for the diffusion process
        """
        teacher_net = self._select_teacher_for_sampling()
        _, _, condition, uncondition = self.get_data_and_condition(data_batch)

        input_key = self.config.input_data_key

        if n_sample is None:
            n_sample = data_batch[input_key].shape[0]
        if state_shape is None:
            _T, _H, _W = data_batch[input_key].shape[-3:]
            state_shape = [
                self.config.state_ch,
                self.tokenizer.get_latent_num_frames(_T),
                _H // self.tokenizer.spatial_compression_factor,
                _W // self.tokenizer.spatial_compression_factor,
            ]

        generator = torch.Generator(device=self.tensor_kwargs["device"])
        generator.manual_seed(seed)

        if init_noise is None:
            init_noise = torch.randn(
                n_sample,
                *state_shape,
                dtype=torch.float32,
                device=self.tensor_kwargs["device"],
                generator=generator,
            )
        init_noise, condition, uncondition = self.sync(init_noise, condition, uncondition)

        if teacher_net is None:
            if self.config.training_type in {"tf", "df"}:
                t_steps = self._build_shifted_ode_t_steps(num_steps=50, device=init_noise.device)
                return self._causal_rollout_sampling(
                    model=self.net,
                    init_noise=init_noise,
                    t_steps=t_steps,
                    t_steps_per_chunk=None,
                    condition=condition,
                    generator=generator,
                    uncondition=uncondition,
                    guidance=self.config.causal_guidance,
                    ode=True,
                    sequential=True,
                )
            else:
                return torch.zeros(
                    n_sample,
                    *state_shape,
                    dtype=torch.float32,
                    device=self.tensor_kwargs["device"],
                )

        if teacher_net is self.net_causal_teacher:
            t_steps = self._build_shifted_ode_t_steps(num_steps=num_steps, device=init_noise.device)
            return self._causal_rollout_sampling(
                model=teacher_net,
                init_noise=init_noise,
                t_steps=t_steps,
                t_steps_per_chunk=None,
                condition=condition,
                generator=generator,
                uncondition=uncondition,
                guidance=self.config.causal_guidance,
                ode=True,
                sequential=True,
            )

        x = init_noise.to(torch.float64)

        timestep_shift = self.config.sample_timestep_shift
        sigma_max = self.config.sigma_max / (self.config.sigma_max + 1)
        unshifted_sigma_max = sigma_max / (timestep_shift - (timestep_shift - 1) * sigma_max)

        samplers = {"Euler": FlowEulerSampler, "UniPC": FlowUniPCMultistepSampler}
        sampler = samplers[sampler](num_train_timesteps=1000, sigma_max=unshifted_sigma_max, sigma_min=0.0)
        sampler.set_timesteps(num_inference_steps=num_steps, device=self.tensor_kwargs["device"], shift=timestep_shift)

        ones = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)
        for _, t in enumerate(sampler.timesteps):
            timesteps = t * ones

            with torch.no_grad():
                v_cond = teacher_net(
                    x_B_C_T_H_W=x.to(**self.tensor_kwargs), timesteps_B_T=timesteps.to(**self.tensor_kwargs), **condition.to_dict()
                ).float()
                v_uncond = teacher_net(
                    x_B_C_T_H_W=x.to(**self.tensor_kwargs), timesteps_B_T=timesteps.to(**self.tensor_kwargs), **uncondition.to_dict()
                ).float()

            v_pred = v_uncond + self.config.bidirectional_guidance * (v_cond - v_uncond)

            x = sampler.step(v_pred, t, x)

        samples = x.float()

        return torch.nan_to_num(samples)

    @torch.no_grad()
    def validation_step(self, data: dict[str, torch.Tensor], iteration: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """
        Current code does nothing.
        """
        pass

    # ------------------------ Distributed Parallel ------------------------

    @staticmethod
    def get_context_parallel_group():
        if parallel_state.is_initialized():
            return parallel_state.get_context_parallel_group()
        return None

    def sync(self, *args):
        cp_group = self.get_context_parallel_group()
        cp_size = 1 if cp_group is None else cp_group.size()
        if cp_size > 1:
            out = tuple(broadcast(arg, cp_group) if isinstance(arg, torch.Tensor) else arg.broadcast(cp_group) for arg in args)
        else:
            out = args
        return out[0] if len(out) == 1 else out

    # ------------------ Data Preprocessing ------------------

    def get_data_and_condition(self, data_batch: dict[str, torch.Tensor]) -> Tuple[Tensor, TextCondition]:
        if IS_PROCESSED_KEY not in data_batch or not data_batch[IS_PROCESSED_KEY]:
            if self.config.input_latent_key in data_batch:
                latents = data_batch[self.config.input_latent_key]
                if latents.dim() == 6:
                    # ODE pair: [B, K, C, T, H, W]
                    assert latents.shape[3] == self.config.state_t
                    data_batch[self.config.input_data_key] = self.decode(latents[:, 0]).contiguous().float().clamp(-1, 1)
                else:
                    assert latents.shape[2] >= self.config.state_t
                    data_batch[self.config.input_latent_key] = latents[:, :, : self.config.state_t, :, :]
                    data_batch[self.config.input_data_key] = self.decode(data_batch[self.config.input_latent_key]).contiguous().float().clamp(-1, 1)

            data_batch[IS_PROCESSED_KEY] = True

        raw_state = data_batch.get(self.config.input_data_key)
        latent_state = data_batch.get(self.config.input_latent_key)
        # Condition
        if self.neg_embed is not None:
            data_batch["neg_t5_text_embeddings"] = repeat(
                self.neg_embed.to(**self.tensor_kwargs), "l d -> b l d", b=data_batch["t5_text_embeddings"].shape[0]
            )
            condition, uncondition = self.conditioner.get_condition_with_negative_prompt(data_batch)
        else:
            condition, uncondition = self.conditioner.get_condition_uncondition(data_batch)
        condition = condition.edit_data_type(DataType.VIDEO)
        uncondition = uncondition.edit_data_type(DataType.VIDEO)
        return raw_state, latent_state, condition, uncondition

    # ------------------ Checkpointing ------------------

    def model_dict(self) -> Dict[str, Any]:
        model_dict = {"net": self.net}
        if self.net_fake_score:
            model_dict["fake_score"] = self.net_fake_score
        return model_dict

    def state_dict(self) -> Dict[str, Any]:
        net_state_dict = self.net.state_dict(prefix="net.")
        if self.config.ema.enabled:
            ema_state_dict = self.net_ema.state_dict(prefix="net_ema.")
            net_state_dict.update(ema_state_dict)
        if self.net_fake_score:
            fake_score_state_dict = self.net_fake_score.state_dict(prefix="net_fake_score.")
            net_state_dict.update(fake_score_state_dict)
        return net_state_dict

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        """
        Loads a state dictionary into the model and optionally its EMA counterpart.
        Different from torch strict=False mode, the method will not raise error for unmatched state shape while raise warning.

        Parameters:e
            state_dict (Mapping[str, Any]): A dictionary containing separate state dictionaries for the model and
                                            potentially for an EMA version of the model under the keys 'model' and 'ema', respectively.
            strict (bool, optional): If True, the method will enforce that the keys in the state dict match exactly
                                    those in the model and EMA model (if applicable). Defaults to True.
            assign (bool, optional): If True and in strict mode, will assign the state dictionary directly rather than
                                    matching keys one-by-one. This is typically used when loading parts of state dicts
                                    or using customized loading procedures. Defaults to False.
        """
        _reg_state_dict = collections.OrderedDict()
        _ema_state_dict = collections.OrderedDict()
        _fake_score_state_dict = collections.OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("net."):
                _reg_state_dict[k.replace("net.", "")] = v
            elif k.startswith("net_ema."):
                _ema_state_dict[k.replace("net_ema.", "")] = v
            elif k.startswith("net_fake_score."):
                _fake_score_state_dict[k.replace("net_fake_score.", "")] = v

        state_dict = _reg_state_dict

        if strict:
            reg_results: _IncompatibleKeys = self.net.load_state_dict(_reg_state_dict, strict=strict, assign=assign)

            if self.config.ema.enabled:
                ema_results: _IncompatibleKeys = self.net_ema.load_state_dict(_ema_state_dict, strict=strict, assign=assign)
            if self.net_fake_score:
                fake_score_results: _IncompatibleKeys = self.net_fake_score.load_state_dict(_fake_score_state_dict, strict=strict, assign=assign)

            return _IncompatibleKeys(
                missing_keys=reg_results.missing_keys
                + (ema_results.missing_keys if self.config.ema.enabled else [])
                + (fake_score_results.missing_keys if self.net_fake_score else []),
                unexpected_keys=reg_results.unexpected_keys
                + (ema_results.unexpected_keys if self.config.ema.enabled else [])
                + (fake_score_results.unexpected_keys if self.net_fake_score else []),
            )
        else:
            log.critical("load model in non-strict mode")
            log.critical(non_strict_load_model(self.net, _reg_state_dict), rank0_only=False)
            if self.config.ema.enabled:
                log.critical("load ema model in non-strict mode")
                log.critical(non_strict_load_model(self.net_ema, _ema_state_dict), rank0_only=False)
            if self.net_fake_score:
                log.critical("load fake score model in non-strict mode")
                log.critical(non_strict_load_model(self.net_fake_score, _fake_score_state_dict), rank0_only=False)

    # ------------------ public methods ------------------
    def ema_beta(self, iteration: int) -> float:
        """
        Calculate the beta value for EMA update.
        weights = weights * beta + (1 - beta) * new_weights

        Args:
            iteration (int): Current iteration number.

        Returns:
            float: The calculated beta value.
        """
        iteration = iteration + self.config.ema.iteration_shift
        if iteration < 1:
            return 0.0
        return (1 - 1 / (iteration + 1)) ** (self.ema_exp_coefficient + 1)

    def model_param_stats(self) -> Dict[str, int]:
        return {"total_learnable_param_num": self._param_count}

    def is_image_batch(self, data_batch: dict[str, Tensor]) -> bool:
        return False

    @torch.no_grad()
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        return self.tokenizer.encode(state) * self.config.sigma_data

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.tokenizer.decode(latent / self.config.sigma_data)

    def get_num_video_latent_frames(self) -> int:
        return self.config.state_t

    @property
    def text_encoder_class(self) -> str:
        return self.config.text_encoder_class

    @contextmanager
    def ema_scope(self, context=None, is_cpu=False):
        if self.config.ema.enabled:
            # https://github.com/pytorch/pytorch/issues/144289
            for module in self.net.modules():
                if isinstance(module, FSDPModule):
                    module.reshard()
            self.net_ema_worker.cache(self.net.parameters(), is_cpu=is_cpu)
            self.net_ema_worker.copy_to(src_model=self.net_ema, tgt_model=self.net)
            if context is not None:
                log.info(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.config.ema.enabled:
                for module in self.net.modules():
                    if isinstance(module, FSDPModule):
                        module.reshard()
                self.net_ema_worker.restore(self.net.parameters())
                if context is not None:
                    log.info(f"{context}: Restored training weights")

    def clip_grad_norm_(
        self,
        max_norm: float,
        norm_type: float = 2.0,
        error_if_nonfinite: bool = False,
        foreach: Optional[bool] = None,
        iteration: int = 0,
    ):
        if not self.config.grad_clip:
            max_norm = 1e12
        if self.is_student_phase(iteration):
            return clip_grad_norm_(
                self.net.parameters(),
                max_norm=max_norm,
                norm_type=norm_type,
                error_if_nonfinite=error_if_nonfinite,
                foreach=foreach,
            ).cpu()
        if self.net_fake_score:
            clip_grad_norm_(
                self.net_fake_score.parameters(),
                max_norm=max_norm,
                norm_type=norm_type,
                error_if_nonfinite=error_if_nonfinite,
                foreach=foreach,
            )
        return None
