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

from hydra.core.config_store import ConfigStore

from imaginaire.lazy_config import LazyCall as L
from imaginaire.lazy_config import LazyDict
from rcm.utils.timestep_utils import LogNormal, UniformShift


def build_debug_run(job):
    return dict(
        defaults=[
            f"/experiment/{job['job']['name']}",
            "_self_",
        ],
        job=dict(
            group=job["job"]["group"] + "_debug",
            name=f"{job['job']['name']}" + "_${now:%Y-%m-%d}_${now:%H-%M-%S}",
        ),
        trainer=dict(
            max_iter=25,
            logging_iter=2,
            callbacks=dict(
                every_n_sample_reg=dict(
                    every_n=6,
                    num_samples=5,
                ),
                every_n_sample_ema=dict(
                    every_n=6,
                    num_samples=5,
                ),
            ),
        ),
        checkpoint=dict(
            save_iter=10,
            load_path="",
            load_training_state=False,
            strict_resume=False,
        ),
        model=dict(
            config=dict(
                tangent_warmup=8,
            )
        ),
    )


"""
torchrun --nproc_per_node=4 --master_port=12341 -m scripts.train --config=rcm/configs/registry_distill.py -- experiment="causal_wan2pt1_1pt3B_res480p_t2v_tf_c1-1_debug"

To override timestep distribution configs (e.g., p_G), use:
torchrun ... -m scripts.train --config=... -- \
    experiment=... \
    '++model.config.p_G={_target_:rcm.utils.timestep_utils.UniformShift,shift:3.0}'
"""
CAUSAL_WAN2PT1_1PT3B_RES480P_T2V_TF: LazyDict = LazyDict(
    dict(
        defaults=[
            {"override /trainer": "distill"},
            {"override /data_train": "webdataset"},
            {"override /model": "fsdp_t2v_causal"},
            {"override /net": "wan2pt1_1pt3B_t2v"},
            {"override /conditioner": "text_nodrop"},
            {"override /ckpt_type": "dcp_distill"},
            {"override /optimizer": "fusedadamw"},
            {
                "override /callbacks": [
                    "basic",
                    "dataloading_speed",
                    "wandb",
                    "viz_online_sampling_distill",
                ]
            },
            {"override /checkpoint": "local"},
            {"override /tokenizer": "wan2pt1_tokenizer"},
            "_self_",
        ],
        job=dict(
            group="Causal_rCM_Wan",
            name="causal_wan2pt1_1pt3B_res480p_t2v_tf",
        ),
        optimizer=dict(
            lr=1e-5,
            weight_decay=0.01,
            betas=(0.9, 0.999),
        ),
        model=dict(
            config=dict(
                loss_scale=100.0,
                fsdp_shard_size=4,
                resolution="480p",
                p_G=L(UniformShift)(shift=5.0),
                state_t=21,
                tokenizer=dict(vae_pth="assets/checkpoints/Wan2.1_VAE.pth"),
                text_encoder_path="assets/checkpoints/models_t5_umt5-xxl-enc-bf16.pth",
                student_ckpt="assets/checkpoints/Wan2.1-T2V-1.3B.pth",
                neg_embed_path="assets/checkpoints/umT5_wan_negative_emb.pt",
                training_type="tf",
                loss_weighting="gaussian_bell",
                first_chunk_t=0,
                chunk_t=0,
                net=dict(
                    sac_config=dict(
                        mode="block_wise",
                    ),
                ),
            )
        ),
        checkpoint=dict(
            save_iter=1000,
            load_path="",
            load_training_state=False,
            strict_resume=False,
        ),
        trainer=dict(
            max_iter=100_000,
            logging_iter=50,
            callbacks=dict(
                every_n_sample_reg=dict(
                    every_n=5000,
                    num_samples=5,
                    run_at_start=True,
                ),
                every_n_sample_ema=dict(
                    every_n=5000,
                    num_samples=5,
                ),
            ),
            grad_accum_iter=1,
        ),
        model_parallel=dict(
            context_parallel_size=1,
        ),
        dataloader_train=dict(
            tar_path_pattern="assets/datasets/Wan2.1_14B_480p_16:9_Euler-step100_shift-3.0_cfg-5.0_seed-0_250K/shard*.tar",
            batch_size=1,
        ),
    ),
    flags={"allow_objects": True},
)

CAUSAL_WAN2PT1_14B_RES480P_T2V_TF: LazyDict = LazyDict(
    dict(
        defaults=[
            f"/experiment/{CAUSAL_WAN2PT1_1PT3B_RES480P_T2V_TF['job']['name']}",
            {"override /net": "wan2pt1_14B_t2v"},
            "_self_",
        ],
        job=dict(
            group="Causal_rCM_Wan",
            name="causal_wan2pt1_14B_res480p_t2v_tf",
        ),
        model=dict(
            config=dict(
                fsdp_shard_size=32,
                student_ckpt="assets/checkpoints/Wan2.1-T2V-14B.pth",
                net=dict(
                    sac_config=dict(
                        mode="block_wise",
                    ),
                ),
            )
        ),
        checkpoint=dict(
            save_iter=500,
        ),
        trainer=dict(
            max_iter=100000,
            logging_iter=50,
            callbacks=dict(
                every_n_sample_reg=dict(
                    every_n=2500,
                ),
                every_n_sample_ema=dict(
                    every_n=2500,
                ),
            ),
            grad_accum_iter=1,
        ),
        model_parallel=dict(
            context_parallel_size=8,
        ),
    ),
    flags={"allow_objects": True},
)

CAUSAL_WAN2PT1_1PT3B_RES480P_T2V_DF: LazyDict = LazyDict(
    dict(
        defaults=[
            {"override /trainer": "distill"},
            {"override /data_train": "webdataset"},
            {"override /model": "fsdp_t2v_causal"},
            {"override /net": "wan2pt1_1pt3B_t2v"},
            {"override /conditioner": "text_nodrop"},
            {"override /ckpt_type": "dcp_distill"},
            {"override /optimizer": "fusedadamw"},
            {
                "override /callbacks": [
                    "basic",
                    "dataloading_speed",
                    "wandb",
                    "viz_online_sampling_distill",
                ]
            },
            {"override /checkpoint": "local"},
            {"override /tokenizer": "wan2pt1_tokenizer"},
            "_self_",
        ],
        job=dict(
            group="Causal_rCM_Wan",
            name="causal_wan2pt1_1pt3B_res480p_t2v_df",
        ),
        optimizer=dict(
            lr=1e-5,
            weight_decay=0.01,
            betas=(0.9, 0.999),
        ),
        model=dict(
            config=dict(
                loss_scale=100.0,
                fsdp_shard_size=4,
                resolution="480p",
                p_G=L(UniformShift)(shift=5.0),
                state_t=21,
                tokenizer=dict(vae_pth="assets/checkpoints/Wan2.1_VAE.pth"),
                text_encoder_path="assets/checkpoints/models_t5_umt5-xxl-enc-bf16.pth",
                student_ckpt="assets/checkpoints/Wan2.1-T2V-1.3B.pth",
                neg_embed_path="assets/checkpoints/umT5_wan_negative_emb.pt",
                training_type="df",
                loss_weighting="none",
                df_timestep_strategy="random",
                first_chunk_t=0,
                chunk_t=0,
                net=dict(
                    sac_config=dict(
                        mode="block_wise",
                    ),
                ),
            )
        ),
        checkpoint=dict(
            save_iter=1000,
            load_path="",
            load_training_state=False,
            strict_resume=False,
        ),
        trainer=dict(
            max_iter=100_000,
            logging_iter=50,
            callbacks=dict(
                every_n_sample_reg=dict(
                    every_n=5000,
                    num_samples=5,
                    run_at_start=True,
                ),
                every_n_sample_ema=dict(
                    every_n=5000,
                    num_samples=5,
                ),
            ),
            grad_accum_iter=1,
        ),
        model_parallel=dict(
            context_parallel_size=1,
        ),
        dataloader_train=dict(
            tar_path_pattern="assets/datasets/Wan2.1_14B_480p_16:9_Euler-step100_shift-3.0_cfg-5.0_seed-0_250K/shard*.tar",
            batch_size=1,
        ),
    ),
    flags={"allow_objects": True},
)

CAUSAL_WAN2PT1_14B_RES480P_T2V_DF: LazyDict = LazyDict(
    dict(
        defaults=[
            f"/experiment/{CAUSAL_WAN2PT1_1PT3B_RES480P_T2V_DF['job']['name']}",
            {"override /net": "wan2pt1_14B_t2v"},
            "_self_",
        ],
        job=dict(
            group="Causal_rCM_Wan",
            name="causal_wan2pt1_14B_res480p_t2v_df",
        ),
        model=dict(
            config=dict(
                fsdp_shard_size=32,
                student_ckpt="assets/checkpoints/Wan2.1-T2V-14B.pth",
                net=dict(
                    sac_config=dict(
                        mode="mm_only",
                    ),
                ),
            )
        ),
        checkpoint=dict(
            save_iter=500,
        ),
        trainer=dict(
            max_iter=100000,
            logging_iter=50,
            callbacks=dict(
                every_n_sample_reg=dict(
                    every_n=2500,
                ),
                every_n_sample_ema=dict(
                    every_n=2500,
                ),
            ),
            grad_accum_iter=1,
        ),
        model_parallel=dict(
            context_parallel_size=8,
        ),
    ),
    flags={"allow_objects": True},
)

CAUSAL_WAN2PT1_1PT3B_RES480P_T2V_TF_DCM: LazyDict = LazyDict(
    dict(
        defaults=[
            {"override /trainer": "distill"},
            {"override /data_train": "webdataset"},
            {"override /model": "fsdp_t2v_causal"},
            {"override /net": "wan2pt1_1pt3B_t2v"},
            {"override /net_causal_teacher": "wan2pt1_14B_t2v"},
            {"override /conditioner": "text_nodrop"},
            {"override /ckpt_type": "dcp_distill"},
            {"override /optimizer": "fusedadamw"},
            {
                "override /callbacks": [
                    "basic",
                    "dataloading_speed",
                    "wandb",
                    "viz_online_sampling_distill",
                ]
            },
            {"override /checkpoint": "local"},
            {"override /tokenizer": "wan2pt1_tokenizer"},
            "_self_",
        ],
        job=dict(
            group="Causal_rCM_Wan",
            name="causal_wan2pt1_1pt3B_res480p_t2v_tf_dcm",
        ),
        optimizer=dict(
            lr=2e-6,
            weight_decay=0.01,
            betas=(0.0, 0.999),
        ),
        model=dict(
            config=dict(
                loss_scale=100.0,
                fsdp_shard_size=32,
                resolution="480p",
                state_t=21,
                tokenizer=dict(vae_pth="assets/checkpoints/Wan2.1_VAE.pth"),
                text_encoder_path="assets/checkpoints/models_t5_umt5-xxl-enc-bf16.pth",
                student_ckpt="assets/checkpoints/Wan2.1-T2V-1.3B-causal.pt",
                causal_teacher_ckpt="assets/checkpoints/Wan2.1-T2V-14B-causal.pt",
                neg_embed_path="assets/checkpoints/umT5_wan_negative_emb.pt",
                training_type="tf_dcm",
                first_chunk_t=0,
                chunk_t=0,
                net=dict(
                    sac_config=dict(
                        mode="block_wise",
                    ),
                ),
            )
        ),
        checkpoint=dict(
            save_iter=500,
            load_path="",
            load_training_state=False,
            strict_resume=False,
        ),
        trainer=dict(
            max_iter=100_000,
            logging_iter=50,
            callbacks=dict(
                every_n_sample_reg=dict(
                    every_n=500,
                    num_samples=5,
                    run_at_start=True,
                ),
                every_n_sample_ema=dict(
                    every_n=500,
                    num_samples=5,
                ),
            ),
            grad_accum_iter=1,
        ),
        model_parallel=dict(
            context_parallel_size=4,
        ),
        dataloader_train=dict(
            tar_path_pattern="assets/datasets/Wan2.1_14B_480p_16:9_Euler-step100_shift-3.0_cfg-5.0_seed-0_250K/shard*.tar",
            batch_size=1,
        ),
    ),
    flags={"allow_objects": True},
)

CAUSAL_WAN2PT1_1PT3B_RES480P_T2V_TF_SCM: LazyDict = LazyDict(
    dict(
        defaults=[
            {"override /trainer": "distill"},
            {"override /data_train": "webdataset"},
            {"override /model": "fsdp_t2v_causal"},
            {"override /net": "wan2pt1_1pt3B_t2v_jvp"},
            {"override /net_causal_teacher": "wan2pt1_14B_t2v"},
            {"override /conditioner": "text_nodrop"},
            {"override /ckpt_type": "dcp_distill"},
            {"override /optimizer": "fusedadamw"},
            {
                "override /callbacks": [
                    "basic",
                    "dataloading_speed",
                    "wandb",
                    "viz_online_sampling_distill",
                ]
            },
            {"override /checkpoint": "local"},
            {"override /tokenizer": "wan2pt1_tokenizer"},
            "_self_",
        ],
        job=dict(
            group="Causal_rCM_Wan",
            name="causal_wan2pt1_1pt3B_res480p_t2v_tf_scm",
        ),
        optimizer=dict(
            lr=2e-6,
            weight_decay=0.01,
            betas=(0.0, 0.999),
        ),
        model=dict(
            config=dict(
                loss_scale=100.0,
                fsdp_shard_size=32,
                resolution="480p",
                p_G=L(LogNormal)(p_mean=-0.8, p_std=1.6),
                state_t=21,
                tokenizer=dict(vae_pth="assets/checkpoints/Wan2.1_VAE.pth"),
                text_encoder_path="assets/checkpoints/models_t5_umt5-xxl-enc-bf16.pth",
                student_ckpt="assets/checkpoints/Wan2.1-T2V-1.3B-causal.pt",
                causal_teacher_ckpt="assets/checkpoints/Wan2.1-T2V-14B-causal.pt",
                neg_embed_path="assets/checkpoints/umT5_wan_negative_emb.pt",
                training_type="tf_scm",
                tangent_warmup=1000,
                first_chunk_t=0,
                chunk_t=0,
                net=dict(
                    sac_config=dict(
                        mode="block_wise",
                    ),
                ),
            )
        ),
        checkpoint=dict(
            save_iter=500,
            load_path="",
            load_training_state=False,
            strict_resume=False,
        ),
        trainer=dict(
            max_iter=100_000,
            logging_iter=50,
            callbacks=dict(
                every_n_sample_reg=dict(
                    every_n=500,
                    num_samples=5,
                    run_at_start=True,
                ),
                every_n_sample_ema=dict(
                    every_n=500,
                    num_samples=5,
                ),
            ),
            grad_accum_iter=1,
        ),
        model_parallel=dict(
            context_parallel_size=4,
        ),
        dataloader_train=dict(
            tar_path_pattern="assets/datasets/Wan2.1_14B_480p_16:9_Euler-step100_shift-3.0_cfg-5.0_seed-0_250K/shard*.tar",
            batch_size=1,
        ),
    ),
    flags={"allow_objects": True},
)

CAUSAL_WAN2PT1_1PT3B_RES480P_T2V_SF_DMD: LazyDict = LazyDict(
    dict(
        defaults=[
            {"override /trainer": "distill"},
            {"override /data_train": "webdataset"},
            {"override /model": "fsdp_t2v_causal"},
            {"override /net": "wan2pt1_1pt3B_t2v"},
            {"override /net_bidirectional_teacher": "wan2pt1_14B_t2v"},
            {"override /net_fake_score": "wan2pt1_14B_t2v"},
            {"override /conditioner": "text_nodrop"},
            {"override /ckpt_type": "dcp_distill"},
            {"override /optimizer": "fusedadamw"},
            {
                "override /callbacks": [
                    "basic",
                    "dataloading_speed",
                    "wandb",
                    "viz_online_sampling_distill",
                ]
            },
            {"override /checkpoint": "local"},
            {"override /tokenizer": "wan2pt1_tokenizer"},
            "_self_",
        ],
        job=dict(
            group="Causal_rCM_Wan",
            name="causal_wan2pt1_1pt3B_res480p_t2v_sf_dmd",
        ),
        optimizer=dict(
            lr=2e-6,
            weight_decay=0.01,
            betas=(0.0, 0.999),
        ),
        model=dict(
            config=dict(
                loss_scale=100.0,
                fsdp_shard_size=32,
                resolution="480p",
                p_D=L(UniformShift)(shift=5.0),
                state_t=21,
                tokenizer=dict(vae_pth="assets/checkpoints/Wan2.1_VAE.pth"),
                text_encoder_path="assets/checkpoints/models_t5_umt5-xxl-enc-bf16.pth",
                student_ckpt="assets/checkpoints/Wan2.1-T2V-1.3B-causal-init.pt",
                bidirectional_teacher_ckpt="assets/checkpoints/Wan2.1-T2V-14B.pth",
                neg_embed_path="assets/checkpoints/umT5_wan_negative_emb.pt",
                training_type="sf_dmd",
                max_simulation_steps_fake=4,
                student_update_freq=6,
                optimizer_fake_score=dict(
                    lr=4e-7,
                    weight_decay=0.01,
                    betas=(0.0, 0.999),
                ),
                first_chunk_t=0,
                chunk_t=0,
                net=dict(
                    sac_config=dict(
                        mode="block_wise",
                    ),
                ),
            )
        ),
        checkpoint=dict(
            save_iter=50,
            load_path="",
            load_training_state=False,
            strict_resume=False,
        ),
        trainer=dict(
            max_iter=100_000,
            logging_iter=50,
            callbacks=dict(
                every_n_sample_reg=dict(
                    every_n=50,
                    num_samples=5,
                    run_at_start=True,
                ),
                every_n_sample_ema=dict(
                    every_n=100,
                    num_samples=5,
                ),
            ),
            grad_accum_iter=1,
        ),
        model_parallel=dict(
            context_parallel_size=4,
        ),
        dataloader_train=dict(
            tar_path_pattern="assets/datasets/Wan2.1_14B_480p_16:9_Euler-step100_shift-3.0_cfg-5.0_seed-0_250K/shard*.tar",
            batch_size=1,
        ),
    ),
    flags={"allow_objects": True},
)


CAUSAL_WAN2PT1_1PT3B_RES480P_T2V_SF_DMD_1STEP: LazyDict = LazyDict(
    dict(
        defaults=[
            f"/experiment/{CAUSAL_WAN2PT1_1PT3B_RES480P_T2V_SF_DMD['job']['name']}",
            "_self_",
        ],
        job=dict(
            group="Causal_rCM_Wan",
            name="causal_wan2pt1_1pt3B_res480p_t2v_sf_dmd_1step",
        ),
        model=dict(
            config=dict(
                sf_simulation_steps_per_chunk=[4, 1],
                sf_backward_timestep_schedules=[[15 / 16, 5 / 6, 5 / 8], []],
            )
        ),
    ),
    flags={"allow_objects": True},
)


CAUSAL_WAN2PT1_1PT3B_RES480P_T2V_SF_DMD_2STEP: LazyDict = LazyDict(
    dict(
        defaults=[
            f"/experiment/{CAUSAL_WAN2PT1_1PT3B_RES480P_T2V_SF_DMD['job']['name']}",
            "_self_",
        ],
        job=dict(
            group="Causal_rCM_Wan",
            name="causal_wan2pt1_1pt3B_res480p_t2v_sf_dmd_2step",
        ),
        model=dict(
            config=dict(
                sf_simulation_steps_per_chunk=[4, 2],
                sf_backward_timestep_schedules=[[15 / 16, 5 / 6, 5 / 8], [5 / 6]],
            )
        ),
    ),
    flags={"allow_objects": True},
)


CAUSAL_WAN2PT1_1PT3B_RES480P_T2V_SF_DMD_2STEP_NOISY_CTX: LazyDict = LazyDict(
    dict(
        defaults=[
            f"/experiment/{CAUSAL_WAN2PT1_1PT3B_RES480P_T2V_SF_DMD['job']['name']}",
            "_self_",
        ],
        job=dict(
            group="Causal_rCM_Wan",
            name="causal_wan2pt1_1pt3B_res480p_t2v_sf_dmd_2step_noisy_ctx",
        ),
        model=dict(
            config=dict(
                sf_simulation_steps_per_chunk=[4, 2],
                sf_backward_timestep_schedules=[[15 / 16, 5 / 6, 5 / 8], [5 / 8]],
                context_from_last_step=True,
                context_from_last_step_start_chunk=1,
            )
        ),
    ),
    flags={"allow_objects": True},
)


BLOCK_VARIANTS = [
    {"first_chunk_t": 1, "chunk_t": 1, "suffix": "_c1-1"},
    {"first_chunk_t": 3, "chunk_t": 3, "suffix": "_c3-3"},
    {"first_chunk_t": 1, "chunk_t": 4, "suffix": "_c1-4"},
]


def build_variant(base_job, first_chunk_t, chunk_t, suffix, replayed):
    base_name = base_job["job"]["name"]
    full_suffix = suffix + ("_replayed" if replayed else "")
    variant = LazyDict(
        dict(
            defaults=[
                f"/experiment/{base_name}",
                "_self_",
            ],
            job=dict(
                group=base_job["job"]["group"],
                name=f"{base_name}{full_suffix}",
            ),
            model=dict(
                config=dict(
                    first_chunk_t=first_chunk_t,
                    chunk_t=chunk_t,
                    replayed_training=replayed,
                )
            ),
        ),
        flags={"allow_objects": True},
    )
    return variant


cs = ConfigStore.instance()

job_list = [
    CAUSAL_WAN2PT1_1PT3B_RES480P_T2V_TF,
    CAUSAL_WAN2PT1_14B_RES480P_T2V_TF,
    CAUSAL_WAN2PT1_1PT3B_RES480P_T2V_DF,
    CAUSAL_WAN2PT1_14B_RES480P_T2V_DF,
    CAUSAL_WAN2PT1_1PT3B_RES480P_T2V_TF_DCM,
    CAUSAL_WAN2PT1_1PT3B_RES480P_T2V_TF_SCM,
    CAUSAL_WAN2PT1_1PT3B_RES480P_T2V_SF_DMD,
    CAUSAL_WAN2PT1_1PT3B_RES480P_T2V_SF_DMD_1STEP,
    CAUSAL_WAN2PT1_1PT3B_RES480P_T2V_SF_DMD_2STEP,
    CAUSAL_WAN2PT1_1PT3B_RES480P_T2V_SF_DMD_2STEP_NOISY_CTX,
]

for job in job_list:
    cs.store(group="experiment", package="_global_", name=job["job"]["name"], node=job)
    for v in BLOCK_VARIANTS:
        for replayed in [False, True]:
            variant = build_variant(job, v["first_chunk_t"], v["chunk_t"], v["suffix"], replayed)
            cs.store(group="experiment", package="_global_", name=variant["job"]["name"], node=variant)
            cs.store(group="experiment", package="_global_", name=variant["job"]["name"] + "_debug", node=build_debug_run(variant))
