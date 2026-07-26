#!/usr/bin/env bash
# Extract frame-level attention for a whole prompt set, sharded round-robin
# across GPUs, in the run_<NNN>/layer<N>.pt layout that jupyter-plot's
# spectral_analysis_utils.compute_head_period_homology_256() globs.
#
#   ./experiments/extract_attn/run_extraction_batch.sh <prompts.txt> <out_root> <gpus> [latent_frames] [capture_step]
#
#   ./experiments/extract_attn/run_extraction_batch.sh \
#       prompts/MovieGenVideoBench_num128.txt cache/attn/mgvb128 5,6,7 72 append
#
# run_<NNN> is the prompt's 0-based line number, so the shard layout does not
# affect which index a prompt lands on -- reruns and partial reruns stay
# consistent.
set -euo pipefail

PROMPTS=${1:?usage: run_extraction_batch.sh <prompts.txt> <out_root> <gpus> [latent_frames] [capture_step]}
OUT_ROOT=${2:?missing out_root}
GPUS=${3:?missing gpus, e.g. 5,6,7}
LATENT_FRAMES=${4:-72}
CAPTURE_STEP=${5:-append}

CKPT=${CKPT:-assets/checkpoints/Causal_rCM_Wan2.1_T2V_1.3B_480p_TF-dCM-init_SF-DMD_c1-1_step4.pt}
SEED=${SEED:-0}
# Latent frames per chunk. Must match how the checkpoint was distilled: the
# c1-1 weights want 1/1, the c3-3 weights want 3/3. Setting 3/3 also makes
# block_sizes [3]*24, matching Self-Forcing's num_frame_per_block=3 so that
# last_block_frame_attention averages the same number of query rows on both
# sides -- otherwise the field has the same name and a different meaning.
FIRST_CHUNK_T=${FIRST_CHUNK_T:-1}
CHUNK_T=${CHUNK_T:-1}

# The spectral consumer slices last_block_frame_attention[0:69].
if [[ "$LATENT_FRAMES" -lt 69 ]]; then
    echo "error: latent_frames=$LATENT_FRAMES < 69; the spectral analysis needs at least 69" >&2
    exit 1
fi
# latent T = 1 + (pixel_frames - 1) / 4
PIXEL_FRAMES=$(( 4 * (LATENT_FRAMES - 1) + 1 ))

if (( (LATENT_FRAMES - FIRST_CHUNK_T) % CHUNK_T != 0 )); then
    echo "error: (latent_frames - first_chunk_t) must be divisible by chunk_t; got ($LATENT_FRAMES - $FIRST_CHUNK_T) % $CHUNK_T" >&2
    exit 1
fi

IFS=',' read -r -a GPU_ARR <<< "$GPUS"
NGPU=${#GPU_ARR[@]}
mapfile -t PROMPT_LINES < "$PROMPTS"
NPROMPT=${#PROMPT_LINES[@]}

mkdir -p "$OUT_ROOT" "$OUT_ROOT/videos" logs
echo "prompts=$NPROMPT  gpus=${GPU_ARR[*]}  latent_frames=$LATENT_FRAMES (=$PIXEL_FRAMES pixel)  capture=$CAPTURE_STEP  chunks=$FIRST_CHUNK_T/$CHUNK_T"
echo "out=$OUT_ROOT/run_000 .. run_$(printf '%03d' $((NPROMPT - 1)))"

shard() {                       # $1 = position in GPU_ARR
    local slot=$1 gpu=${GPU_ARR[$1]} i
    for (( i = slot; i < NPROMPT; i += NGPU )); do
        # Skip a run that already has all 30 layers, so an interrupted sweep resumes.
        if [[ $(ls "$OUT_ROOT/$(printf 'run_%03d' "$i")"/layer*.pt 2>/dev/null | wc -l) -eq 30 ]]; then
            continue
        fi
        CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=. uv run --no-sync python \
            rcm/inference/wan2pt1_t2v_causal_infer.py \
            --distilled --dit_path "$CKPT" \
            --num_steps 4 --mid_t 15/16 5/6 5/8 \
            --first_chunk_t "$FIRST_CHUNK_T" --chunk_t "$CHUNK_T" \
            --num_frames "$PIXEL_FRAMES" --seed "$SEED" \
            --prompt "${PROMPT_LINES[$i]}" \
            --save_path "$OUT_ROOT/videos/$(printf 'run_%03d' "$i").mp4" \
            --extract_attn_layers all \
            --attn_capture_step "$CAPTURE_STEP" \
            --attn_layout runs --attn_run_index "$i" \
            --attn_output_dir "$OUT_ROOT" \
            > "logs/extract_$(printf '%03d' "$i").log" 2>&1 \
            || echo "FAILED run_$(printf '%03d' "$i") on gpu $gpu" >&2
    done
    echo "gpu $gpu (slot $slot) done"
}

for (( s = 0; s < NGPU; s++ )); do shard "$s" & done
wait

DONE=$(find "$OUT_ROOT" -name 'layer*.pt' | wc -l)
echo "complete: $DONE artifacts across $(ls -d "$OUT_ROOT"/run_* 2>/dev/null | wc -l) runs (expected $((NPROMPT * 30)))"
