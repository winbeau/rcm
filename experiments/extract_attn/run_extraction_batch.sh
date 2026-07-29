#!/usr/bin/env bash
# Extract frame-level attention for a whole prompt set, sharded round-robin
# across GPUs, in the run_<NNN>/layer<N>.pt layout that jupyter-plot's
# spectral_analysis_utils.compute_head_period_homology_256() globs.
#
#   ./experiments/extract_attn/run_extraction_batch.sh <prompts.txt> <out_root> <gpus> [latent_frames] [capture_step]
#
#   ./experiments/extract_attn/run_extraction_batch.sh \
#       prompts/MovieGenVideoBench_num128.txt cache/attn/mgvb128 5,6,7 72 denoise0
#
# run_<NNN> is the prompt's 0-based line number, so the shard layout does not
# affect which index a prompt lands on -- reruns and partial reruns stay
# consistent.
set -euo pipefail

PROMPTS=${1:?usage: run_extraction_batch.sh <prompts.txt> <out_root> <gpus> [latent_frames] [capture_step]}
OUT_ROOT=${2:?missing out_root}
GPUS=${3:?missing gpus, e.g. 5,6,7}
LATENT_FRAMES=${4:-72}
CAPTURE_STEP=${5:-denoise0}

CKPT=${CKPT:-assets/checkpoints/Causal_rCM_Wan2.1_T2V_1.3B_480p_TF-dCM-init_SF-DMD_c1-1_step4.pt}
SEED=${SEED:-0}
UV=${UV:-uv}
DIT_SHA256=${RCM_DIT_SHA256:-}
if [[ -z "$DIT_SHA256" ]]; then
    if [[ ! -f "$CKPT" ]]; then
        echo "error: checkpoint directory requires RCM_DIT_SHA256" >&2
        exit 1
    fi
    DIT_SHA256=$(sha256sum "$CKPT" | cut -d' ' -f1)
fi
if [[ ! "$DIT_SHA256" =~ ^[0-9a-fA-F]{64}$ ]]; then
    echo "error: invalid checkpoint SHA-256: $DIT_SHA256" >&2
    exit 1
fi
DIT_SHA256=${DIT_SHA256,,}
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

LOG_ROOT=${LOG_ROOT:-$OUT_ROOT/logs}
STATUS_ROOT=$OUT_ROOT/status
PARTIAL_ROOT=${PARTIAL_ROOT:-$OUT_ROOT/partial}
mkdir -p "$OUT_ROOT" "$OUT_ROOT/videos" "$LOG_ROOT" "$STATUS_ROOT"
echo "prompts=$NPROMPT  gpus=${GPU_ARR[*]}  latent_frames=$LATENT_FRAMES (=$PIXEL_FRAMES pixel)  capture=$CAPTURE_STEP  chunks=$FIRST_CHUNK_T/$CHUNK_T"
echo "checkpoint_sha256=$DIT_SHA256"
echo "out=$OUT_ROOT/run_000 .. run_$(printf '%03d' $((NPROMPT - 1)))"

is_complete() {
    local manifest=$1
    [[ -f "$manifest" ]] || return 1
    PYTHONPATH=. "$UV" run --no-sync python -m rcm.utils.attention_artifact \
        "$manifest" >/dev/null 2>&1
}

quarantine_partial() {
    local run_dir=$1 path has_artifact=0 target base suffix
    for path in "$run_dir"/layer*.pt "$run_dir"/run.json; do
        if [[ -e "$path" ]]; then
            has_artifact=1
            break
        fi
    done
    if (( ! has_artifact )); then
        return 0
    fi

    mkdir -p "$PARTIAL_ROOT"
    base="$PARTIAL_ROOT/$(basename "$run_dir")"
    target="$base"
    suffix=1
    while [[ -e "$target" ]]; do
        target="${base}.${suffix}"
        suffix=$((suffix + 1))
    done
    mv "$run_dir" "$target"
    echo "QUARANTINED partial attention run: $run_dir -> $target" >&2
}

write_status() {
    local index=$1 status=$2 gpu=$3 log_path=$4
    PYTHONPATH=. "$UV" run --no-sync python - \
        "$STATUS_ROOT/$(printf 'run_%03d.json' "$index")" \
        "$index" "$status" "$gpu" "$log_path" "$CAPTURE_STEP" \
        "$DIT_SHA256" <<'PY'
import json
import sys
from pathlib import Path

path, index, status, gpu, log_path, capture_step, checkpoint_sha = sys.argv[1:]
payload = {
    "run_index": int(index),
    "status": status,
    "gpu": gpu,
    "log_path": log_path,
    "capture_pass": capture_step,
    "checkpoint_sha256": checkpoint_sha,
}
target = Path(path)
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

shard() {                       # $1 = position in GPU_ARR
    local slot=$1 gpu=${GPU_ARR[$1]} i run_name run_dir manifest log_path
    for (( i = slot; i < NPROMPT; i += NGPU )); do
        run_name=$(printf 'run_%03d' "$i")
        run_dir="$OUT_ROOT/$run_name"
        manifest="$run_dir/run.json"
        log_path="$LOG_ROOT/extract_$(printf '%03d' "$i").log"
        # Completion is defined by the versioned manifest and verified artifact
        # hashes, not by a model-specific hard-coded layer count.
        if is_complete "$manifest"; then
            write_status "$i" complete "$gpu" "$log_path"
            continue
        fi
        quarantine_partial "$run_dir"
        if CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=. "$UV" run --no-sync python \
            rcm/inference/wan2pt1_t2v_causal_infer.py \
            --distilled --dit_path "$CKPT" --dit_sha256 "$DIT_SHA256" \
            --num_steps 4 --mid_t 15/16 5/6 5/8 \
            --first_chunk_t "$FIRST_CHUNK_T" --chunk_t "$CHUNK_T" \
            --num_frames "$PIXEL_FRAMES" --seed "$SEED" \
            --prompt "${PROMPT_LINES[$i]}" \
            --save_path "$OUT_ROOT/videos/$run_name.mp4" \
            --extract_attn_layers all \
            --attn_capture_step "$CAPTURE_STEP" \
            --attn_layout runs --attn_run_index "$i" \
            --attn_output_dir "$OUT_ROOT" \
            > "$log_path" 2>&1; then
            if is_complete "$manifest"; then
                write_status "$i" complete "$gpu" "$log_path"
            else
                write_status "$i" incomplete "$gpu" "$log_path"
                echo "INCOMPLETE $run_name on gpu $gpu" >&2
            fi
        else
            write_status "$i" failed "$gpu" "$log_path"
            echo "FAILED $run_name on gpu $gpu" >&2
        fi
    done
    echo "gpu $gpu (slot $slot) done"
}

for (( s = 0; s < NGPU; s++ )); do shard "$s" & done
wait

VALID=0
for (( i = 0; i < NPROMPT; i++ )); do
    if is_complete "$OUT_ROOT/$(printf 'run_%03d' "$i")/run.json"; then
        ((VALID += 1))
    fi
done
echo "complete: $VALID/$NPROMPT validated runs"
if (( VALID != NPROMPT )); then
    exit 1
fi
