#!/usr/bin/env bash
# G3: generate both arms over a whole prompt set, in the layout t2v-eval's
# `discover_videos()` walks -- <out_root>/<group>/video_NNN.mp4 plus a
# prompts.csv whose zero-padded `index` column pairs with the filename.
#
#   ./experiments/pyramid_port/run_g3_quality_batch.sh <prompts.txt> <out_root> <gpus> [latent_frames]
#
#   ./experiments/pyramid_port/run_g3_quality_batch.sh \
#       prompts/MovieGenVideoBench_num128.txt output/g3-30s 5,6,7 121
#
# Unlike the G2 timing runs this does NOT hold the cards exclusively. G2 needed
# `gpu_guard.sh` because wall clock on this box swings 1.43x/1.00x/0.92x/1.55x
# with host load; G3 measures pixels, and the output is deterministic
# (md5-identical across three repeats per arm), so a shared card changes how
# long it takes and not what comes out.
#
# Both arms are interleaved on the same shard rather than run as two passes, so
# an interrupted sweep leaves the two groups at the same prompt index instead of
# one complete and one empty -- a partial result stays a usable paired
# comparison.
set -euo pipefail

PROMPTS=${1:?usage: run_g3_quality_batch.sh <prompts.txt> <out_root> <gpus> [latent_frames]}
OUT_ROOT=${2:?missing out_root}
GPUS=${3:?missing gpus, e.g. 5,6,7}
LATENT_FRAMES=${4:-121}

CKPT=${CKPT:-assets/checkpoints/Causal_rCM_Wan2.1_T2V_1.3B_480p_TF-dCM-init_SF-DMD_c1-1_step4.pt}
LABELS=${LABELS:-assets/rcm-head-labels-thp6.4-ths0.8.csv}
SEED=${SEED:-0}

# latent T = 1 + (pixel_frames - 1) / 4, so the 30s tier is 121 latent = 481
# pixel. Passing the latent count straight to --num_frames yields a quarter of
# the intended clip; that mistake already cost one 120s measurement.
PIXEL_FRAMES=$(( 4 * (LATENT_FRAMES - 1) + 1 ))

[[ -f "$LABELS" ]] || { echo "error: label CSV not found: $LABELS" >&2; exit 1; }

BASE_GROUP=rcm-baseline
PYR_GROUP=rcm-pyramid

IFS=',' read -r -a GPU_ARR <<< "$GPUS"
NGPU=${#GPU_ARR[@]}
mapfile -t PROMPT_LINES < "$PROMPTS"
NPROMPT=${#PROMPT_LINES[@]}

mkdir -p "$OUT_ROOT/$BASE_GROUP" "$OUT_ROOT/$PYR_GROUP" logs/g3

# The eval side joins on (group, video_id) and reads the prompt from each
# group's own prompts.csv, so both groups get their own identical copy.
python3 - "$PROMPTS" "$OUT_ROOT/$BASE_GROUP/prompts.csv" "$OUT_ROOT/$PYR_GROUP/prompts.csv" <<'PY'
import csv, sys
src, *dests = sys.argv[1:]
rows = [l.rstrip("\n") for l in open(src, encoding="utf-8") if l.strip()]
for d in dests:
    with open(d, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["index", "prompt"])
        # Zero-padded to pair with video_%03d.mp4; discover_videos() compares
        # the index as a string, so "0" would not match stem "video_000".
        w.writerows([f"{i:03d}", p] for i, p in enumerate(rows))
print(f"wrote {len(rows)} prompts to {len(dests)} group dirs")
PY

echo "prompts=$NPROMPT gpus=${GPU_ARR[*]} latent=$LATENT_FRAMES (=$PIXEL_FRAMES pixel) seed=$SEED"
echo "out=$OUT_ROOT/{$BASE_GROUP,$PYR_GROUP}/video_000.mp4 .. video_$(printf '%03d' $((NPROMPT - 1))).mp4"

COMMON=(--distilled --dit_path "$CKPT" --num_steps 4 --mid_t 15/16 5/6 5/8
        --first_chunk_t 1 --chunk_t 1 --num_frames "$PIXEL_FRAMES" --seed "$SEED")

gen () {                        # $1=gpu $2=index $3=group  (rest: extra args)
    local gpu=$1 i=$2 group=$3; shift 3
    local vid; vid=$(printf 'video_%03d' "$i")
    local out="$OUT_ROOT/$group/$vid.mp4"
    local tmp="$OUT_ROOT/$group/${vid}.part.mp4"
    # Resume only from an atomically published file. Encoding directly to
    # `$out` can leave a non-empty but truncated MP4 after SIGKILL; the next run
    # would then skip it as "done". A completed encode is renamed into place.
    [[ -s "$out" ]] && return 0
    rm -f "$tmp"
    if CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=. uv run --no-sync python \
        rcm/inference/wan2pt1_t2v_causal_infer.py \
        "${COMMON[@]}" "$@" \
        --prompt "${PROMPT_LINES[$i]}" --save_path "$tmp" \
        > "logs/g3/${group}_$(printf '%03d' "$i").log" 2>&1; then
        mv "$tmp" "$out"
    else
        echo "FAILED $group/$vid on gpu $gpu" >&2
        rm -f "$tmp"
    fi
}

shard () {
    local slot=$1 gpu=${GPU_ARR[$1]} i
    for (( i = slot; i < NPROMPT; i += NGPU )); do
        gen "$gpu" "$i" "$PYR_GROUP" --pyramid_kv_labels "$LABELS"
        gen "$gpu" "$i" "$BASE_GROUP"
    done
    echo "gpu $gpu (slot $slot) done"
}

for (( s = 0; s < NGPU; s++ )); do shard "$s" & done
wait

for g in "$BASE_GROUP" "$PYR_GROUP"; do
    n=$(find "$OUT_ROOT/$g" -name 'video_*.mp4' -size +0 | wc -l)
    echo "$g: $n/$NPROMPT"
done
echo "=== G3 GENERATION DONE ==="
