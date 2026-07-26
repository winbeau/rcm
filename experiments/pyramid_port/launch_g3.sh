#!/usr/bin/env bash
# Detached launcher for the G3 generation sweep.
#
# Everything here is a workaround for one fact: a non-interactive ssh does not
# load the profile. Without HF_HOME the umT5 tokenizer can fall through to
# `HuggingfaceTokenizer("google/umt5-xxl")` and attempt the Hub. Requiring the
# caller to name its existing cache avoids baking a host-specific path into the
# repository; offline mode turns a missing cache into a fast error instead of a
# silent network wait.
#
# Detached via setsid so a tmux session rebind cannot take it down.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"

: "${HF_HOME:?export HF_HOME to the existing Hugging Face cache before launching}"
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
command -v uv >/dev/null || { echo "error: uv is not on PATH" >&2; exit 1; }

PROMPTS=${PROMPTS:-prompts/MovieGenVideoBench_num128.txt}
OUT_ROOT=${OUT_ROOT:-output/g3-30s}
GPUS=${GPUS:-5,6,7}
LATENT=${LATENT:-121}
LOG=${LOG:-logs_g3_30s.txt}

mkdir -p logs
setsid nohup ./experiments/pyramid_port/run_g3_quality_batch.sh \
    "$PROMPTS" "$OUT_ROOT" "$GPUS" "$LATENT" \
    > "$LOG" 2>&1 < /dev/null &
PID=$!
disown || true
sleep 2
echo "launched pid=$PID log=$LOG"
# SID == PID and tty '?' is the check that it actually detached.
ps -o pid,sid,tty,etime -p "$PID" 2>/dev/null | tail -2
