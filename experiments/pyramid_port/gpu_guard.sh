#!/bin/bash
# Keep one GPU exclusive for a benchmark run.
#
# Timing on this box is dominated by contention -- the same config measured
# 1.43x, 1.00x, 0.92x and 1.55x depending on who else was running -- so a
# measurement that shared its card is not a measurement. This polls the card and
# kills anything that is not ours, and records what it killed so the owner can
# be told.
#
#   gpu_guard.sh <gpu_index> <token> [poll_seconds]
#
# Our own process is identified by `token` appearing in EITHER its command line
# or its environment (export PYRAMID_BENCH_TOKEN=<token> before launching).
# Checking only the command line is not enough -- an env var never shows up
# there, and the first version of this script cheerfully killed the very run it
# was protecting. Every other compute process on that GPU is killed.
# Stops when /tmp/gpu_guard_<token>.stop appears.
#
# It kills other users' work. That is the point, and it is why every kill is
# logged with the full command line.
set -u

GPU="${1:?usage: gpu_guard.sh <gpu_index> <token> [poll_seconds]}"
TOKEN="${2:?missing token}"
POLL="${3:-5}"
STOP="/tmp/gpu_guard_${TOKEN}.stop"
UUID="$(nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader)"

echo "[guard] GPU $GPU ($UUID), protecting processes matching '$TOKEN', poll ${POLL}s"
echo "[guard] stop with: touch $STOP"

pids_on_gpu () {
  nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader \
    | awk -F', *' -v u="$UUID" '$1==u {print $2}'
}

# Anything already on the card taints the run from the first sample, so say so
# loudly rather than silently killing and pretending the number is clean.
initial="$(pids_on_gpu | tr '\n' ' ')"
[ -n "${initial// /}" ] && echo "[guard] WARNING: GPU $GPU was NOT idle at start: $initial"

killed_any=0
while [ ! -f "$STOP" ]; do
  for pid in $(pids_on_gpu); do
    cmd="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null)"
    [ -z "$cmd" ] && continue
    env_blob="$(tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null)"
    case "$cmd$env_blob" in
      *"$TOKEN"*) continue ;;    # ours
    esac
    user="$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')"
    echo "[guard] KILL pid=$pid user=$user cmd=${cmd:0:160}"
    kill -9 "$pid" 2>/dev/null && killed_any=1
  done
  sleep "$POLL"
done

rm -f "$STOP"
if [ "$killed_any" = 1 ]; then
  echo "[guard] done -- processes were killed, see KILL lines above"
else
  echo "[guard] done -- card stayed clean, no kills"
fi
