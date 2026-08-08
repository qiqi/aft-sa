#!/bin/bash
# Run one serial queue of extension cases on one GPU.
#   run_queue.sh <gpu> <case> [<case> ...]
# Logs to scripts/ext2026/logs/<host>-gpu<gpu>.log
set -u
GPU="$1"; shift
D=/home/qiqi/flexcompute/sa-ai/scripts/ext2026
. "$D/host_env.sh" || exit 1

mkdir -p "$D/logs"
LOG="$D/logs/$(hostname -s)-gpu${GPU}.log"
echo "=== QUEUE START gpu=$GPU $(date -u +%FT%TZ) cases: $* ===" >> "$LOG"
for c in "$@"; do
  python3 "$D/run_ext_case.py" "$c" "$GPU" >> "$LOG" 2>&1
  echo "--- queue advanced past $c ($(date -u +%FT%TZ)) ---" >> "$LOG"
done
echo "=== QUEUE DONE gpu=$GPU $(date -u +%FT%TZ) ===" >> "$LOG"
