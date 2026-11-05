#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 6 ]]; then
  echo "Usage: $0 MODEL.rknn INPUT.mp4 [STATE_DIR] [PORT] [latest|fifo] [--loop-video]" >&2
  exit 2
fi

model_path=$1
video_path=$2
state_dir=${3:-state}
port=${4:-8080}
queue_policy=${5:-latest}
loop_flag=${6:-}

if [[ -n "$loop_flag" && "$loop_flag" != "--loop-video" ]]; then
  echo "the optional sixth argument must be --loop-video" >&2
  exit 2
fi

extra_args=()
if [[ "$loop_flag" == "--loop-video" ]]; then
  extra_args+=(--loop-video)
fi

exec env PYTHONPATH=python python3 -m edge_station.server \
  --worker build/bin/rk_edge_worker \
  --model "$model_path" \
  --video "$video_path" \
  --state-dir "$state_dir" \
  --port "$port" \
  --queue-policy "$queue_policy" \
  "${extra_args[@]}"
