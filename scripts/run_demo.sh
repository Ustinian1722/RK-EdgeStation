#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 MODEL.rknn INPUT [STATE_DIR] [PORT] [latest|fifo] [--loop-video] [--video-backend opencv|gstreamer-mpp] [--preprocess-backend cpu|rga] [--input-pacing media|none]" >&2
  exit 2
fi

model_path=$1
video_path=$2
shift 2
state_dir=state
port=8080
queue_policy=latest
video_backend=opencv
preprocess_backend=cpu
input_pacing=media
loop_video=false

# Preserve the original positional form: STATE_DIR PORT QUEUE_POLICY [--loop-video].
if [[ $# -gt 0 && "$1" != --* ]]; then state_dir=$1; shift; fi
if [[ $# -gt 0 && "$1" != --* ]]; then port=$1; shift; fi
if [[ $# -gt 0 && "$1" != --* ]]; then queue_policy=$1; shift; fi
if [[ $# -gt 0 && "$1" == "--loop-video" ]]; then loop_video=true; shift; fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --loop-video)
      loop_video=true
      shift
      ;;
    --state-dir)
      [[ $# -ge 2 ]] || { echo "--state-dir requires a value" >&2; exit 2; }
      state_dir=$2
      shift 2
      ;;
    --port)
      [[ $# -ge 2 ]] || { echo "--port requires a value" >&2; exit 2; }
      port=$2
      shift 2
      ;;
    --queue-policy)
      [[ $# -ge 2 ]] || { echo "--queue-policy requires a value" >&2; exit 2; }
      queue_policy=$2
      shift 2
      ;;
    --video-backend)
      [[ $# -ge 2 ]] || { echo "--video-backend requires a value" >&2; exit 2; }
      video_backend=$2
      shift 2
      ;;
    --preprocess-backend)
      [[ $# -ge 2 ]] || { echo "--preprocess-backend requires a value" >&2; exit 2; }
      preprocess_backend=$2
      shift 2
      ;;
    --input-pacing)
      [[ $# -ge 2 ]] || { echo "--input-pacing requires a value" >&2; exit 2; }
      input_pacing=$2
      shift 2
      ;;
    *)
      echo "unknown option: $1" >&2
      exit 2
      ;;
  esac
done

if [[ "$queue_policy" != latest && "$queue_policy" != fifo ]]; then
  echo "queue policy must be latest or fifo" >&2
  exit 2
fi
if [[ "$video_backend" != opencv && "$video_backend" != gstreamer-mpp ]]; then
  echo "video backend must be opencv or gstreamer-mpp" >&2
  exit 2
fi
if [[ "$preprocess_backend" != cpu && "$preprocess_backend" != rga ]]; then
  echo "preprocessing backend must be cpu or rga" >&2
  exit 2
fi
if [[ "$input_pacing" != media && "$input_pacing" != none ]]; then
  echo "input pacing must be media or none" >&2
  exit 2
fi

extra_args=()
if [[ "$loop_video" == true ]]; then
  extra_args+=(--loop-video)
fi

exec env PYTHONPATH=python python3 -m edge_station.server \
  --worker build/bin/rk_edge_worker \
  --model "$model_path" \
  --video "$video_path" \
  --state-dir "$state_dir" \
  --port "$port" \
  --queue-policy "$queue_policy" \
  --video-backend "$video_backend" \
  --preprocess-backend "$preprocess_backend" \
  --input-pacing "$input_pacing" \
  "${extra_args[@]}"
