#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 MODEL.rknn INPUT.mp4 OUTPUT_DIR [REPETITIONS=3] [QUEUE_CAPACITY=1] [CONFIDENCE=0.25]"
}

if [[ $# -eq 1 && ( $1 == "--help" || $1 == "-h" ) ]]; then
  usage
  exit 0
fi
if [[ $# -lt 3 || $# -gt 6 ]]; then
  usage >&2
  exit 2
fi

model_path=$1
video_path=$2
output_dir=$3
repetitions=${4:-3}
queue_capacity=${5:-1}
confidence=${6:-0.25}
worker=${RK_EDGE_WORKER:-build/bin/rk_edge_worker}

[[ -r "$model_path" ]] || { echo "model is not readable: $model_path" >&2; exit 2; }
[[ -r "$video_path" ]] || { echo "video is not readable: $video_path" >&2; exit 2; }
[[ -x "$worker" ]] || { echo "worker is not executable: $worker" >&2; exit 2; }
[[ "$repetitions" =~ ^[1-9][0-9]*$ ]] || { echo "repetitions must be a positive integer" >&2; exit 2; }
[[ "$queue_capacity" =~ ^[1-9][0-9]*$ ]] || { echo "queue capacity must be a positive integer" >&2; exit 2; }
[[ ! -e "$output_dir" ]] || { echo "output directory already exists: $output_dir" >&2; exit 2; }
mkdir -p "$output_dir"

record_temperatures() {
  for zone in /sys/class/thermal/thermal_zone*/temp; do
    [[ -r "$zone" ]] && echo "${zone##*/}=$(cat "$zone")"
  done
}

for ((rep = 1; rep <= repetitions; ++rep)); do
  for policy in latest fifo; do
    run="${policy}-${rep}"
    run_dir="$output_dir/$run"
    mkdir -p "$run_dir/state"
    {
      echo "run=$run policy=$policy queue_capacity=$queue_capacity"
      echo "start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      sha256sum "$video_path" | awk '{print "video_sha256=" $1}'
      sha256sum "$model_path" | awk '{print "model_sha256=" $1}'
      record_temperatures
    } > "$run_dir/metadata.txt"

    start_ns=$(date +%s%N)
    set +e
    "$worker" --model "$model_path" --video "$video_path" \
      --state-dir "$run_dir/state" --queue-capacity "$queue_capacity" \
      --queue-policy "$policy" --confidence "$confidence" \
      > "$run_dir/stdout.jsonl" 2> "$run_dir/stderr.txt"
    status=$?
    set -e
    end_ns=$(date +%s%N)
    {
      echo "exit_status=$status"
      echo "end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      awk -v start="$start_ns" -v end="$end_ns" \
        'BEGIN {printf "wall_seconds=%.3f\n", (end - start) / 1e9}'
      for zone in /sys/class/thermal/thermal_zone*/temp; do
        [[ -r "$zone" ]] && echo "${zone##*/}_end=$(cat "$zone")"
      done
    } >> "$run_dir/metadata.txt"
    echo "completed=$run exit_status=$status"
    if ((status != 0)); then
      exit "$status"
    fi
  done
done
