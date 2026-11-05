#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 MODEL.rknn INPUT.mp4 OUTPUT_DIR [WORKER=build/bin/rk_edge_worker]"
}

if [[ $# -eq 1 && ( $1 == "--help" || $1 == "-h" ) ]]; then
  usage
  exit 0
fi
if [[ $# -lt 3 || $# -gt 4 ]]; then
  usage >&2
  exit 2
fi

model_path=$1
video_path=$2
output_dir=$3
worker=${4:-build/bin/rk_edge_worker}

[[ -r "$model_path" ]] || { echo "model is not readable: $model_path" >&2; exit 2; }
[[ -r "$video_path" ]] || { echo "video is not readable: $video_path" >&2; exit 2; }
[[ -x "$worker" ]] || { echo "worker is not executable: $worker" >&2; exit 2; }
command -v timeout >/dev/null || { echo "timeout command is required" >&2; exit 2; }
[[ ! -e "$output_dir" ]] || { echo "output directory already exists: $output_dir" >&2; exit 2; }

mkdir -p "$output_dir"
head -c 1024 "$video_path" > "$output_dir/truncated.mp4"
: > "$output_dir/empty.mp4"
dd if=/dev/zero of="$output_dir/invalid.rknn" bs=64 count=1 status=none

run_case() {
  local name=$1
  local candidate_model=$2
  local candidate_video=$3
  local expected=$4
  local run_dir="$output_dir/$name"
  mkdir -p "$run_dir/state"
  {
    echo "case=$name timeout_seconds=30"
    echo "model_sha256=$(sha256sum "$candidate_model" | awk '{print $1}')"
    echo "video_sha256=$(sha256sum "$candidate_video" | awk '{print $1}')"
    echo "model_bytes=$(stat -c %s "$candidate_model")"
    echo "video_bytes=$(stat -c %s "$candidate_video")"
    echo "start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "$run_dir/metadata.txt"

  set +e
  timeout 30s "$worker" --model "$candidate_model" --video "$candidate_video" \
    --state-dir "$run_dir/state" --queue-capacity 1 --queue-policy latest \
    > "$run_dir/stdout.jsonl" 2> "$run_dir/stderr.txt"
  local status=$?
  set -e
  {
    echo "exit_status=$status"
    echo "end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } >> "$run_dir/metadata.txt"

  if [[ $status -eq 0 || $status -eq 124 ]]; then
    echo "unexpected exit status $status for $name" >&2
    return 1
  fi
  case "$expected" in
    video-error)
      grep -Fq '"status":"error"' "$run_dir/stdout.jsonl" || {
        echo "$name did not emit an error metric" >&2; return 1;
      }
      if grep -Fq '"type":"event"' "$run_dir/stdout.jsonl"; then
        echo "$name emitted an event for invalid media" >&2; return 1
      fi
      ;;
    model-error)
      grep -Fq 'rknn_init failed' "$run_dir/stderr.txt" || {
        echo "$name did not report RKNN initialization failure" >&2; return 1;
      }
      ;;
  esac
  echo "case=$name exit_status=$status result=expected-failure"
}

run_case truncated_video "$model_path" "$output_dir/truncated.mp4" video-error
run_case empty_video "$model_path" "$output_dir/empty.mp4" video-error
run_case invalid_model "$output_dir/invalid.rknn" "$video_path" model-error
