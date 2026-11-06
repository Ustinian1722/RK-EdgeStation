#!/usr/bin/env python3
"""Summarize RK3576 Worker zero-copy A/B/A experiments from JSONL logs."""

from __future__ import annotations

import argparse
import csv
import html
import json
import statistics
from pathlib import Path
from typing import Any


METRICS = {
    "throughput_fps": "Worker throughput (FPS)",
    "p95_end_to_end_ms": "Decode-to-result P95 (ms)",
    "process_cpu_core_percent": "Worker CPU (one core %)",
    "process_rss_kb": "RSS (MiB)",
}

MODE_ORDER = ("z0", "z1", "z2")
COLORS = {"z0": "#4C78A8", "z1": "#F2A541", "z2": "#54A24B"}


def load_runs(runs_dir: Path) -> tuple[list[dict[str, Any]], str]:
    records: list[dict[str, Any]] = []
    for mode in MODE_ORDER:
        for log_path in sorted(runs_dir.glob(f"{mode}-*/run.jsonl")):
            completed = None
            for line in log_path.read_text(encoding="utf-8").splitlines():
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if item.get("type") == "metric" and item.get("status") == "completed":
                    completed = item
            if completed is None:
                raise ValueError(f"no completed metric found in {log_path}")
            records.append({"mode": mode, "run": log_path.parent.name, **completed})

    environment_file = runs_dir / "environment.txt"
    environment = environment_file.read_text(encoding="utf-8", errors="replace") \
        if environment_file.exists() else ""
    environment_json = runs_dir / "environment.json"
    if environment_json.exists():
        environment = json.dumps(
            json.loads(environment_json.read_text(encoding="utf-8")),
            indent=2, ensure_ascii=False)
    return records, environment


def aggregate(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = []
    for mode in MODE_ORDER:
        selected = [record for record in records if record["mode"] == mode]
        if not selected:
            continue
        metrics: dict[str, float] = {}
        for key in METRICS:
            metrics[key] = float(statistics.median(float(row[key]) for row in selected))
        summary.append({
            "mode": mode,
            "video_memory_mode": selected[0].get("video_memory_mode", ""),
            "rknn_input_mode": selected[0].get("rknn_input_mode", ""),
            "repeats": len(selected),
            **metrics,
            "p50_preprocess_ms": float(statistics.median(float(row["p50_preprocess_ms"]) for row in selected)),
            "p50_rknn_input_set_ms": float(statistics.median(float(row["p50_rknn_input_set_ms"]) for row in selected)),
            "p50_rknn_input_sync_ms": float(statistics.median(float(row["p50_rknn_input_sync_ms"]) for row in selected)),
            "p50_rknn_run_ms": float(statistics.median(float(row["p50_rknn_run_ms"]) for row in selected)),
            "p50_output_fetch_ms": float(statistics.median(float(row["p50_rknn_output_fetch_ms"]) for row in selected)),
            "p50_queue_wait_ms": float(statistics.median(float(row["p50_queue_wait_ms"]) for row in selected)),
            "p95_queue_wait_ms": float(statistics.median(float(row["p95_queue_wait_ms"]) for row in selected)),
            "processed_frames_median": float(statistics.median(float(row["processed_frames"]) for row in selected)),
            "decoded_frames_median": float(statistics.median(float(row["decoded_frames"]) for row in selected)),
            "dropped_frames_median": float(statistics.median(float(row["dropped_frames"]) for row in selected)),
            "source_copy_bytes_total_median": float(statistics.median(float(row["source_copy_bytes_total"]) for row in selected)),
            "detections_median": float(statistics.median(float(row["detections"]) for row in selected)),
        })
    return summary


def render_svg(summary: list[dict[str, Any]], destination: Path) -> None:
    width, height = 1000, 650
    panels = [
        ("throughput_fps", "Effective Worker throughput", "FPS"),
        ("p95_end_to_end_ms", "Decode-to-result latency P95", "ms"),
        ("process_cpu_core_percent", "Worker CPU load", "% of one core"),
        ("process_rss_kb", "Resident memory", "MiB"),
    ]
    panel_w, panel_h = 455, 270
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#263238}.title{font-size:20px;font-weight:bold}.axis{font-size:12px}.value{font-size:13px;font-weight:bold}</style>',
        '<text x="30" y="32" class="title">RK3576 dynamic H.264 Worker A/B comparison (median of three runs)</text>',
    ]
    positions = [(25, 55), (520, 55), (25, 355), (520, 355)]
    for (metric, title, unit), (origin_x, origin_y) in zip(panels, positions):
        values = [float(row[metric]) / 1024 if metric == "process_rss_kb"
                  else float(row[metric]) for row in summary]
        maximum = max(values, default=1.0) or 1.0
        plot_x, plot_y = origin_x + 55, origin_y + 45
        plot_h, plot_w = 165, 355
        svg.append(f'<text x="{origin_x + 5}" y="{origin_y + 22}" class="title" font-size="16">{html.escape(title)} ({html.escape(unit)})</text>')
        svg.append(f'<line x1="{plot_x}" y1="{plot_y + plot_h}" x2="{plot_x + plot_w}" y2="{plot_y + plot_h}" stroke="#455a64"/>')
        for tick in range(4):
            y = plot_y + plot_h - (plot_h * tick / 3)
            label = maximum * tick / 3
            svg.append(f'<line x1="{plot_x}" y1="{y:.1f}" x2="{plot_x + plot_w}" y2="{y:.1f}" stroke="#e0e0e0"/>')
            svg.append(f'<text x="{plot_x - 8}" y="{y + 4:.1f}" text-anchor="end" class="axis">{label:.1f}</text>')
        for index, row in enumerate(summary):
            mode = row["mode"]
            value = (float(row[metric]) / 1024
                     if metric == "process_rss_kb" else float(row[metric]))
            bar_w, gap = 65, 43
            x = plot_x + 42 + index * (bar_w + gap)
            bar_h = value / maximum * plot_h
            y = plot_y + plot_h - bar_h
            svg.append(f'<rect x="{x}" y="{y:.1f}" width="{bar_w}" height="{bar_h:.1f}" fill="{COLORS[mode]}"/>')
            svg.append(f'<text x="{x + bar_w / 2}" y="{y - 6:.1f}" text-anchor="middle" class="value">{value:.2f}</text>')
            svg.append(f'<text x="{x + bar_w / 2}" y="{plot_y + plot_h + 20}" text-anchor="middle" class="axis">{mode.upper()}</text>')
    svg.append('<text x="30" y="640" class="axis">Z0: MPP → CPU NV12 copy → RGA → rknn_inputs_set; Z1: copied NV12 → RGA writes shared RKNN input; Z2: MPP DMA-BUF → RGA → shared RKNN input.</text>')
    svg.append("</svg>")
    destination.write_text("\n".join(svg) + "\n", encoding="utf-8")


def summarize(runs_dir: Path, output_dir: Path) -> list[dict[str, Any]]:
    records, environment = load_runs(runs_dir)
    summary = aggregate(records)
    if not summary:
        raise ValueError(f"no benchmark run folders found in {runs_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "runs_directory": str(runs_dir),
        "repeats_per_mode": {mode: sum(row["mode"] == mode for row in records) for mode in MODE_ORDER},
        "environment_snapshot": environment,
        "modes": summary,
        "raw_runs": records,
    }
    (output_dir / "zero_copy_ablation_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (output_dir / "zero_copy_ablation_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    render_svg(summary, output_dir / "zero_copy_ablation.svg")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    for row in summarize(args.runs_dir, args.output_dir):
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
