#!/usr/bin/env python3
"""Aggregate board-side 1/2/4/8/16/32 RTSP benchmark artifacts.

Reads only evidence/multistream-32-20261109/raw/* created by
run_multistream_benchmark.py, writes CSV/JSON and a dependency-free SVG. The
confounded-prior directory is intentionally excluded.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import statistics
from pathlib import Path
from typing import Any


COLORS = {
    1: "#2563eb",
    2: "#059669",
    4: "#d97706",
    8: "#7c3aed",
    16: "#db2777",
    32: "#0891b2",
}


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def sample_sd(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) > 1 else (0.0 if values else None)


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def load_worker_history(path: Path) -> dict[str, Any]:
    connected: set[str] = set()
    status_counts: dict[str, int] = {}
    for name in ("running", "completed", "error"):
        status_counts[name] = 0
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("type") != "camera_metric":
                continue
            status = str(item.get("status", "unknown"))
            status_counts[status] = status_counts.get(status, 0) + 1
            if status == "running" and item.get("source_status") == "connected":
                connected.add(str(item.get("camera_id", "")))
    connected.discard("")
    return {"ever_connected_camera_count": len(connected), "camera_metric_status_records": status_counts}


def load_runs(raw_dir: Path) -> list[dict[str, Any]]:
    result = []
    for run_dir in sorted(path for path in raw_dir.iterdir() if path.is_dir()):
        summary_path = run_dir / "summary.json"
        telemetry_path = run_dir / "telemetry.jsonl"
        worker_path = run_dir / "worker.jsonl.gz"
        if not (summary_path.is_file() and telemetry_path.is_file() and worker_path.is_file()):
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        samples = [json.loads(line) for line in telemetry_path.read_text(encoding="utf-8").splitlines()
                   if line.strip()]
        history = load_worker_history(worker_path)
        cameras = list(summary.get("camera_metrics", {}).values())
        per_camera = {
            # Align each stream to the benchmark's common wall-clock interval.
            # Camera-local Worker FPS includes small connection-start offsets.
            "fps": [int(item.get("processed_frames", 0)) / max(float(summary["duration_s"]), 1e-6)
                    for item in cameras],
            "rknn_run_p50_ms": [float(item.get("rknn_run_p50_ms", 0.0)) for item in cameras
                                 if item.get("rknn_run_p50_ms") is not None],
            "rknn_run_p95_ms": [float(item.get("rknn_run_p95_ms", 0.0)) for item in cameras
                                 if item.get("rknn_run_p95_ms") is not None],
            "end_to_end_p50_ms": [float(item.get("end_to_end_p50_ms", 0.0)) for item in cameras
                                   if item.get("end_to_end_p50_ms") is not None],
            "end_to_end_p95_ms": [float(item.get("end_to_end_p95_ms", 0.0)) for item in cameras
                                   if item.get("end_to_end_p95_ms") is not None],
            "inference_calls": [int(item.get("inference_calls", 0)) for item in cameras],
        }
        cpu_values = [float(value) for name, value in summary.get("system_cpu_percent_mean", {}).items()
                      if name != "cpu" and value is not None]
        temp_values = [int(value) for value in summary.get("temperature_max_milli_c", {}).values()
                       if value is not None]
        avail = [int(sample.get("memory", {}).get("MemAvailable_kB")) for sample in samples
                 if sample.get("memory", {}).get("MemAvailable_kB") is not None]
        result.append({
            "run_id": run_dir.name,
            "camera_count": int(summary["camera_count"]),
            "duration_s": float(summary["duration_s"]),
            "aggregate_processed_fps": float(summary["aggregate_processed_fps"]),
            "aggregate_decoded_fps": float(summary["aggregate_decoded_fps"]),
            "processed_frames": int(summary["processed_frames"]),
            "decoded_frames": int(summary["input_frames_decoded"]),
            "dropped_frames": int(summary["dropped_frames"]),
            "drop_rate_percent": float(summary["drop_rate_percent"]),
            "inference_calls": int(summary["inference_calls"]),
            "worker_cpu_core_percent_mean": summary.get("worker_cpu_core_percent_mean"),
            "system_cpu_percent_mean": mean(cpu_values),
            "worker_rss_kb_peak": summary.get("worker_rss_kb_peak"),
            "system_mem_available_kb_min": min(avail) if avail else None,
            "max_temperature_milli_c": max(temp_values) if temp_values else None,
            "npu_frequency_hz_mean": mean([float(sample["frequencies"]["npu_hz"])
                                           for sample in samples
                                           if sample.get("frequencies", {}).get("npu_hz") is not None]),
            "cooling_state_nonzero_observed": bool(summary.get("cooling_state_nonzero_observed", False)),
            "per_camera_fps_median": statistics.median(per_camera["fps"]) if per_camera["fps"] else None,
            "per_camera_fps_min": min(per_camera["fps"]) if per_camera["fps"] else None,
            "per_camera_fps_max": max(per_camera["fps"]) if per_camera["fps"] else None,
            "per_camera_rknn_p50_median_ms": statistics.median(per_camera["rknn_run_p50_ms"]) if per_camera["rknn_run_p50_ms"] else None,
            "per_camera_rknn_p95_median_ms": statistics.median(per_camera["rknn_run_p95_ms"]) if per_camera["rknn_run_p95_ms"] else None,
            "per_camera_e2e_p50_median_ms": statistics.median(per_camera["end_to_end_p50_ms"]) if per_camera["end_to_end_p50_ms"] else None,
            "per_camera_e2e_p95_median_ms": statistics.median(per_camera["end_to_end_p95_ms"]) if per_camera["end_to_end_p95_ms"] else None,
            "inference_calls_per_camera_min": min(per_camera["inference_calls"]) if per_camera["inference_calls"] else 0,
            "inference_calls_per_camera_median": statistics.median(per_camera["inference_calls"]) if per_camera["inference_calls"] else 0,
            "inference_calls_per_camera_max": max(per_camera["inference_calls"]) if per_camera["inference_calls"] else 0,
            "ever_connected_camera_count": history["ever_connected_camera_count"],
            "camera_metric_status_records": history["camera_metric_status_records"],
            "npu_utilization": str(summary.get("npu_utilization", "unavailable")),
            "npu_load_raw_samples": [sample.get("npu", {}).get("raw_load") for sample in samples],
            "worker_sha256": summary.get("worker_sha256"),
            "model_sha256": summary.get("model_sha256"),
            "camera_config_sha256": summary.get("camera_config_sha256"),
            "telemetry": samples,
        })
    return result


def aggregate(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for run in runs:
        grouped.setdefault(run["camera_count"], []).append(run)
    rows = []
    for camera_count in sorted(grouped):
        group = grouped[camera_count]
        def values(key: str) -> list[float]:
            return [float(run[key]) for run in group if run.get(key) is not None]
        rows.append({
            "cameras": camera_count,
            "repeats": len(group),
            "duration_s_mean": mean(values("duration_s")),
            "connected_cameras_min": min(int(r["ever_connected_camera_count"]) for r in group),
            "aggregate_decode_fps_mean": mean(values("aggregate_decoded_fps")),
            "aggregate_processed_fps_mean": mean(values("aggregate_processed_fps")),
            "aggregate_processed_fps_sd": sample_sd(values("aggregate_processed_fps")),
            "per_camera_fps_median_mean": mean(values("per_camera_fps_median")),
            "per_camera_fps_min_mean": mean(values("per_camera_fps_min")),
            "per_camera_fps_max_mean": mean(values("per_camera_fps_max")),
            "inference_calls_total_mean": mean(values("inference_calls")),
            "drop_rate_percent_mean": mean(values("drop_rate_percent")),
            "rknn_run_p50_ms_median_per_camera_mean": mean(values("per_camera_rknn_p50_median_ms")),
            "rknn_run_p95_ms_median_per_camera_mean": mean(values("per_camera_rknn_p95_median_ms")),
            "end_to_end_p50_ms_median_per_camera_mean": mean(values("per_camera_e2e_p50_median_ms")),
            "end_to_end_p95_ms_median_per_camera_mean": mean(values("per_camera_e2e_p95_median_ms")),
            "inference_calls_per_camera_min": min(int(r["inference_calls_per_camera_min"]) for r in group),
            "inference_calls_per_camera_median_mean": mean(values("inference_calls_per_camera_median")),
            "inference_calls_per_camera_max": max(int(r["inference_calls_per_camera_max"]) for r in group),
            "worker_cpu_core_percent_mean": mean(values("worker_cpu_core_percent_mean")),
            "system_cpu_percent_mean_across_cores": mean(values("system_cpu_percent_mean")),
            "worker_rss_peak_mib_mean": mean([float(r["worker_rss_kb_peak"]) / 1024 for r in group
                                              if r.get("worker_rss_kb_peak") is not None]),
            "worker_rss_peak_mib_max": max([float(r["worker_rss_kb_peak"]) / 1024 for r in group
                                             if r.get("worker_rss_kb_peak") is not None], default=None),
            "system_mem_available_mib_min": min([float(r["system_mem_available_kb_min"]) / 1024 for r in group
                                                  if r.get("system_mem_available_kb_min") is not None], default=None),
            "soc_temperature_c_max": max([float(r["max_temperature_milli_c"]) / 1000 for r in group
                                            if r.get("max_temperature_milli_c") is not None], default=None),
            "npu_frequency_mhz_mean": mean([float(r["npu_frequency_hz_mean"]) / 1e6 for r in group
                                              if r.get("npu_frequency_hz_mean") is not None]),
            "npu_utilization": "unavailable",
            "cooling_state_nonzero_runs": sum(bool(r["cooling_state_nonzero_observed"]) for r in group),
        })
    return rows


def avg_samples(runs: list[dict[str, Any]], metric_getter) -> list[float | None]:
    columns: list[list[float]] = []
    max_count = max((len(run["telemetry"]) for run in runs), default=0)
    for index in range(max_count):
        values = []
        for run in runs:
            if index >= len(run["telemetry"]):
                continue
            value = metric_getter(run["telemetry"][index])
            if value is not None:
                values.append(float(value))
        columns.append(values)
    return [mean(values) for values in columns]


def svg_chart(runs: list[dict[str, Any]], output: Path) -> None:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for run in runs:
        grouped.setdefault(run["camera_count"], []).append(run)
    width, panel_height, top = 1120, 172, 72
    panels = [
        ("Worker CPU (one-core equivalent, %)", 100.0,
         lambda s: s.get("worker_cpu_core_percent")),
        ("System CPU (mean across cores, %)", 50.0,
         lambda s: statistics.fmean([v for k, v in s.get("cpu_percent", {}).items()
                                     if k != "cpu" and v is not None])
         if any(k != "cpu" and v is not None for k, v in s.get("cpu_percent", {}).items()) else None),
        ("Worker peak RSS sample (MiB)", 200.0,
         lambda s: float(s["worker_rss_kb"]) / 1024 if s.get("worker_rss_kb") is not None else None),
        ("Maximum SoC thermal-zone temperature (°C)", 60.0,
         lambda s: max([float(v) / 1000 for v in s.get("temperatures_milli_c", {}).values()
                        if v is not None], default=None)),
        ("NPU current frequency (MHz; utilization unavailable)", 1200.0,
         lambda s: float(s["frequencies"]["npu_hz"]) / 1e6
         if s.get("frequencies", {}).get("npu_hz") is not None else None),
    ]
    height = top + panel_height * len(panels) + 42
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#1f2937}.small{font-size:12px}.title{font-size:22px;font-weight:700}.axis{font-size:11px;fill:#4b5563}</style>',
        '<text x="30" y="34" class="title">RK3576 single-Worker RTSP scaling telemetry</text>',
        '<text x="30" y="55" class="small">Mean of three 12-second runs per camera count; board-side one-second samples</text>',
    ]
    plot_left, plot_right = 90, width - 55
    plot_width = plot_right - plot_left
    max_elapsed = max((float(sample.get("elapsed_s", 0)) for run in runs for sample in run["telemetry"]), default=12.0)
    max_elapsed = max(max_elapsed, 12.0)
    for panel_index, (title, y_max, getter) in enumerate(panels):
        y_top = top + panel_index * panel_height
        y_bottom = y_top + panel_height - 30
        plot_h = y_bottom - y_top - 24
        parts.append(f'<text x="30" y="{y_top + 13}" class="small" font-weight="700">{title}</text>')
        for tick in (0, y_max / 2, y_max):
            y = y_bottom - (tick / y_max) * plot_h
            parts.append(f'<line x1="{plot_left}" y1="{y:.1f}" x2="{plot_right}" y2="{y:.1f}" stroke="#e5e7eb"/>')
            parts.append(f'<text x="{plot_left - 10}" y="{y + 4:.1f}" text-anchor="end" class="axis">{tick:g}</text>')
        parts.append(f'<line x1="{plot_left}" y1="{y_top + 25}" x2="{plot_left}" y2="{y_bottom}" stroke="#6b7280"/>')
        parts.append(f'<line x1="{plot_left}" y1="{y_bottom}" x2="{plot_right}" y2="{y_bottom}" stroke="#6b7280"/>')
        for tick in (0, 3, 6, 9, 12):
            x = plot_left + (tick / max_elapsed) * plot_width
            parts.append(f'<line x1="{x:.1f}" y1="{y_bottom}" x2="{x:.1f}" y2="{y_bottom + 4}" stroke="#6b7280"/>')
            parts.append(f'<text x="{x:.1f}" y="{y_bottom + 17}" text-anchor="middle" class="axis">{tick}</text>')
        legend_x = plot_right - 250
        for offset, n in enumerate(sorted(grouped)):
            series = avg_samples(grouped[n], getter)
            coordinates = []
            for index, value in enumerate(series):
                if value is None:
                    continue
                elapsed_values = [float(run["telemetry"][index]["elapsed_s"])
                                  for run in grouped[n] if index < len(run["telemetry"])]
                elapsed = statistics.fmean(elapsed_values) if elapsed_values else float(index + 1)
                x = plot_left + (elapsed / max_elapsed) * plot_width
                y = y_bottom - (min(max(value, 0.0), y_max) / y_max) * plot_h
                coordinates.append(f"{x:.1f},{y:.1f}")
            if len(coordinates) > 1:
                color = COLORS[n]
                parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{" ".join(coordinates)}"/>')
            color = COLORS[n]
            lx = legend_x + (offset % 3) * 80
            ly = y_top + 14 + (offset // 3) * 15
            parts.append(f'<line x1="{lx}" y1="{ly - 4}" x2="{lx + 15}" y2="{ly - 4}" stroke="{color}" stroke-width="3"/>')
            parts.append(f'<text x="{lx + 19}" y="{ly}" class="axis">{n} cam</text>')
    parts.append(f'<text x="{(plot_left + plot_right) / 2:.1f}" y="{height - 8}" text-anchor="middle" class="axis">Seconds from Worker start</text>')
    parts.append('</svg>')
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, default=Path("evidence/multistream-32-20261109"))
    args = parser.parse_args()
    evidence = args.evidence_dir
    runs = load_runs(evidence / "raw")
    if len(runs) != 18:
        raise SystemExit(f"expected 18 clean runs, found {len(runs)} in {evidence / 'raw'}")
    expected = {1, 2, 4, 8, 16, 32}
    if {run["camera_count"] for run in runs} != expected:
        raise SystemExit("clean run set is missing one or more camera-count conditions")
    rows = aggregate(runs)
    fields = list(rows[0])
    with (evidence / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    report_json = {
        "schema_version": 1,
        "measurement": {
            "device": "KICKPI K7 / RK3576",
            "model_family": "Constructelligence PPE v0.3 FP RKNN",
            "source_kind": "distinct 320x240@5fps synthetic moving-ball H.264 RTSP mounts",
            "worker_mode": "one C++ Worker process, one RKNN context, one capture thread and bounded Latest queue per camera",
            "repeats_per_condition": 3,
            "run_seconds": 12,
            "sample_period_seconds": 1,
            "npu_utilization": "unavailable",
            "p95_note": "Per-camera P95 is summarized as the median of emitted per-camera P95 values; at 16/32 sources each camera has only 3/1-3 inference calls per short run, so these tails are descriptive, not stable percentile estimates.",
            "confounded_prior_excluded": "confounded-prior/ was produced while an unintended API Worker remained active; it is preserved for audit but excluded from all clean aggregates.",
        },
        "conditions": rows,
        "runs": [{key: value for key, value in run.items() if key != "telemetry"}
                 for run in runs],
    }
    (evidence / "summary.json").write_text(json.dumps(report_json, indent=2) + "\n", encoding="utf-8")
    svg_chart(runs, evidence / "resource-curves.svg")
    print(json.dumps({"clean_runs": len(runs), "conditions": rows,
                      "outputs": ["summary.csv", "summary.json", "resource-curves.svg"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
