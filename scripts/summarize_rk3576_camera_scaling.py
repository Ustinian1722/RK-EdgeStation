#!/usr/bin/env python3
"""Summarize aligned RK3576 camera-scale resource-monitor JSONL runs."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


CONFIGS = (1, 2, 4)


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def read_run(root: Path, count: int) -> dict[str, Any]:
    directory = root / "camera-scale-{}".format(count)
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    samples = [json.loads(line) for line in
               (directory / "monitor.jsonl").read_text(encoding="utf-8").splitlines() if line]
    return {"summary": summary, "samples": samples}


def write_csv(runs: dict[int, dict[str, Any]], output: Path) -> None:
    fields = [
        "workers", "elapsed_s", "system_cpu_percent", "system_memory_used_mb",
        "total_worker_rss_mb", "npu_frequency_mhz", "npu_utilization_percent",
        "npu_temperature_c", "soc_temperature_c", "cpu0_percent", "cpu1_percent",
        "cpu2_percent", "cpu3_percent", "cpu4_percent", "cpu5_percent",
        "cpu6_percent", "cpu7_percent",
    ]
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for count in CONFIGS:
            for record in runs[count]["samples"]:
                cpu = record.get("cpu_utilization_percent", {})
                workers = record.get("workers", {})
                worker_rss = [entry.get("rss_kb") for entry in workers.values()
                              if entry.get("rss_kb") is not None]
                temps = record.get("temperatures_millicelsius", {})
                npu = record.get("npu", {})
                row = {
                    "workers": count,
                    "elapsed_s": record.get("elapsed_s"),
                    "system_cpu_percent": cpu.get("cpu"),
                    "system_memory_used_mb": (
                        record.get("system_memory", {}).get("MemUsed_kB", 0) / 1024.0
                        if record.get("system_memory", {}).get("MemUsed_kB") is not None else None
                    ),
                    "total_worker_rss_mb": sum(worker_rss) / 1024.0 if worker_rss else None,
                    "npu_frequency_mhz": (
                        npu.get("cur_freq_hz", 0) / 1000000.0
                        if npu.get("cur_freq_hz") is not None else None
                    ),
                    "npu_utilization_percent": npu.get("utilization_percent"),
                    "npu_temperature_c": (
                        temps.get("npu-thermal") / 1000.0
                        if temps.get("npu-thermal") is not None else None
                    ),
                    "soc_temperature_c": (
                        temps.get("soc-thermal") / 1000.0
                        if temps.get("soc-thermal") is not None else None
                    ),
                }
                for core in range(8):
                    row["cpu{}_percent".format(core)] = cpu.get("cpu{}".format(core))
                writer.writerow(row)


def write_summary(root: Path, runs: dict[int, dict[str, Any]], output: Path) -> None:
    result = []
    for count in CONFIGS:
        run = runs[count]
        summary = run["summary"]
        samples = run["samples"]
        system_cpu = [record.get("cpu_utilization_percent", {}).get("cpu") for record in samples]
        system_cpu = [float(value) for value in system_cpu if value is not None]
        used_memory = [record.get("system_memory", {}).get("MemUsed_kB") for record in samples]
        used_memory = [float(value) / 1024.0 for value in used_memory if value is not None]
        rss_total = []
        for record in samples:
            values = [item.get("rss_kb") for item in record.get("workers", {}).values()]
            values = [value for value in values if value is not None]
            if values:
                rss_total.append(sum(values) / 1024.0)
        npu_frequency = [record.get("npu", {}).get("cur_freq_hz") for record in samples]
        npu_frequency = [float(value) / 1000000.0 for value in npu_frequency if value is not None]
        npu_temperature = [record.get("temperatures_millicelsius", {}).get("npu-thermal")
                           for record in samples]
        npu_temperature = [float(value) / 1000.0 for value in npu_temperature if value is not None]
        soc_temperature = [record.get("temperatures_millicelsius", {}).get("soc-thermal")
                           for record in samples]
        soc_temperature = [float(value) / 1000.0 for value in soc_temperature if value is not None]
        worker_summaries = summary["worker_summaries"]
        result.append({
            "workers": count,
            "elapsed_s": summary.get("elapsed_s"),
            "total_fps": summary.get("total_fps"),
            "total_inference_calls": summary.get("total_inference_calls"),
            "worker_fps": [worker.get("fps") for worker in worker_summaries],
            "worker_drop_rate_percent": [worker.get("drop_rate_percent") for worker in worker_summaries],
            "rknn_run_p95_ms_max_worker": max(
                (worker.get("p95_rknn_run_ms") or 0 for worker in worker_summaries), default=0),
            "end_to_end_p95_ms_max_worker": max(
                (worker.get("p95_end_to_end_ms") or 0 for worker in worker_summaries), default=0),
            "system_cpu_percent_mean": mean(system_cpu),
            "system_cpu_percent_max": max(system_cpu) if system_cpu else None,
            "system_memory_used_mb_mean": mean(used_memory),
            "system_memory_used_mb_max": max(used_memory) if used_memory else None,
            "total_worker_rss_mb_max": max(rss_total) if rss_total else None,
            "npu_utilization": summary.get("npu_utilization"),
            "npu_frequency_mhz_min": min(npu_frequency) if npu_frequency else None,
            "npu_frequency_mhz_max": max(npu_frequency) if npu_frequency else None,
            "npu_temperature_c_max": max(npu_temperature) if npu_temperature else None,
            "soc_temperature_c_max": max(soc_temperature) if soc_temperature else None,
            "samples": len(samples),
        })
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_svg(runs: dict[int, dict[str, Any]], output: Path) -> None:
    width, height, left, right = 1000, 680, 72, 25
    panel_top, panel_height, gap = 38, 130, 28
    colors = {1: "#2463a6", 2: "#188358", 4: "#bf4b39"}
    panels = [
        ("System CPU busy (%)", lambda r: r.get("cpu_utilization_percent", {}).get("cpu"), 0, 100),
        ("NPU devfreq current (MHz; utilization unavailable)",
         lambda r: (r.get("npu", {}).get("cur_freq_hz") or 0) / 1000000.0, 0, 1100),
        ("NPU temperature (°C)", lambda r: (r.get("temperatures_millicelsius", {}).get("npu-thermal") or 0) / 1000.0, 20, 50),
        ("System memory used (MiB)",
         lambda r: (r.get("system_memory", {}).get("MemUsed_kB") or 0) / 1024.0, 600, 1300),
    ]
    body = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">'.format(width, height, width, height),
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#222} .small{font-size:12px} .title{font-size:14px;font-weight:bold}</style>',
        '<text x="{}" y="20" class="title">RK3576 aligned telemetry; one-second samples, NPU busy counter unavailable</text>'.format(left),
    ]
    for panel_index, panel in enumerate(panels):
        label, getter, ymin, ymax = panel
        top = panel_top + panel_index * (panel_height + gap)
        bottom = top + panel_height
        body.append('<text x="{}" y="{}" class="title">{}</text>'.format(left, top - 8, label))
        body.append('<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="#555"/>'.format(left, bottom, width - right, bottom))
        body.append('<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="#bbb"/>'.format(left, top, left, bottom))
        for tick in range(3):
            y = top + panel_height * tick / 2.0
            value = ymax - (ymax - ymin) * tick / 2.0
            body.append('<line x1="{}" y1="{:.1f}" x2="{}" y2="{:.1f}" stroke="#ddd"/>'.format(left, y, width - right, y))
            body.append('<text x="8" y="{:.1f}" class="small">{:.0f}</text>'.format(y + 4, value))
        for count in CONFIGS:
            samples = runs[count]["samples"]
            points = []
            for record in samples:
                value = getter(record)
                if value is None:
                    continue
                elapsed = float(record.get("elapsed_s", 0.0))
                x = left + (width - right - left) * min(1.0, elapsed / 36.0)
                y = bottom - panel_height * (float(value) - ymin) / (ymax - ymin)
                points.append('{:.1f},{:.1f}'.format(x, y))
            if points:
                body.append('<polyline fill="none" stroke="{}" stroke-width="2" points="{}"/>'.format(colors[count], ' '.join(points)))
        body.append('<text x="{}" y="{}" class="small">0</text>'.format(left, bottom + 16))
        body.append('<text x="{}" y="{}" class="small">36 s</text>'.format(width - right - 30, bottom + 16))
        body.append('<text x="{}" y="{}" class="small">elapsed from run start</text>'.format(width / 2 - 65, bottom + 16))
    legend_y = height - 12
    for index, count in enumerate(CONFIGS):
        x = left + index * 100
        body.append('<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="3"/>'.format(x, legend_y - 4, x + 20, legend_y - 4, colors[count]))
        body.append('<text x="{}" y="{}" class="small">{} worker{}</text>'.format(x + 25, legend_y, count, "" if count == 1 else "s"))
    body.append('</svg>')
    output.write_text('\n'.join(body) + '\n', encoding='utf-8')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("evidence/constructelligence-ppe-scaling/board"))
    parser.add_argument("--output-dir", type=Path, default=Path("evidence/constructelligence-ppe-scaling/board"))
    args = parser.parse_args()
    runs = {count: read_run(args.root, count) for count in CONFIGS}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(runs, args.output_dir / "camera-scale-monitor.csv")
    write_summary(args.root, runs, args.output_dir / "camera-scale-summary.json")
    write_svg(runs, args.output_dir / "resource-curves.svg")
    print("Wrote camera-scale-monitor.csv, camera-scale-summary.json and resource-curves.svg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
