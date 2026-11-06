#!/usr/bin/env python3
"""Plot board-side offline and media-clock summaries from evidence/flagship."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence" / "flagship"


def load_summary(experiment: str) -> dict[str, dict[str, object]]:
    path = EVIDENCE / experiment / "summary.json"
    return {item["config_id"]: item for item in json.loads(path.read_text(encoding="utf-8"))}


def metric(summary: dict[str, object], name: str) -> float:
    return float(summary["median_stage_metrics"][name])


def main() -> None:
    offline = load_summary("offline-final")
    realtime = load_summary("media-clock-final")
    labels = ["A0\nOpenCV/CPU FP", "A1\nMPP/CPU FP", "A2\nMPP/RGA FP", "A3\nMPP/RGA INT8"]
    offline_ids = ["A0", "A1", "A2", "A3"]
    rt_ids = ["R0", "R2", "R3"]
    rt_labels = ["R0\nOpenCV/CPU FP", "R2\nMPP/RGA FP", "R3\nMPP/RGA INT8"]
    colors = ["#5b7897", "#8799a9", "#47a38b", "#e09a43"]

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0), constrained_layout=True)
    ax = axes[0]
    fps = [float(offline[key]["median_throughput_fps"]) for key in offline_ids]
    bars = ax.bar(range(4), fps, color=colors, width=0.68)
    ax.set_xticks(range(4), labels)
    ax.set_ylabel("Processed frames/s")
    ax.set_title("Unpaced offline worker")
    ax.grid(axis="y", alpha=0.22)
    ax.bar_label(bars, fmt="%.2f", padding=3)
    ax.set_ylim(0, max(fps) * 1.2)

    ax = axes[1]
    rt_fps = [float(realtime[key]["median_throughput_fps"]) for key in rt_ids]
    p95 = [metric(realtime[key], "p95_end_to_end_ms") for key in rt_ids]
    bars = ax.bar(range(3), rt_fps, color=[colors[i] for i in (0, 2, 3)], width=0.62)
    ax.axhline(10.0, color="#444444", linestyle="--", linewidth=1.2, label="10 FPS input")
    ax.set_xticks(range(3), rt_labels)
    ax.set_ylabel("Processed frames/s")
    ax.set_title("Media-clock input (10 FPS)")
    ax.set_ylim(0, 12.0)
    ax.grid(axis="y", alpha=0.22)
    drops = [int(np.median(realtime[key]["dropped_frames_each"])) for key in rt_ids]
    ax.bar_label(bars, labels=[f"{v:.2f}\n{d} drops" for v, d in zip(rt_fps, drops)], padding=3)
    right = ax.twinx()
    right.plot(range(3), p95, color="#af5353", marker="o", linewidth=2, label="Decode-to-result P95")
    right.set_ylabel("Decode-to-result P95 (ms)")
    right.set_ylim(0, max(p95) * 1.35)
    handles, labels2 = ax.get_legend_handles_labels()
    handles2, labels3 = right.get_legend_handles_labels()
    ax.legend(handles + handles2, labels2 + labels3, loc="upper left", fontsize=8, frameon=False)

    ax = axes[2]
    stage_names = [
        ("p50_preprocess_ms", "Preprocess"),
        ("p50_rknn_input_set_ms", "RKNN input set"),
        ("p50_rknn_run_ms", "RKNN run"),
        ("p50_rknn_output_fetch_ms", "Output fetch"),
        ("p50_postprocess_ms", "Postprocess"),
    ]
    width = 0.14
    x = np.arange(4)
    for stage_index, (field, label) in enumerate(stage_names):
        values = [metric(offline[key], field) for key in offline_ids]
        ax.bar(x + (stage_index - 2) * width, values, width=width, label=label)
    ax.set_xticks(x, ["A0", "A1", "A2", "A3"])
    ax.set_ylabel("Median stage time (ms)")
    ax.set_title("Offline worker stage medians")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(fontsize=8, frameon=False)

    fig.suptitle("RK3576 end-to-end backend and INT8 ablation", fontsize=14)
    fig.savefig(EVIDENCE / "flagship-ablation.png", dpi=180, bbox_inches="tight")
    fig.savefig(EVIDENCE / "flagship-ablation.svg", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
