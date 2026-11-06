#!/usr/bin/env python3
"""Generate fixed-scene MP4 inputs for repeatable PPE event-rule replays.

The public validation images are not video footage. Each generated clip repeats
one source image so that the board's temporal confirmation/cooldown and event
path can be exercised with a known binary PPE-event label.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / ".work/construction-ppe/extracted/construction-ppe"
DEFAULT_OUTPUT = ROOT / ".work/ppe-event-replays-20261106"
DEFAULT_AUDIT = ROOT / "evidence/ppe/construction-ppe-audit.json"
SCENARIOS = {
    "positive": {"image_id": 90, "expected_event": True},
    "negative": {"image_id": 200, "expected_event": False},
}
VIOLATIONS = {"no_helmet", "no_vest"}
GEAR = {"helmet", "vest"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contains_center(item: dict[str, Any], person: dict[str, Any]) -> bool:
    px, py, pw, ph = person["bbox"]
    x, y, width, height = item["bbox"]
    cx, cy = x + width / 2.0, y + height / 2.0
    return px <= cx <= px + pw and py <= cy <= py + ph


def validate_source(annotations_path: Path, images_dir: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(annotations_path.read_text(encoding="utf-8"))
    categories = {int(row["id"]): row["name"] for row in payload["categories"]}
    images = {int(row["id"]): row for row in payload["images"]}
    by_image: dict[int, list[dict[str, Any]]] = {}
    for annotation in payload["annotations"]:
        by_image.setdefault(int(annotation["image_id"]), []).append(annotation)

    result: dict[str, dict[str, Any]] = {}
    for scenario_name, spec in SCENARIOS.items():
        image_id = spec["image_id"]
        if image_id not in images:
            raise ValueError(f"COCO validation image id {image_id} is missing")
        image = images[image_id]
        rows = by_image.get(image_id, [])
        persons = [row for row in rows if categories[int(row["category_id"])] == "Person"]
        if len(persons) != 1:
            raise ValueError(f"image {image_id} must have exactly one labeled person, got {len(persons)}")
        labels = Counter(categories[int(row["category_id"])] for row in rows)
        linked = {
            categories[int(row["category_id"])]
            for row in rows
            if categories[int(row["category_id"])] in VIOLATIONS | GEAR
            and contains_center(row, persons[0])
        }
        event_present = any(labels[name] for name in VIOLATIONS)
        if event_present != spec["expected_event"]:
            raise ValueError(f"image {image_id} no longer matches the expected {scenario_name} label")
        if scenario_name == "positive" and not linked.intersection(VIOLATIONS):
            raise ValueError("positive validation image has no target violation linked to the person")
        if scenario_name == "negative" and not GEAR.issubset(linked):
            raise ValueError("negative validation image must have linked helmet and vest labels")
        image_path = images_dir / image["file_name"]
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        result[scenario_name] = {
            "source_image_id": image_id,
            "source_image": image["file_name"],
            "source_image_path": str(image_path.resolve()),
            "source_image_sha256": sha256_file(image_path),
            "source_labels": dict(sorted(labels.items())),
            "person_linked_ppe_labels": sorted(linked),
            "expected_ppe_violation_event": event_present,
        }
    return result


def probe_video(ffprobe: str, path: Path, expected_frames: int, expected_fps: int) -> dict[str, Any]:
    command = [
        ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=codec_name,profile,width,height,pix_fmt,r_frame_rate,nb_frames",
        "-of", "json", str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    streams = json.loads(result.stdout).get("streams", [])
    if len(streams) != 1:
        raise ValueError(f"expected one video stream in {path}")
    stream = streams[0]
    frames = int(stream.get("nb_frames", -1))
    fps = Fraction(stream.get("r_frame_rate", "0/1"))
    if frames != expected_frames or fps != expected_fps:
        raise ValueError(f"unexpected replay format for {path}: {frames} frames at {fps} fps")
    return {
        "codec": stream.get("codec_name"),
        "profile": stream.get("profile"),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "pixel_format": stream.get("pix_fmt"),
        "fps": float(fps),
        "frames": frames,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--frames", type=int, default=150)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    ffmpeg = shutil.which(args.ffmpeg)
    ffprobe = shutil.which(args.ffprobe)
    if not ffmpeg or not ffprobe:
        raise FileNotFoundError("ffmpeg and ffprobe must both be available on PATH")
    if args.frames <= 0 or args.fps <= 0 or args.width <= 0 or args.height <= 0:
        raise ValueError("frames, fps, width, and height must be positive")
    if args.width % 2 or args.height % 2:
        raise ValueError("H.264 yuv420p output requires even width and height")

    dataset_root = args.dataset_root.resolve()
    annotations_path = dataset_root / "annotations" / "instances_val.json"
    images_dir = dataset_root / "images" / "val"
    if not annotations_path.is_file():
        raise FileNotFoundError(f"COCO validation annotations not found: {annotations_path}")
    source_records = validate_source(annotations_path, images_dir)
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    version_result = subprocess.run([ffmpeg, "-version"], check=True,
                                    capture_output=True, text=True)
    ffmpeg_version = version_result.stdout.splitlines()[0]
    manifest: dict[str, Any] = {
        "generator": "scripts/generate_ppe_replay_videos.py",
        "dataset_release": audit.get("asset_release"),
        "dataset_archive_sha256": audit.get("archive", {}).get("sha256"),
        "dataset_license": audit.get("license"),
        "ffmpeg_version": ffmpeg_version,
        "settings": {
            "frames": args.frames,
            "fps": args.fps,
            "duration_seconds": args.frames / args.fps,
            "size": [args.width, args.height],
            "codec": "libx264",
            "profile": "baseline",
            "preset": "medium",
            "crf": args.crf,
            "gop_frames": args.fps,
            "pixel_format": "yuvj420p",
            "scale_filter": "area",
        },
        "scenarios": {},
    }

    for name, source_record in source_records.items():
        output_path = args.output_dir / f"ppe-{name}-{args.frames}f-{args.fps}fps.mp4"
        if output_path.exists() and not args.force:
            raise FileExistsError(f"refusing to overwrite {output_path}; pass --force to replace it")
        command = [
            ffmpeg, "-hide_banner", "-loglevel", "error",
            "-loop", "1", "-framerate", str(args.fps), "-i", source_record["source_image_path"],
            "-frames:v", str(args.frames),
            "-vf", f"scale={args.width}:{args.height}:flags=area,format=yuvj420p",
            "-c:v", "libx264", "-profile:v", "baseline", "-preset", "medium",
            "-crf", str(args.crf), "-g", str(args.fps), "-sc_threshold", "0",
            "-color_range", "pc", "-threads", "1", "-movflags", "+faststart",
        ]
        if args.force:
            command.append("-y")
        else:
            command.append("-n")
        command.append(str(output_path.resolve()))
        subprocess.run(command, check=True)
        video_format = probe_video(ffprobe, output_path, args.frames, args.fps)
        manifest["scenarios"][name] = {
            **source_record,
            "ffmpeg_command": command,
            "video": str(output_path.resolve()),
            "video_sha256": sha256_file(output_path),
            "video_bytes": output_path.stat().st_size,
            "video_format": video_format,
        }

    manifest_path = args.output_dir / "replay-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                             encoding="utf-8")
    print(json.dumps({
        "manifest": str(manifest_path.resolve()),
        "ffmpeg_version": ffmpeg_version,
        "scenarios": {
            name: {"video": row["video"], "sha256": row["video_sha256"],
                   "format": row["video_format"]}
            for name, row in manifest["scenarios"].items()
        },
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
