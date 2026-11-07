#!/usr/bin/env python3
"""Download only the Test split from a Roboflow Universe version export.

ROBOFLOW_API_KEY is read from the process environment and is never persisted
or included in output. The temporary ZIP is deleted after the Test files are
extracted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any


WORKSPACE = "roboflow-universe-projects"
PROJECT = "construction-site-safety"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def api_json(url: str) -> tuple[int, dict[str, Any]]:
    request = urllib.request.Request(url, headers={"User-Agent": "RK-EdgeStation dataset audit"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        if error.code == 202:
            return error.code, json.loads(error.read())
        raise RuntimeError(f"Roboflow API HTTP status {error.code}") from None
    except urllib.error.URLError as error:
        raise RuntimeError(f"Roboflow API network error: {error.reason}") from None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--version", default="1")
    args = parser.parse_args()
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        raise SystemExit("ROBOFLOW_API_KEY is not set")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive = args.output_dir / ".roboflow-export.tmp.zip"
    query = urllib.parse.urlencode({"api_key": key, "nocache": "true"})
    endpoint = f"https://api.roboflow.com/{WORKSPACE}/{PROJECT}/{args.version}/yolov8?{query}"
    deadline = time.monotonic() + 600
    payload: dict[str, Any]
    while True:
        status, payload = api_json(endpoint)
        if status == 200 and "export" in payload:
            break
        if status != 202 and payload.get("ready") is not False:
            raise RuntimeError(f"Roboflow export response missing link; fields={sorted(payload)}")
        if time.monotonic() >= deadline:
            raise TimeoutError("Timed out waiting for Roboflow YOLOv8 export")
        print(f"Roboflow v{args.version} YOLOv8 export is generating ({payload.get('progress', 0)}).")
        time.sleep(3)

    download_url = payload["export"].get("link")
    if not isinstance(download_url, str) or not download_url.startswith("https://"):
        raise RuntimeError("Roboflow export did not return an HTTPS download link")
    request = urllib.request.Request(download_url, headers={"User-Agent": "RK-EdgeStation dataset audit"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            expected_size = response.headers.get("Content-Length")
            free_bytes = shutil.disk_usage(args.output_dir).free
            if expected_size and int(expected_size) > free_bytes - 32 * 1024 * 1024:
                raise RuntimeError("Roboflow export ZIP exceeds the available local working space")
            with archive.open("wb") as output:
                shutil.copyfileobj(response, output)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"Roboflow ZIP HTTP status {error.code}") from None
    except urllib.error.URLError as error:
        raise RuntimeError(f"Roboflow ZIP network error: {error.reason}") from None

    archive_hash = sha256_file(archive)
    test_root = args.output_dir / "test"
    if test_root.exists():
        shutil.rmtree(test_root)
    (test_root / "images").mkdir(parents=True)
    (test_root / "labels").mkdir(parents=True)
    image_suffixes = {".jpg", ".jpeg", ".png", ".bmp"}
    image_count = label_count = 0
    yaml_member = None
    try:
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                path = PurePosixPath(member.filename)
                if path.name == "data.yaml" and yaml_member is None:
                    yaml_member = member
                parts = path.parts
                try:
                    split_index = parts.index("test")
                except ValueError:
                    continue
                suffix = path.suffix.lower()
                if "images" in parts[split_index + 1 :] and suffix in image_suffixes:
                    destination = test_root / "images" / path.name
                    with bundle.open(member) as source, destination.open("wb") as target:
                        shutil.copyfileobj(source, target)
                    image_count += 1
                elif "labels" in parts[split_index + 1 :] and suffix == ".txt":
                    destination = test_root / "labels" / (path.stem + ".txt")
                    with bundle.open(member) as source, destination.open("wb") as target:
                        shutil.copyfileobj(source, target)
                    label_count += 1
            if yaml_member is not None:
                with bundle.open(yaml_member) as source, (test_root / "data.yaml").open("wb") as target:
                    shutil.copyfileobj(source, target)
    finally:
        archive.unlink(missing_ok=True)

    if yaml_member is None or image_count != 34:
        raise RuntimeError(
            f"Expected data.yaml and exactly 34 Test images; found yaml={yaml_member is not None}, "
            f"images={image_count}, labels={label_count}"
        )
    manifest = {
        "dataset": f"{WORKSPACE}/{PROJECT}",
        "version": args.version,
        "format": "yolov8",
        "split": "test",
        "source_url": f"https://universe.roboflow.com/{WORKSPACE}/{PROJECT}/dataset/{args.version}",
        "test_image_count": image_count,
        "test_label_file_count": label_count,
        "export_zip_sha256": archive_hash,
        "files": {
            str(path.relative_to(test_root)).replace("\\", "/"): sha256_file(path)
            for path in sorted(test_root.rglob("*")) if path.is_file()
        },
        "api_key_persisted": False,
        "other_splits_retained": False,
    }
    (args.output_dir / "download-manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: manifest[key] for key in
                      ("dataset", "version", "split", "test_image_count", "test_label_file_count", "export_zip_sha256")},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
