#!/usr/bin/env python3
"""Write the ordered COCO image list consumed by the board PPE evaluator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.image_root.resolve()
    document = json.loads(args.annotations.read_text(encoding="utf-8"))
    names = [image["file_name"] for image in document["images"]]
    if not names or len(names) != len(set(names)):
        raise ValueError("COCO annotations need a non-empty, unique image list")
    missing = [name for name in names if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} annotated images are missing: {missing[:5]}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(str(root) + "\n" + "\n".join(names) + "\n", encoding="utf-8")
    print(f"manifest_images={len(names)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
