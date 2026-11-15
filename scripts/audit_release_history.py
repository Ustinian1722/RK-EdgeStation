#!/usr/bin/env python3
"""Best-effort secret and asset screening of reachable Git objects.

Print rule IDs and file paths only; never print secret values.
Run in a full clone with all branches and tags fetched.
"""
import gzip
import io
import re
import subprocess
import sys
from pathlib import PurePosixPath

RULES = {
    "private-key": rb"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----",
    "github-token": rb"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{35,})",
    "aws-access-key": rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b",
    "huggingface-token": rb"\bhf_[A-Za-z0-9]{30,}\b",
    "api-key": rb"\bsk-(?:proj-)?[A-Za-z0-9_-]{32,}\b",
    "slack-token": rb"\bxox[baprs]-[A-Za-z0-9-]{20,}\b",
    "jwt": rb"\beyJ[A-Za-z0-9_-]{12,}\.eyJ[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{15,}\b",
    "assigned-secret": rb"(?im)^\s*(?:export\s+)?(?:[A-Z_]*(?:TOKEN|API_KEY|PASSWORD|SECRET|ACCESS_KEY)[A-Z_]*)\s*=\s*['\"]?[A-Za-z0-9_+/.-]{20,}",
}
RULES = {name: re.compile(pattern) for name, pattern in RULES.items()}
SENSITIVE_EXT = {
    ".pem", ".key", ".p12", ".pfx", ".rknn", ".onnx", ".pt", ".pth",
    ".safetensors", ".mp4", ".avi", ".webm", ".mov", ".mkv",
    ".sqlite", ".sqlite3", ".db", ".zip", ".7z", ".rar", ".tar"
}

def git(*args):
    return subprocess.check_output(["git", *args])

def main():
    objects = []
    for row in git("rev-list", "--objects", "--all").splitlines():
        oid, _, path = row.partition(b" ")
        objects.append((oid, path.decode("utf-8", "replace")))
    meta = subprocess.check_output(
        ["git", "cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)"],
        input=b"".join(oid + b"\n" for oid, _ in objects),
    ).splitlines()
    paths = dict(objects)
    findings = set()
    assets = set()
    skipped = set()
    scanned = 0
    # Batch the blobs so no large archive is expanded without a cap.
    for line in meta:
        oid, kind, size = line.split()
        if kind != b"blob":
            continue
        path = paths[oid] or "(unnamed blob)"
        p = PurePosixPath(path)
        if p.suffix.lower() in SENSITIVE_EXT or p.name.startswith(("id_rsa", "id_ed25519")):
            assets.add(path)
        if int(size) > 16 * 1024 * 1024:
            skipped.add(path)
            continue
        data = git("cat-file", "blob", oid.decode())
        if data.startswith(b"\x1f\x8b"):
            try:
                with gzip.GzipFile(fileobj=io.BytesIO(data)) as stream:
                    data = stream.read(16 * 1024 * 1024 + 1)
                if len(data) > 16 * 1024 * 1024:
                    skipped.add(path)
                    continue
            except (OSError, EOFError):
                skipped.add(path)
                continue
        if b"\x00" in data[:4096]:
            continue  # Binary/media/license inventory is reviewed separately.
        scanned += 1
        for rule, pattern in RULES.items():
            if pattern.search(data):
                # Do not flag this script's own patterns as literal private keys.
                if rule == "private-key" and path == "scripts/audit_release_history.py":
                    continue
                findings.add((rule, path))
        for match in re.finditer(rb"(?i)rtsps?://([^/\s@]+):([^/\s@]+)@", data):
            user, password = match.groups()
            if path == "tests/python/test_server.py" and (user, password) == (b"u", b"p") and data[match.end():].startswith(b"camera.example:8554/live?x=1"):
                continue  # Verified synthetic credential-redaction test fixture.
            if user.lower() not in {b"user", b"username", b"admin", b"camera_user", b"test", b"demo"} or password.lower() not in {b"pass", b"password", b"secret", b"camera_password", b"test", b"demo", b"p", b"pw", b"pwd"}:
                findings.add(("rtsp-userinfo-review", path))
    print(f"Reachable Git objects: {len(objects)}; scanned text/gzip blobs: {scanned}")
    for rule, path in sorted(findings):
        print(f"REVIEW [{rule}] {path}")
    for path in sorted(assets):
        print(f"ASSET REVIEW {path}")
    for path in sorted(skipped):
        print(f"UNSCANNED {path}")
    print(f"Credential findings: {len(findings)}; asset paths: {len(assets)}; oversized/invalid blobs: {len(skipped)}")
    print("Pattern screening only; does not establish absence of secrets or media licensing.")
    return 1 if findings or assets or skipped else 0

if __name__ == "__main__":
    sys.exit(main())
