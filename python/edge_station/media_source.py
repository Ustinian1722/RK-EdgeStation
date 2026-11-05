"""Helpers for keeping RTSP URIs distinct from local filesystem paths."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import SplitResult, urlsplit


def _rtsp_parts(value: str) -> SplitResult | None:
    try:
        parsed = urlsplit(value)
    except ValueError:
        return None
    if parsed.scheme.lower() not in {"rtsp", "rtsps"}:
        return None
    return parsed


def is_rtsp_source(value: str) -> bool:
    return _rtsp_parts(value) is not None


def resolve_worker_source(value: str) -> str:
    parsed = _rtsp_parts(value)
    if parsed is None:
        return str(Path(value).resolve())
    if not parsed.hostname:
        raise ValueError("RTSP URL must include a host")
    try:
        parsed.port
    except ValueError as error:
        raise ValueError(f"invalid RTSP URL port: {error}") from error
    return value


def display_source(value: str) -> str:
    parsed = _rtsp_parts(value)
    if parsed is None:
        return Path(value).name
    if not parsed.hostname:
        return parsed.scheme.lower() + "://invalid"
    host = parsed.hostname
    if ":" in host:
        host = f"[{host}]"
    try:
        port = parsed.port
    except ValueError:
        port = None
    default_port = 322 if parsed.scheme.lower() == "rtsps" else 554
    suffix = f":{port}" if port is not None and port != default_port else ""
    return f"{parsed.scheme.lower()}://{host}{suffix}"
