#!/usr/bin/env python3
"""Serve a deterministic JPEG/RTP test pattern over RTSP from the WSL test environment."""

from __future__ import annotations

import argparse
import signal
import sys

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstRtsp", "1.0")
gi.require_version("GstRtspServer", "1.0")
from gi.repository import GLib, Gst, GstRtsp, GstRtspServer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8554)
    parser.add_argument("--mount", default="/test")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--pattern", choices=("ball", "smpte", "snow"), default="ball")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not (1 <= args.port <= 65535):
        raise SystemExit("port must be 1..65535")
    if not args.mount.startswith("/") or args.mount == "/":
        raise SystemExit("mount must be a non-root path beginning with /")
    if min(args.width, args.height, args.fps) <= 0:
        raise SystemExit("width, height, and fps must be positive")

    Gst.init(None)
    server = GstRtspServer.RTSPServer()
    server.set_address(args.bind)
    server.set_service(str(args.port))
    factory = GstRtspServer.RTSPMediaFactory()
    factory.set_launch(
        "( videotestsrc is-live=true pattern={pattern} "
        "! video/x-raw,format=I420,width={width},height={height},framerate={fps}/1 "
        "! jpegenc ! rtpjpegpay name=pay0 pt=26 )".format(
            pattern=args.pattern,
            width=args.width,
            height=args.height,
            fps=args.fps,
        )
    )
    factory.set_shared(True)
    factory.set_protocols(GstRtsp.RTSPLowerTrans.TCP)
    server.get_mount_points().add_factory(args.mount, factory)
    source_id = server.attach(None)
    if not source_id:
        print("failed to attach RTSP server socket", file=sys.stderr)
        return 1

    loop = GLib.MainLoop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda _signum, _frame: loop.quit())
    print(
        f"RTSP test stream ready: rtsp://<WSL-LAN-IP>:{args.port}{args.mount}; "
        f"pattern={args.pattern} {args.width}x{args.height}@{args.fps} tcp",
        flush=True,
    )
    loop.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
