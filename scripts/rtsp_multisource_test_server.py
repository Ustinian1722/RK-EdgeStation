#!/usr/bin/env python3
"""Serve 1..32 independent, moving synthetic H.264 camera sources over RTSP.

This is a transport/decoder/concurrency fixture, not PPE accuracy data. Each
mount has a unique ball velocity so clients do not subscribe to one shared
camera pipeline. The source runs on the development host; RK3576 telemetry is
collected separately by run_multistream_benchmark.py.
"""

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
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8554)
    parser.add_argument("--streams", type=int, default=32)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--bitrate", type=int, default=256_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not (1 <= args.port <= 65535):
        raise SystemExit("port must be 1..65535")
    if not (1 <= args.streams <= 32):
        raise SystemExit("streams must be 1..32")
    if min(args.width, args.height, args.fps, args.bitrate) <= 0:
        raise SystemExit("width, height, fps, and bitrate must be positive")

    Gst.init(None)
    server = GstRtspServer.RTSPServer()
    server.set_address(args.bind)
    server.set_service(str(args.port))
    mounts = server.get_mount_points()
    for index in range(args.streams):
        speed = index + 1
        launch = (
            "( videotestsrc is-live=true do-timestamp=true pattern=ball "
            f"animation-mode=wall-time horizontal-speed={speed} "
            f"! video/x-raw,format=I420,width={args.width},height={args.height},"
            f"framerate={args.fps}/1 "
            f"! openh264enc bitrate={args.bitrate} complexity=low multi-thread=1 "
            f"gop-size={max(args.fps * 2, 2)} "
            "! video/x-h264,profile=baseline "
            "! rtph264pay name=pay0 pt=96 config-interval=1 )"
        )
        factory = GstRtspServer.RTSPMediaFactory()
        factory.set_launch(launch)
        factory.set_shared(True)
        factory.set_protocols(GstRtsp.RTSPLowerTrans.TCP)
        mount = f"/cam-{index + 1:02d}"
        mounts.add_factory(mount, factory)

    source_id = server.attach(None)
    if not source_id:
        print("failed to attach RTSP server socket", file=sys.stderr)
        return 1

    loop = GLib.MainLoop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda _signum, _frame: loop.quit())
    print(
        f"RTSP multi-source fixture ready: rtsp://{args.bind}:{args.port}/cam-NN; "
        f"streams={args.streams} {args.width}x{args.height}@{args.fps} "
        f"H.264/OpenH264, TCP, synthetic moving ball",
        flush=True,
    )
    loop.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
