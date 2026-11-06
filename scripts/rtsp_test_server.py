#!/usr/bin/env python3
"""Serve a deterministic RTSP test pattern on a loopback interface by default."""

from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstRtsp", "1.0")
gi.require_version("GstRtspServer", "1.0")
from gi.repository import GLib, Gst, GstRtsp, GstRtspServer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8554)
    parser.add_argument("--mount", default="/test")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--pattern", choices=("ball", "smpte", "snow"), default="ball")
    parser.add_argument("--codec", choices=("jpeg", "h264"), default="jpeg")
    parser.add_argument("--h264-encoder", choices=("x264enc", "mpph264enc"), default="x264enc")
    parser.add_argument("--source-file", type=Path,
                        help="serve an existing H.264 MP4 when the selected encoder is unavailable")
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
    if args.source_file is not None:
        if args.codec != "h264":
            parser.error("--source-file requires --codec h264")
        source_file = args.source_file.resolve()
        if not source_file.is_file():
            parser.error(f"H.264 source file does not exist: {source_file}")
        escaped_path = str(source_file).replace("\\", "\\\\").replace('"', '\\"')
        launch = (
            f'( filesrc location="{escaped_path}" ! qtdemux ! identity sync=true '
            "! rtph264pay name=pay0 pt=96 config-interval=1 )"
        )
    elif args.codec == "h264":
        encoder = (
            "x264enc tune=zerolatency speed-preset=ultrafast bitrate=600 "
            f"key-int-max={args.fps}"
            if args.h264_encoder == "x264enc"
            else "mpph264enc"
        )
        launch = (
            "( videotestsrc is-live=true pattern={pattern} "
            "! video/x-raw,format=I420,width={width},height={height},framerate={fps}/1 "
            "! {encoder} ! video/x-h264,profile=baseline "
            "! rtph264pay name=pay0 pt=96 config-interval=1 )"
        ).format(pattern=args.pattern, width=args.width, height=args.height,
                 fps=args.fps, encoder=encoder)
    else:
        launch = (
            "( videotestsrc is-live=true pattern={pattern} "
            "! video/x-raw,format=I420,width={width},height={height},framerate={fps}/1 "
            "! jpegenc ! rtpjpegpay name=pay0 pt=26 )"
        ).format(pattern=args.pattern, width=args.width, height=args.height, fps=args.fps)
    factory.set_launch(launch)
    # File-backed test clients need a fresh media pipeline after EOS/reconnect.
    factory.set_shared(args.source_file is None)
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
        f"pattern={args.pattern} codec={args.codec} {args.width}x{args.height}@{args.fps} tcp",
        flush=True,
    )
    loop.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
