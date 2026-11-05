#!/usr/bin/env python3
"""Measure decoded buffers from the RK GStreamer MPP H.264 path."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from pathlib import Path
from typing import Any

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def process_rss_kb() -> int | None:
    try:
        for line in Path("/proc/self/status").read_text(encoding="ascii").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        return None
    return None


def make_element(factory: str, name: str) -> Gst.Element:
    element = Gst.ElementFactory.make(factory, name)
    if element is None:
        raise RuntimeError(f"GStreamer element is unavailable: {factory}")
    return element


def run_once(video: Path, expected_frames: int | None,
             output_format: str) -> dict[str, Any]:
    pipeline = Gst.Pipeline.new("mpp-decode-benchmark")
    if pipeline is None:
        raise RuntimeError("could not create GStreamer pipeline")

    source = make_element("filesrc", "source")
    demux = make_element("qtdemux", "demux")
    input_caps = make_element("capsfilter", "parsed-h264-caps")
    decoder = make_element("mppvideodec", "mpp-decoder")
    converter = make_element("videoconvert", "color-converter") if output_format == "BGR" else None
    output_caps = make_element("capsfilter", "nv12-caps")
    sink = make_element("fakesink", "frame-sink")

    source.set_property("location", str(video.resolve()))
    input_caps.set_property(
        "caps", Gst.Caps.from_string(
            "video/x-h264,parsed=true,stream-format=avc,alignment=au"))
    output_caps.set_property("caps", Gst.Caps.from_string(f"video/x-raw,format={output_format}"))
    sink.set_property("sync", False)
    sink.set_property("signal-handoffs", True)
    sink.set_property("enable-last-sample", False)

    elements = [source, demux, input_caps, decoder]
    if converter is not None:
        elements.append(converter)
    elements.extend((output_caps, sink))
    for element in elements:
        pipeline.add(element)
    if not source.link(demux):
        raise RuntimeError("could not link filesrc to qtdemux")
    downstream = decoder
    if converter is not None:
        if not downstream.link(converter):
            raise RuntimeError("could not link MPP decoder to videoconvert")
        downstream = converter
    if not input_caps.link(decoder) or not downstream.link(output_caps) or not output_caps.link(sink):
        raise RuntimeError("could not link the MPP decoder pipeline")

    pad_link_result: list[str] = []

    def on_demux_pad(_element: Gst.Element, pad: Gst.Pad) -> None:
        caps = pad.get_current_caps() or pad.query_caps(None)
        if caps is None or caps.get_size() == 0:
            return
        if caps.get_structure(0).get_name() != "video/x-h264":
            return
        sink_pad = input_caps.get_static_pad("sink")
        if sink_pad is not None and not sink_pad.is_linked():
            result = pad.link(sink_pad)
            if result != Gst.PadLinkReturn.OK:
                pad_link_result.append(str(result))

    demux.connect("pad-added", on_demux_pad)

    stats: dict[str, Any] = {
        "decoded_frames": 0,
        "first_frame_latency_ms": None,
        "first_buffer_bytes": None,
        "first_pts_ns": None,
        "last_pts_ns": None,
        "output_caps": None,
        "warnings": [],
    }
    start_time: list[float | None] = [None]

    def on_handoff(_sink: Gst.Element, buffer: Gst.Buffer, _pad: Gst.Pad,
                   _data: Any = None) -> None:
        now = time.monotonic()
        stats["decoded_frames"] += 1
        pts = buffer.pts
        if pts != Gst.CLOCK_TIME_NONE:
            if stats["first_pts_ns"] is None:
                stats["first_pts_ns"] = int(pts)
            stats["last_pts_ns"] = int(pts)
        if stats["first_frame_latency_ms"] is None:
            stats["first_frame_latency_ms"] = (now - start_time[0]) * 1000.0
            stats["first_buffer_bytes"] = int(buffer.get_size())
            caps = output_caps.get_static_pad("src").get_current_caps()
            stats["output_caps"] = caps.to_string() if caps is not None else None

    sink.connect("handoff", on_handoff)
    bus = pipeline.get_bus()
    cpu_before = resource.getrusage(resource.RUSAGE_SELF)
    start_time[0] = time.monotonic()
    pipeline.set_state(Gst.State.PLAYING)
    try:
        while True:
            message = bus.timed_pop_filtered(
                60 * Gst.SECOND,
                Gst.MessageType.EOS | Gst.MessageType.ERROR | Gst.MessageType.WARNING,
            )
            if message is None:
                raise TimeoutError("MPP decode pipeline did not reach EOS within 60 seconds")
            if message.type == Gst.MessageType.WARNING:
                warning, debug = message.parse_warning()
                stats["warnings"].append({"message": str(warning), "debug": debug})
                continue
            if message.type == Gst.MessageType.ERROR:
                error, debug = message.parse_error()
                raise RuntimeError(f"GStreamer error: {error}; debug={debug}")
            if message.type == Gst.MessageType.EOS:
                break
    finally:
        end_time = time.monotonic()
        pipeline.set_state(Gst.State.NULL)
    cpu_after = resource.getrusage(resource.RUSAGE_SELF)

    elapsed = end_time - start_time[0]
    cpu_seconds = ((cpu_after.ru_utime - cpu_before.ru_utime) +
                   (cpu_after.ru_stime - cpu_before.ru_stime))
    frames = int(stats["decoded_frames"])
    if pad_link_result:
        raise RuntimeError(f"could not link qtdemux H.264 pad: {pad_link_result[0]}")
    if frames == 0:
        raise RuntimeError("MPP pipeline reached EOS without producing decoded frames")
    if expected_frames is not None and frames != expected_frames:
        raise RuntimeError(f"MPP decoded {frames} frames; expected {expected_frames}")

    return {
        **stats,
        "elapsed_seconds": elapsed,
        "throughput_fps": frames / elapsed,
        "process_cpu_seconds": cpu_seconds,
        "process_cpu_core_percent": cpu_seconds / elapsed * 100.0 if elapsed else None,
        "process_rss_kb_after": process_rss_kb(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--expected-frames", type=int)
    parser.add_argument("--output-format", choices=("NV12", "BGR"), default="NV12")
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    video = args.video.resolve()
    if not video.is_file():
        parser.error(f"input video does not exist: {video}")
    if args.warmup_runs < 0 or args.repetitions < 1:
        parser.error("warmup-runs must be >= 0 and repetitions must be >= 1")

    Gst.init(None)
    factory = Gst.ElementFactory.find("mppvideodec")
    if factory is None:
        raise RuntimeError("Rockchip GStreamer mppvideodec is unavailable")
    plugin = factory.get_plugin()

    for _ in range(args.warmup_runs):
        run_once(video, args.expected_frames, args.output_format)
    results = [run_once(video, args.expected_frames, args.output_format)
               for _ in range(args.repetitions)]
    throughputs = sorted(result["throughput_fps"] for result in results)
    median_fps = throughputs[len(throughputs) // 2]

    report = {
        "video_filename": video.name,
        "video_bytes": video.stat().st_size,
        "video_sha256": file_sha256(video),
        "gstreamer_version": Gst.version_string(),
        "mpp_plugin_name": plugin.get_name() if plugin is not None else None,
        "mpp_plugin_version": plugin.get_version() if plugin is not None else None,
        "mpp_element": factory.get_name(),
        "decoder_output_format": args.output_format,
        "expected_frames": args.expected_frames,
        "warmup_runs": args.warmup_runs,
        "repetitions": results,
        "median_throughput_fps": median_fps,
        "interpretation": (
            "MPP decode to NV12 and fakesink; no color conversion, resize, inference, or DMA-BUF "
            "import is measured."
            if args.output_format == "NV12" else
            "MPP decode followed by GStreamer videoconvert to BGR and fakesink; no resize, "
            "inference, or DMA-BUF import is measured."
        ),
    }
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
