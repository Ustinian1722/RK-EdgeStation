# Changelog

## v0.1.0 — experimental source release

- Consolidates the original single-stream, DMA-BUF research and multistream research lineage in one candidate.
- Documents the C++17 worker, MPP/RGA/RKNN backends, event service and model families.
- Documents optional partial low-copy DMA-BUF/shared-input mode; the copy path remains default.
- Documents 1–32 synthetic RTSP connection tests and serialized shared-context throughput limitations.
- Adds bilingual README, current documentation/benchmark indexes, architecture diagram, host test entry point and release gates.
- Historical board measurements are retained; no new board qualification is implied.

Source release scope: historical board evidence and host checks. A fresh board acceptance run or demo is optional; no production qualification is implied. See [release checklist](docs/release-checklist.md).
