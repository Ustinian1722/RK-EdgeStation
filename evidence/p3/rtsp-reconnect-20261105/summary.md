# RK3576 RTSP reconnect check

**Result:** Pass for the tested synthetic stream and TCP transport. This does not establish compatibility with a physical camera, alternate codecs, camera authentication, or a customer network.

## Setup

- Target: KICKPI K7 / RK3576, Ubuntu 24.04.3, G++ 13.3, CMake 3.28.3, OpenCV 4.6, RKNN Runtime 2.0.0b0.
- Input: WSL GStreamer RTSP server, JPEG/RTP at 640×480 and 10 FPS, reached through an SSH reverse tunnel bound to board loopback port 18554.
- Model SHA-256: `2162e0cf6a71270c092353d94abb391ac111910ce178547ef80dd89b225dab0b`.
- Candidate worker SHA-256: `f5957481d4118af1d3a7f697625363e71a6a24b4ced0d806c97e728f2faed66f`.
- Build directory: `.work/rtsp-build`; the in-progress 24-hour run continued to use its original `build/bin/rk_edge_worker` binary.

## Procedure and outcome

1. Started the updated Python service on board loopback port 18084 with the candidate worker and test stream URL.
2. Confirmed `/api/health` reported `source_type=rtsp`, `stream_epoch=0`, and advancing decoded/processed frame counters. The worker performed RKNN inference and stored ROI events.
3. Stopped the WSL RTSP server while leaving the reverse tunnel and board service up. The worker remained alive; `/api/health` showed `source_status=reconnecting` and `reconnect_attempts=3` while frame counters stayed available from the last successful frame. Retry status is merged into the health snapshot without adding a SQLite metric row per retry.
4. Restarted the server at the same endpoint. The worker resumed decoding; health returned to `source_status=connected` with `reconnect_attempts=4` and `stream_epoch=1`. Tracker and active ROI state reset while cumulative-entry state remained available.
5. Sent SIGTERM to the service. The worker exited cleanly. The final service log reported 667 decoded frames, 582 processed frames, 299 detections, 7 events, and 85 dropped frames. SQLite `integrity_check` returned `ok`; the database contained 7 events and 21 metrics.

The captured post-reconnect health record shows the service running with storage available and zero storage/event/metric write failures. The latest saved RTSP events use `source_id=rtsp`. The run is an integration/recovery check, not a throughput benchmark: the synthetic source ran at 10 FPS, and the service included an outage interval.

## Artifacts

- [Post-reconnect health](health-after-reconnect.json)
- [Health during the source outage](health-during-outage.json)
- [Recent stored events](events-after-reconnect.json)
- [Board service log](board-service.log)
- [WSL Python test log](python-tests.log)
- Source and unit tests: `src/main.cpp`, `src/tracker.hpp`, `src/roi_events.hpp`, `tests/cpp_core_tests.cpp`, `python/edge_station/media_source.py`, `python/edge_station/server.py`, `tests/python/test_server.py`, and `scripts/rtsp_test_server.py`.

CTest passed 2/2 on RK3576 after building the candidate in the isolated directory. Python API/storage tests passed 9/9 in the WSL Toolkit2 environment after adding reconnect-health and worker-failure supervision cases. Python bytecode compilation passed for the changed service and RTSP test server.
