# Build and deploy

Status: native RK3576 build/tests, still/moving detection-to-event/API smokes, synthetic RTSP reconnect, and temporary systemd user-service recovery pass. Full P1 acceptance remains open for tracker continuity, numerical model comparison, labeled accuracy, and reliable NPU-utilization evidence.

## Current supported development path

The board is aarch64 Ubuntu 24.04.3 with GCC/G++ 13.3 and CMake 3.28.3, so native C++17/CMake builds are available without adding a cross toolchain. The Windows host has no C++ compiler; the application is built on the board. Toolkit2 is only needed to create or convert a project model. The app's build and inference path uses the board's existing RKNN C runtime. See [development environments](DEVELOPMENT_ENVIRONMENT.md) for the verified WSL setup, quick checks, and repair command.

Board's dedicated project path is `/home/kickpi/projects/RK-EdgeStation`. It was created for this project without touching existing user project directories.

## Planned reproducible build interface

From the project root on the board, build and run the local checks with:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 4
ctest --test-dir build --output-on-failure
PYTHONPATH=python python3 -m unittest discover -s tests/python -v
```

The same sequence is available as `bash scripts/build_and_test_board.sh`; it passed on the target board on 2026-11-05 after the ROI event and media-time changes. Launch a local MP4/model run with `bash scripts/run_demo.sh MODEL.rknn INPUT.mp4`. The UI listens on `127.0.0.1:8080` by default. Queue policy defaults to `latest` (replace the oldest pending frame); select FIFO with `bash scripts/run_demo.sh MODEL.rknn INPUT.mp4 state 8080 fifo` or pass `--queue-policy fifo` to the worker/service. FIFO keeps queued frames in order and drops newly decoded frames while full. The worker also supports `--dwell-ms N` (default 5000) and `--roi-debounce-frames N` (default 2); the Python service forwards these options. Dwell time follows monotonicized media timestamps, using nominal FPS or decode elapsed time when source timestamps are invalid. The opt-in `--loop-video` flag reopens a local MP4 at EOF while preserving increasing frame sequence and monotonicized media time; normal runs remain one-shot. Model files and run state stay outside Git. The first project model was converted in the isolated WSL Toolkit2 environment and run from the board workspace; see [MODEL_CONVERSION.md](MODEL_CONVERSION.md) and [P1_ACCEPTANCE_REPORT.md](P1_ACCEPTANCE_REPORT.md).

## RTSP input and reconnect

Pass an RTSP URL as the service's `--video` value; local-file path resolution and existence checks are skipped for RTSP sources. The worker opens RTSP through OpenCV's FFmpeg backend over TCP with 3-second open/read timeouts. A failed read or connection is released and retried with an interruptible 0.5-to-5-second exponential backoff. Frame sequence stays monotonic across reconnects; each successful reconnection increments `stream_epoch`, resets tracker and current ROI occupancy, and preserves cumulative ROI entries. Health metrics include input type, connected/reconnecting state, reconnect attempts, and stream epoch. Retry status updates the health API without writing every retry to SQLite. A nonzero worker exit makes the Python service exit with failure so `Restart=on-failure` can recover it. Health source display strips URL credentials, path, and query parameters.

The board reconnect gate passed against a synthetic 640×480@10 FPS JPEG/RTP stream served from the isolated WSL test environment through an SSH reverse tunnel. On the RK3576 worker, the run decoded 667 frames, processed 582, and emitted 7 stored events. During source loss, `/api/health` showed `source_status=reconnecting` after 3 attempts; restoring the source resumed inference at `source_status=connected`, `reconnect_attempts=4`, and `stream_epoch=1`. SIGTERM stopped the worker cleanly and SQLite integrity passed. This verifies software reconnect behavior with the selected codec/transport; physical cameras, alternate codecs, authentication, and customer network behavior remain separate checks. The reproducible harness is [scripts/rtsp_test_server.py](scripts/rtsp_test_server.py), and the captured evidence is in [evidence/p3/rtsp-reconnect-20261105](evidence/p3/rtsp-reconnect-20261105/summary.md).

## P3 stability run

The monitored runner below defaults to 24 hours and writes its manifest, one-minute JSONL samples, service log, and final database checks under a fresh state directory. It refuses a non-empty output directory. It stops on service/worker failure, any SQLite operation/write error, free space below 8 GiB, output growth above 4 GiB, `MemAvailable` below 512 MiB, worker RSS above 2 GiB, or a thermal zone at 80 °C. The runner records system CPU deltas, worker CPU/RSS, thermal readings, raw available NPU-load telemetry when exposed, throughput/latency, frame counters, SQLite/event counts, and free disk. At completion it checks SQLite integrity and verifies every event's evidence path. These checks are guardrails; the bounded file-size simulation is not a substitute for actual filesystem exhaustion.

On the RK3576 board, create a low-event deterministic 10-second, 5-FPS source from one selected sample frame, then start the full soak with a unique state directory and the source commit identifier:

```bash
build/bin/create_static_video \
  .work/p2-mpp-test/vtest-150f-h264.mp4 75 \
  .work/p3-static-frame75-5fps-raw.avi 5 10

PYTHONPATH=python python3 scripts/run_stability_test.py \
  --model .work/p1-yolov8n-rk3576-fp/yolov8n-rk3576-fp.rknn \
  --video .work/p3-static-frame75-5fps-raw.avi \
  --state-dir ".work/p3-soak-$(date -u +%Y%m%dT%H%M%SZ)" \
  --source-commit "$SOURCE_COMMIT"
```

Set `SOURCE_COMMIT` to the short commit hash synced and built on the board before running the command. The uncompressed static source is verified for exact frame count and identical decoded pixels; its SHA-256 is recorded in [the source manifest](evidence/p3/static-video-manifest.txt) and the soak manifest. A 45-second loop smoke passed. The board run started at `2026-11-05T14:12:38Z` from source commit `640d6f3`, using state directory `.work/p3-soak-640d6f3-20261105`; its first sample showed the service and worker running. The 24-hour gate remains in progress until the requested duration completes and the final SQLite/evidence checks pass. The start snapshot is [recorded here](evidence/p3/24h-soak-start-20261105.txt). Use a new output directory for every attempt and inspect `monitor/summary.json`, `monitor/samples.jsonl`, and `monitor/server.log` after completion.

The worker discovers board OpenCV and RKNN headers/libraries and links dynamically to those installed libraries. The current reference path uses OpenCV CPU decode/preprocessing and paces file input at its reported media rate when valid. The queue policy is included in worker metric JSON as `queue_policy`; queue capacity and `dropped_frames` are reported alongside it. P2 exercised MPP decode and RGA letterboxing as isolated board probes; full worker integration remains deferred pending controlled end-to-end and output checks. Python's standard library provides the local HTTP server and SQLite store.

## P3 graceful process restart recovery

This board-side probe starts the service and worker, records health and process resources, sends the service SIGTERM, checks worker exit and SQLite/evidence integrity, then restarts against the same state directory and verifies event retention. It refuses an existing state path. The probe records per-second samples and separate service logs under its state directory.

```bash
PYTHONPATH=python python3 scripts/restart_recovery_smoke.py \
  --model .work/p1-yolov8n-rk3576-fp/yolov8n-rk3576-fp.rknn \
  --video .work/p3-static-frame75-5fps-raw.avi \
  --state-dir ".work/p3-restart-$(date -u +%Y%m%dT%H%M%SZ)" \
  --port 18084 --observe-seconds 15 \
  --source-commit "$SOURCE_COMMIT"
```

Set `SOURCE_COMMIT` to the probe source commit synced to the board. The 2026-11-05 RK3576 run passed in 31.63 seconds: manual stop exit codes were both 0, the worker PID changed after restart, 14 saved events remained available on restart, and final SQLite integrity/evidence checks passed. The run sampled service and worker RSS, thread count, and file descriptor count. This does not verify systemd automatic restart.

## Deployment controls

- Deploy only to `/home/kickpi/projects/RK-EdgeStation`.
- Use the task-specific SSH key under the user's `.ssh` directory; do not copy private keys or passwords to the board or repo.
- Do not run `sudo`, modify groups, alter system libraries, replace kernel/drivers, or flash storage.
- The management service binds to `127.0.0.1`; no LAN listener is enabled by default.
- The board runtime reports 2.0.0b0. The official RKNN C API smoke test passed even though `/dev/rknpu*` nodes were not visible.
- Toolkit2 2.0.0b0 conversion is available in the isolated Ubuntu 22.04 x86-64 WSL2 distro, but no Rockchip silicon or runtime is installed on the host. Conversion is optional once a compatible `.rknn` exists. Model inference and the deployed application run on RK3576.
- The project YOLOv8n path completed 16/16 repeated still-image frames and generated five stored events. The moving sample API run after tracker refinement completed with 129 processed, 21 dropped, 1,021 detections, and 29 stored events. In a separate three-run, full-150-frame comparison, the original greedy IoU tracker emitted 33 enter events per run and the motion-predicted candidate emitted 29; this is not an identity-accuracy result. Two low-confidence `skis` predictions remain.
- The queue-capacity-150 tracker comparison preserved every input frame but accumulated a 3.5–4.6 s p95 end-to-end delay. Keep queue capacity 1 for the live freshness path until scheduling is profiled; the larger queue was used only to compare tracker behavior on identical frames.
- ROI enter/leave transitions require the configured number of consecutive processed frames inside/outside. Dwell emits once per confirmed visit. Current occupancy counts confirmed tracks; when a track expires while inside, the count decreases but no synthetic leave event is emitted. The board event run used a queue of 150 for reproducible frame-by-frame behavior and is not a live latency result.
- During repeated-still queue probes, the worker processed 88/150 and 91/150 frames at capacity 1. One run recorded shared NPU/IOMMU interrupt counts +5,096 during inference and 0 change during a two-second idle check. In the dynamic run, a later same-boot read was lower than the captured sample, so IRQ evidence remains exploratory. `/dev/rknpu*` remains absent, per-core utilization is not exposed, and devfreq stayed at 300 MHz.
- systemd installation and a LAN-bound API are P3 tasks and require review of their exact unit/configuration before activation.

## Prepared user-level systemd unit

[`deploy/systemd/rk-edge-station@.service`](deploy/systemd/rk-edge-station@.service) is a per-user unit template. It runs the service as the logged-in board account, binds the UI/API through the application's existing loopback listener, restarts the service after a failure, uses a private `/tmp`, and allows writes only under `%h/.local/state/rk-edge-station`. [`deploy/systemd/demo.env.example`](deploy/systemd/demo.env.example) lists the required model, source, state, port, ROI, queue, and timing values. Copy it to `~/.config/rk-edge-station/demo.env`, replace the model/source values, protect it with mode 600, and create `~/.local/state/rk-edge-station` before activation. For an RTSP URL containing credentials, protect the environment file; the health API redacts URL details, while the process command line can still be visible to users with permission to inspect that process.

The unit passed `systemd-analyze verify` and a temporary `systemctl --user` recovery run on the RK3576. SIGKILL of the service main process triggered a restart; SIGKILL of the worker caused the Python service to exit nonzero and triggered a second restart. The test instance was stopped, its copied unit and environment file were removed, and it was not enabled for boot. Review the exact unit and environment configuration before permanent activation. A user unit starts with the user's systemd manager; boot-time startup without login would require a separate lingering decision.

## Deployment evidence

Every deployable release will record source commit, compiler/CMake versions, dependency versions, model SHA-256, command line, configuration, board identity, and run log. Build success alone is not inference success.
