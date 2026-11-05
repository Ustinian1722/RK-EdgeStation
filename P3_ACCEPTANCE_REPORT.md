# P3 reliability and release acceptance

**Status: Core recovery checks passed; remaining release limits are documented.** A 24-hour soak is not required for this project. The earlier exploratory run was stopped after 4 h 52 min; it is not a completed soak. Actual filesystem exhaustion, physical-camera behavior, and persistent boot activation remain untested.

## Passed on RK3576

- Manual graceful service restart preserved SQLite events and evidence references across two runs. See [restart recovery evidence](evidence/p3/restart-recovery-smoke-20261105/summary.json).
- Synthetic RTSP disconnect/reconnect passed through the WSL GStreamer test server and loopback SSH reverse tunnel. The service API showed `reconnecting` during loss and `connected` at `stream_epoch=1` after recovery. See [RTSP report](evidence/p3/rtsp-reconnect-20261105/summary.md).
- The candidate worker and service retained local-file behavior: 50/50 frames, 6 events, 0 drops, clean worker exit, and SQLite integrity `ok`. See [local-file regression](evidence/p3/local-source-regression-20261105/summary.md).
- A temporary `systemctl --user` unit restarted after both service-main failure and worker failure. The test unit was removed afterward and was not enabled for boot. See [systemd recovery report](evidence/p3/systemd-auto-restart-20261105/summary.md).
- Failure-path, SQLite-outage, and bounded-write-failure simulations remain documented in [test results](TEST_RESULTS.md); the bounded write test does not simulate a full filesystem.

## Exploratory long-run board sample (stopped)

- Started: `2026-11-05T14:12:38Z`; stopped: `2026-11-05T19:04:45Z` after 17,526 seconds (about 4 h 52 min).
- Source commit recorded by the runner: `640d6f3`.
- State directory: `.work/p3-soak-640d6f3-20261105` on the board.
- Final counters: 87,300 processed, 87,307 decoded, 7 dropped, 6 events, and 0 SQLite/storage failures.
- Final recorded resources: worker RSS 117,192 KiB, temperatures 26.85–27.77 °C, `MemAvailable` about 7.56 GB, root free space about 46.65 GB. NPU-load telemetry is unavailable.
- Final database check: SQLite integrity `ok`, 6 events, and 0 missing evidence references. The captured [runner summary](evidence/p3/soak-stopped-20261105-summary.json) records a clean service exit.

The requested 24-hour duration was not reached. The run used the older `640d6f3` source and model, so it does not represent the PPE release candidate. Its summary says `result=failed` because the duration gate was not met; the worker/service stopped cleanly and the final SQLite/evidence checks passed. Treat it as a bounded exploratory observation only, not a stability pass.

## Handoff demo artifact

A 36-second H.264 evidence-rendered demo is available locally at `.work/handoff-demo-20261105/RK-EdgeStation_RK3576_evidence-demo_20261105.mp4` (1280×720, 24 fps; SHA-256 `0FADEEF1A8E4818B4C82DAB513A4C2091A7EE490CFCF0E796A000F01F5E77A7A`). Its companion `manifest.txt` records the source files, sample time, metrics, and limitations. A ready-to-transfer archive is at `.work/handoff-demo-20261105/RK-EdgeStation_RK3576_demo-handoff_20261105.zip` (SHA-256 `C03DCF353852C1B818E951728B01671DB9C93DF3DC16E6D2ECF2E46A7732751D`); it contains the MP4 and manifest. The video combines board-saved event frames, the 16:17:42Z soak sample, and the separate synthetic RTSP test. It is not a raw UI screen capture; the fixed-frame input and synthetic RTSP source are labeled in the video. These generated files remain under ignored `.work` and are not committed to Git.

## Remaining release limits

1. Exercise the application on a physical camera and confirm its codec, authentication, and network behavior. The current RTSP proof is synthetic JPEG/RTP over TCP.
2. Decide whether a user-level service is sufficient. The tested unit is not enabled at boot; boot startup without login would need a separate user-lingering decision. Review the unit and actual source/model environment before persistent activation.
3. Keep actual filesystem exhaustion open. Existing SQLite recovery evidence uses an unavailable path and a process-local file-size cap; the board filesystem was not filled.

The unit template and environment example are in [`deploy/systemd`](deploy/systemd/rk-edge-station@.service). The service continues to bind to loopback by default. Full P3 acceptance and release handoff remain open until the required gates above are resolved.
