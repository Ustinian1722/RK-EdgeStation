# P3 reliability and release acceptance

**Status: In progress.** The RTSP recovery path, worker-failure supervision, and temporary systemd user-service recovery have board evidence. The 24-hour soak has not completed; filesystem exhaustion and persistent boot activation remain untested.

## Passed on RK3576

- Manual graceful service restart preserved SQLite events and evidence references across two runs. See [restart recovery evidence](evidence/p3/restart-recovery-smoke-20261105/summary.json).
- Synthetic RTSP disconnect/reconnect passed through the WSL GStreamer test server and loopback SSH reverse tunnel. The service API showed `reconnecting` during loss and `connected` at `stream_epoch=1` after recovery. See [RTSP report](evidence/p3/rtsp-reconnect-20261105/summary.md).
- The candidate worker and service retained local-file behavior: 50/50 frames, 6 events, 0 drops, clean worker exit, and SQLite integrity `ok`. See [local-file regression](evidence/p3/local-source-regression-20261105/summary.md).
- A temporary `systemctl --user` unit restarted after both service-main failure and worker failure. The test unit was removed afterward and was not enabled for boot. See [systemd recovery report](evidence/p3/systemd-auto-restart-20261105/summary.md).
- Failure-path, SQLite-outage, and bounded-write-failure simulations remain documented in [test results](TEST_RESULTS.md); the bounded write test does not simulate a full filesystem.

## 24-hour board run in progress

- Started: `2026-11-05T14:12:38Z`.
- Source commit recorded by the runner: `640d6f3`.
- State directory: `.work/p3-soak-640d6f3-20261105` on the board.
- Latest saved sample: `2026-11-05T16:02:42Z`, 6,603.13 seconds elapsed (about 1 hour 50 minutes).
- Latest counters: 33,000 processed, 33,007 decoded, 7 dropped, 6 events, 0 SQLite/storage failures.
- Latest resources: worker RSS 117,192 KiB, temperatures 26.85–27.77 °C, `MemAvailable` 7.56 GB, root free space 46.65 GB. NPU-load telemetry is unavailable.
- Read-only database check at `2026-11-05T16:03:20Z`: SQLite integrity `ok`, 6 events, 0 missing evidence references.

The soak continues on its original loaded worker binary and the source commit above. The RTSP and service-recovery changes in `637441b` have short board RTSP and local-file integration checks, but no 24-hour run. Do not attribute this in-progress baseline soak to those changes. Preserve the current run through its final SQLite and evidence-path checks, then run the release-candidate soak if acceptance requires exact-commit validation.

## Open before calling P3 complete

1. Let the active soak reach 24 hours and inspect its final summary, SQLite integrity, and every event evidence reference.
2. Run a 24-hour soak on the final pushed commit if the release gate requires exact-commit validation.
3. Exercise the application on a physical camera and confirm its codec, authentication, and network behavior. The current RTSP proof is synthetic JPEG/RTP over TCP.
4. Decide whether a user-level service is sufficient. The tested unit is not enabled at boot; boot startup without login would need a separate user-lingering decision. Review the unit and actual source/model environment before persistent activation.
5. Keep actual filesystem exhaustion open. Existing SQLite recovery evidence uses an unavailable path and a process-local file-size cap; the board filesystem was not filled.

The unit template and environment example are in [`deploy/systemd`](deploy/systemd/rk-edge-station@.service). The service continues to bind to loopback by default. Full P3 acceptance and release handoff remain open until the required gates above are resolved.
