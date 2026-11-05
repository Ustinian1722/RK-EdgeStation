# RK3576 systemd recovery check

**Result:** Pass for a temporary per-user systemd service. The unit was stopped and removed after the test; it is not enabled for boot.

## Configuration

- Unit: [`deploy/systemd/rk-edge-station@.service`](../../../deploy/systemd/rk-edge-station@.service), instance `rtsp-test`.
- Environment snapshot: [rtsp-test.env](rtsp-test.env). It points at the candidate worker and RK3576 model, uses loopback port 18085, and targets an intentionally unavailable local RTSP port 18555. No camera credentials were used.
- Unit verification: `systemd-analyze verify` on systemd 255 reported no diagnostic for this project unit. The board has unrelated warnings for pre-existing vendor units.
- Service manager: `systemctl --user`; no system-level unit or `sudo` was used.

## Failure and recovery sequence

1. Started `rk-edge-station@rtsp-test.service`. The API returned `status=running`, storage available, and the worker reported `source_status=connecting` while retrying the unavailable test URL.
2. Sent SIGKILL to the service main process (PID 32710). systemd restarted the service as PID 32747 after the configured 3-second delay; `NRestarts=1`.
3. Sent SIGKILL to the worker (PID 32749). The Python service detected the nonzero worker exit, returned status 1, and systemd restarted the unit as PID 32782; `NRestarts=2`. The new worker PID was 32784 and `/api/health` returned to running with storage available.
4. Stopped the service cleanly. The unit was not enabled, the instance environment file and copied unit were removed from the board, and no test service or worker remained.

A first setup attempt used the old default worker path, which was still needed by the 24-hour soak. That run failed its RTSP file-path check. The template now takes `WORKER_PATH` from the instance environment; the recovery sequence above used `.work/rtsp-build/bin/rk_edge_worker` and passed.

## Artifacts and limits

- [systemd journal](journal.log)
- [unit status after cleanup](status-after-stop.txt)
- [unit status after the second restart](status-after-worker-restart.txt)
- [health after the second restart](health-after-worker-restart.json)
- [enabled state](enabled-state.txt)
- [test environment](rtsp-test.env)

This verifies systemd restart after service-process failure and worker-process failure in a user session. It does not enable boot startup, user lingering, a system-level service, or a LAN listener. The board's loopback service policy remains in effect.
