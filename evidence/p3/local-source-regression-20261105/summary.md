# RK3576 local-file regression after RTSP changes

The candidate worker and updated Python service passed a 50-frame local AVI smoke on RK3576 after RTSP support and reconnect-state reporting were added.

- Input: `p3-static-frame75-5fps-raw.avi`, 640×480 at 5 FPS.
- Model: `yolov8n-rk3576-fp.rknn`.
- Result: 50/50 frames decoded and processed, 350 detections, 6 stored ROI events, 0 dropped frames. Metrics reported `source_type=local-file`, `source_status=not-applicable`, `reconnect_attempts=0`, and `stream_epoch=0`.
- API: worker status completed with exit code 0; storage was available with no write failures.
- SQLite integrity: `ok`, 6 events.

See [health](health.json), [events](events.json), and [board service log](board-service.log).
