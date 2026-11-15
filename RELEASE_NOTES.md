# RK-EdgeStation v0.1.0

Experimental source release for the KICKPI K7 / Rockchip RK3576.

- C++17 MPP/RGA/RKNN media and inference pipeline with OpenCV reference backends.
- Optional partial DMA-BUF/shared-input optimization and FP/INT8 benchmarking.
- Single-stream and multistream bounded scheduling, PPE/ROI event logic, local REST/Web and SQLite.
- Bilingual documentation, architecture diagram, quick start, benchmark index and host CI.

Historical board results are reported with their exact scope. 32 synthetic RTSP sources connected successfully, but the shared FP model processed about 4 aggregate FPS (not 32-camera real-time PPE). DMA-BUF is partial low-copy, not a full zero-copy guarantee. Independent dynamic PPE accuracy and production safety certification are not established. New board acceptance and demonstrations are optional. See the README and third-party notices. External model weights and original media/datasets are not bundled.
