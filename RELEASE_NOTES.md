# RK-EdgeStation v0.1.0

Initial source release of a hardware-accelerated industrial edge vision platform on Rockchip RK3576.

- C++17 heterogeneous media pipeline: GStreamer/MPP, RGA and RKNN.
- DMA-BUF FD import, shared RKNN input memory and explicit cache synchronization.
- FP/INT8 model optimization, tensor parity checks and stage-level profiling.
- Bounded queue scheduling, object tracking, PPE/ROI event rules, REST/Web and SQLite.
- Bilingual documentation, architecture diagram, benchmark records and automated host tests.

Selected historical board measurements: PPE throughput 4.59 → 7.98 FPS after INT8 conversion; explicit NV12 ingress copy 1.38 MB → 0 B/frame in the optional DMA-BUF path. See the repository benchmark index for complete protocols and results.

Apache-2.0 project source; external models and runtime components have separate terms.
