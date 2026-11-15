# Benchmarks and evidence

All measurements here are historical board evidence from the linked reports. Do not combine different models, run modes or source topologies as a controlled comparison.

| Experiment | Primary report | Retained results |
| --- | --- | --- |
| CPU/MPP/RGA, FP/INT8, queue policy | [Flagship](../FLAGSHIP_OPTIMIZATION_REPORT.md) | [Offline](../evidence/flagship/offline-final/summary.json), [Media-clock](../evidence/flagship/media-clock-final/summary.json), [Queue](../evidence/flagship/realtime-queue-final/summary.json) |
| PPE quality/performance | [Flagship](../FLAGSHIP_OPTIMIZATION_REPORT.md) | [FP/INT8](../evidence/flagship/ppe-int8/summary.json), [PPE acceptance](../PPE_ACCEPTANCE_REPORT.md) |
| Partial DMA-BUF low-copy | [DMA-BUF](../ZERO_COPY_BENCHMARK_REPORT.md) | [Three-run media-clock](../evidence/dynamic-video/worker-ablation/zero_copy_ablation_summary.json), [Offline FIFO](../evidence/dynamic-video/offline-fifo-ablation/zero_copy_ablation_summary.json) |
| Released PPE models | [Model selection](../MODEL_SELECTION_REPORT.md) | [Constructelligence evaluation](../CONSTRUCTELLIGENCE_PPE_SCALING_REPORT.md) |
| Multi-camera scale | [32-source report](../MULTISTREAM_32_REPORT.md) | [Summary](../evidence/multistream-32-20261109/summary.csv), [Resource curves](../evidence/multistream-32-20261109/resource-curves.svg) |
| Recovery | [P3 acceptance](../P3_ACCEPTANCE_REPORT.md) | [RTSP](../evidence/p3/rtsp-reconnect-20261105/summary.md), [systemd](../evidence/p3/systemd-auto-restart-20261105/summary.md) |

## Interpretation rules

- Media-clock and unpaced FIFO are separate modes. Offline backlog age is not real-time latency.
- Use the final bidirectional-sync DMA-BUF results; earlier `*-from-device-sync` folders are superseded.
- Use the clean 18-run multistream matrix; [confounded-prior](../evidence/multistream-32-20261109/confounded-prior/NOTE.md) is excluded due to a concurrent residual Worker.
- CPU percentages are one-core equivalents unless explicitly normalized; process RSS does not include every DMA-BUF allocation.
- NPU busy utilization is unavailable, not inferred from frequency or IRQs.
- PPE development-set and small 34-image test results are not independent dynamic event accuracy.
- 32 RTSP connections do not establish 32-camera real-time PPE coverage.

For new results record source commit, binary/model/video hashes, exact flags, software versions, repetitions and raw metrics. See [protocol](../BENCHMARK_PROTOCOL.md).
