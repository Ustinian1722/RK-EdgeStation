# 面向 Rockchip / RKNN / C++ / Linux 端侧 AI 岗位的项目讲解

状态：RK3576 单路视频闭环已打通；C++17 Worker 可切换 OpenCV/FFmpeg 与 GStreamer/MPP、CPU 与 RGA，并接入 RKNN、跟踪、PPE/ROI 事件和本地 SQLite/REST/Web。MPP DMA-BUF → RGA → RKNN 共享输入作为可选部分低拷贝模式保留。2026-11-07 对两款已发布 PPE 模型进行了同一开发验证集对照；它们都明显弱于现有 YOLOX-S，因此生产后端未替换。动态验证促成了 Person-only ROI 和按 Person Track ID + 违规类别冷却两项修复；误报、未知标签和短视频丢帧仍限制结论。

## 项目解决什么问题

把单块 RK3576 开发板做成独立边缘视觉终端：视频进入设备后在本地完成检测、跟踪、区域规则判断和事件留证，再由本机页面查看运行状态和结果。目标是验证整条设备链路，而不是把一个模型 demo 包装成产品。

## 当前硬件事实

审计设备为 KICKPI K7 / RK3576，Ubuntu 24.04.3、Linux 6.1.75、aarch64、8 GiB RAM。板上 RKNN Runtime 2.0.0b0 / driver 0.9.7，GStreamer 1.24.2、Rockchip MPP plugin 1.14.4、MPP/RGA 组件已在 Worker 实测。Toolkit2 2.0.0b0 在隔离的 x86-64 WSL 中转换模型；推理、视频、事件与 API 都在板端。`/dev/rknpu*` 不可见，无法测逐核 NPU 利用率；没有识别到物理摄像头。

## 架构要点

- 将视频源、CPU参考处理、MPP/RGA、RKNN、跟踪/ROI事件、调度、数据库/API分层；GStreamer/NV12 与 RGA 为真实 Worker 可选后端，不是单独 benchmark。
- 有界队列和新旧帧策略保护内存与事件新鲜度。
- 把推理时延、端到端延迟、帧龄、吞吐和丢帧分开计量。
- 每个事件关联帧、时间、置信度、轨迹/区域、规则和模型版本。
- 只在实测后宣称硬件加速或低拷贝效果。
- 主机 WSL 只承担可选模型转换；模型运行、视频处理、事件与 API 均在 RK3576 板端。

## 面试追问准备

1. **怎样证明模型跑在 NPU 上？** 当前能证明的是：板端 RKNN C Runtime 成功加载 640×640、9 输出的模型并完成动态视频检测与 API 闭环；Runtime 报告 driver 0.9.7，RKNPU driver/devfreq 可见。一次重复静态帧探针的共享 NPU/IOMMU IRQ 增加 5,096，空闲两秒不变；但动态运行采样后，同一启动周期内的计数又低于已采样值，说明 IRQ 证据存在未解释的不一致。`/dev/rknpu*` 不存在，频率采样始终为 300 MHz，也没有可用的利用率计数器，因此不声称已测得 NPU 利用率、逐核负载或动态升频。
2. **如何衡量优化？** 固定模型、视频、分辨率和频率条件，重复测量推理、预处理、后处理和端到端时延；报告原始样本、分位数、CPU/NPU和丢帧率。
3. **为什么有很多 `/dev/video*` 仍不能说有摄像头？** Rockchip 的 CIF/ISP/VPSS 也会创建平台节点；要核对 media graph 中是否有真实 sensor，并成功开流。
4. **为什么不直接用 RCDL 的零拷贝？** 上游 README 明确 RK3576/RK356x 尚未验证。项目需要在本机检查 buffer allocator、fd生命周期、stride和cache同步，再测输出和延迟。
5. **INT8怎样验收？** 记录校准样本、预处理、量化配置和模型散列，与源模型逐帧匹配输出/检测框，并报告数值误差、漏检变化和任务指标。
6. **如何处理队列过载和断流？** 队列限长、明确FIFO/丢旧帧策略、超时与流epoch；断流后关闭旧流资源，重连后拒绝旧epoch迟到结果。
7. **有哪些实测成果？** 2026-11-07：在 RK3576 对同一 150 帧 10-FPS H.264 输入完成 A0–A3 和媒体时钟对照；MPP/RGA FP 为 7.98 FPS，OpenCV/CPU FP 为 7.66 FPS，COCO INT8 为 9.99 FPS。PPE YOLOX-S INT8 在 15-FPS 正样本回放中为 7.98 FPS，FP 为 4.59 FPS；200 张开发验证集的 AP50:95 为 46.77%/45.76%，`no_vest` 召回为 73.08%/70.77%。该验证集参与 FP checkpoint 选择，不能称独立测试。完整阶段数据和边界见 [旗舰报告](FLAGSHIP_OPTIMIZATION_REPORT.md)。
8. **RK3576 的 DMA-BUF 路径做到了什么？** GStreamer `mppvideodec` 普通 NV12 caps 下实际给出 MPP allocator 的有效 FD；RGA 能导入该 FD，同时也能导入 RKNN `rknn_create_mem` 的输入 FD。RGA 直接把预处理结果写入已绑定的 RKNN 输入缓冲。Z2 移除 Worker 的 1.3824 MB/帧 NV12 CPU 行复制，板端输入、三路 RKNN 输出和检测结果在固定帧重复实验中完全一致。不能据此声称 Runtime/驱动内部没有复制，也没有证明所有设备缓存流量归零。
9. **这条优化值得设为默认吗？** 暂不。三次 15-FPS media-clock 测试中，Z2 中位 FPS 为 8.046，Z0 为 7.937；P95 为 184.9/186.0 ms，单核等效 CPU 为 52.3%/56.2%，RSS 为 100.5/107.5 MiB。FPS 增益约 1.4%，P95只小幅变化；保留为可选低拷贝模式，Z0仍是稳定默认。
10. **动态 PPE 验证说明了什么？** 保留的历史 pre-fix run 在 34 秒施工视频上触发 4 个 PPE 告警，人工复核为 3 个明显误报、1 个状态不明；ROI 将装备框也计入人员事件，按违规框轨迹去重还会被 ID churn 绕过。当前代码已修复这两点。相同视频的新板端 run 触发 2 个告警（1 个可见背心误报、1 个遮挡头盔未知），ROI 进入事件 10 个且全为 Person。另有一段 CC0 短片，前景人员的头盔/背心违规可见，但 79 帧中 Latest 队列处理 23、丢弃 56。样本很少且标签稀疏，不能计算部署精度或召回。
11. **为什么没有换成更成熟的公开 PPE 模型？** 在同一 200 张开发验证图上统一 class map、score 0.25、NMS IoU 0.45 后，当前 YOLOX-S FP 的五类宏平均 AP50/AP50:95 为 0.7938/0.5376。Constructelligence v0.3 FP RKNN 为 0.5886/0.3393，Abdelrahman v1.0.0 ONNX 为 0.5762/0.3025。Constructelligence 的 INT8 在 RK3576 上从约 3.39 FPS 提高到 5.20 FPS，但其输出量化让 PPE 类别分数变成零；准确率不可接受，所以保留 YOLOX-S。该验证 split 参与过 YOLOX FP checkpoint 选择，不能称独立测试。详见 [模型选型报告](MODEL_SELECTION_REPORT.md)。

## 三条可用于实习简历的项目描述

1. **RK3576 异构视频 Worker：**基于 C++17 集成 OpenCV/FFmpeg 与 GStreamer/MPP NV12 输入、RGA 预处理及 RKNN 推理，处理 stride、PTS、buffer 所有权、受限队列与 RTSP epoch 重建；输出连接跟踪/PPE/ROI 事件、SQLite 与本机 API/Web。
2. **PPE 模型量化与部署：**在 RTX 5060 训练 YOLOX-S PPE，并用 RKNN-Toolkit2 转换、在 RK3576 运行 FP/INT8；基于训练集 50 张类别覆盖校准，板端 INT8 达 7.98 FPS，FP 达 4.59 FPS，同时量化了 AP50:95 与关键违规类别召回变化。
3. **异构性能实测：**建立同视频、同模型的三次 A0–A3、媒体时钟与 Latest/FIFO 对照，记录阶段 P50/P95、帧龄、丢帧、CPU、RSS、视频/模型/Worker SHA，并做输入/输出与检测一致性检查；MPP/RGA FP 的收益小于 INT8，保留负结果。

面试中应主动说明：PPE AP 和 INT8 差异来自参与 FP checkpoint 选择的 200 张开发验证图；旧的 2/2 回放是两个重复静态场景，不代表动态时序准确率。动态施工视频和短 CC0 片段只支持定性排障，不能给出独立事件精度。公开候选的板端模型探针是无节流、CPU 预处理的模型级实验；不要把它说成 Worker 吞吐。Z2 是部分低拷贝：减少显式 NV12 CPU 行复制并让 RGA 写入共享输入 FD，但运行时内部迁移、缓存维护和所有端到端复制次数仍不可观测。10/15-FPS 单路文件实验也不等于多摄像头或真实现场保证。
