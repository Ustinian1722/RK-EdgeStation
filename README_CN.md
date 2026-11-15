# RK-EdgeStation

**面向 Rockchip RK3576 的硬件加速工业边缘视觉平台**

[English](README.md) · [快速部署](docs/quickstart.md) · [系统架构](docs/architecture.md) · [实验数据](benchmarks/README.md)

RK-EdgeStation 在单块 RK3576 上实现从视频接入、硬件解码、图像预处理、NPU 推理到工业事件管理的端侧链路。当前验证硬件为 KICKPI K7，不代表所有 Rockchip 芯片均已适配。

![系统架构](assets/architecture.svg)

## 核心能力

- C++17：MP4 / H.264 RTSP，OpenCV/FFmpeg 或 GStreamer/MPP 视频解码，RGA 预处理。
- RKNN：FP / INT8 推理，输入提交、推理、后处理与端到端分阶段性能统计。
- 低拷贝：可选 MPP DMA-BUF → RGA → RKNN 共享输入内存，保留普通拷贝默认路径。
- 多路调度：独立采集与重连、有限队列、Latest/FIFO 策略、加权公平调度。
- 业务：目标跟踪、劳保穿戴、区域进入/离开/停留、图像空间人车风险提示。
- 本地管理：REST API、Web、SQLite、事件图片、systemd 用户服务恢复。

## 板端实验结果

| 实验 | 已记录结果 |
| --- | --- |
| COCO 单路 10 FPS | CPU/FP 7.656 FPS → MPP/RGA/INT8 9.991 FPS，P95 236.6 → 106.3 ms |
| YOLOX-S PPE | FP 4.593 → INT8 7.975 FPS；开发验证集 AP50:95 46.77% → 45.76% |
| DMA-BUF 720p 动态视频 | 显式 NV12 输入拷贝 1,382,400 → 0 B/frame，预处理 P50 下降 25.6%，FPS 7.937 → 8.046 |
| 32 路接入 | 32/32 合成 RTSP 源成功连接，聚合推理 4.09 FPS，丢帧率 95.14% |

以上为已有报告的特定实验条件结果，并非本次发布重新测得。详见 [实验索引与证据](benchmarks/README.md)。

## 快速运行

在装有兼容 RKNN Runtime、OpenCV、MPP/RGA 和 GStreamer 开发环境的 RK3576 板卡上：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build --parallel 4
ctest --test-dir build --output-on-failure
bash scripts/run_demo.sh /path/to/model.rknn /path/to/input.mp4
```

在板端打开 http://127.0.0.1:8080。本仓库不分发模型权重、原始数据集、源视频或专有 SDK。模型类型、多路配置和远程访问方式见 [快速部署](docs/quickstart.md)。

## 能力边界

32 路接入不等于 32 路实时推理；DMA-BUF 路径不是全链路零拷贝；现有 PPE 数据不能证明真实工厂的独立事件准确率。明火/烟雾未集成可用模型，真实摄像头、长时间运行及可靠 NPU 占用率尚待验证。本项目不是经认证的工业安全系统。

源码采用 Apache-2.0；外部模型和数据另有许可，详见 [第三方声明](THIRD_PARTY_NOTICES.md)。当前为 v0.1.0 发布候选版本，尚待完成 [发布验收](docs/release-checklist.md)。
