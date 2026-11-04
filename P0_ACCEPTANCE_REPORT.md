# P0 阶段报告

日期：2026-11-05。项目目录：`D:\RK-EdgeStation`。板端工作区：`/home/kickpi/projects/RK-EdgeStation`。

## 验收结论：通过（官方 RKNN 冒烟测试）

官方 RKNN C++ 样例已在 KICKPI K7 / RK3576 上本机编译，并使用板端现有 Runtime 完成一次 MobileNet 推理。板端没有安装 Toolkit2 Python 包，也没有改动系统、驱动、内核或 runtime。

| 项目 | 实测结果 |
|---|---|
| SDK 示例 | Toolkit2 `v2.0.0-beta0`，commit `77b71094e08391c543d9c65fea5f7cf98cc16eee`，RK3576 官方 `rknn_create_mem_demo.cpp` |
| 编译位置 | 板端，G++ 13.3；链接既有 `/lib/librknnrt.so` |
| 模型 | SDK 随附的 RK3576 `mobilenet_v1.rknn`，SHA-256 `bc66943ea85ec0dd8a04da22c4276bfc8a4c6fe24f5ea8be7a1e5c3c22c8259d` |
| 输入 | SDK 随附 `dog_224x224.jpg`，SHA-256 `c350299c6283d5f62fecf1f845b6b3be9aafec8dff528ca09a129990f0a584b0` |
| runtime / driver | Runtime `2.0.0b0` (`35a6907d79@2024-03-24T10:31:14`)，运行时报告 driver `0.9.7` |
| 推理结果 | API 返回成功；Top-1 score `0.927734`，class index `156` |
| 样例计时 | 单次样例记录 `11.06 ms` / `90.39 FPS`；不是视频端到端 FPS，也不是稳定性能基准 |
| 原始证据 | [official-rknn-demo.log](evidence/p0/official-rknn-demo.log)、[运行元数据](evidence/p0/official-rknn-demo-meta.txt) |

该结果证明当前板端 runtime 可以通过 RKNN C API 完成官方 RK3576 模型推理。初始硬件审计中 `/dev/rknpu*` 节点不可见；本次 API 推理仍成功，运行时接口实际访问路径尚未追踪。未为此调整节点、权限或系统服务。

## 已完成并有直接证据

1. **本机和板端实审**：确定 Windows 11 x64 主机、KICKPI K7、RK3576、Ubuntu 24.04.3、Linux 6.1.75、aarch64、8 GiB RAM，以及 CPU、频率、存储、权限、runtime、MPP/RGA/GStreamer/V4L2 现状。原始只读输出保存在 [evidence/p0/hardware-audit.txt](evidence/p0/hardware-audit.txt)。
2. **SSH 安全连接**：使用项目专用 Ed25519 公钥；私钥在仓库外；`authorized_keys` 目录/文件权限分别为 700/600，批处理模式公钥连接已验证。
3. **开源调研**：审查 RKNN Model Zoo、Toolkit2、MPP、librga、RCDL 的源版本和许可；参考与适用边界见 [OPEN_SOURCE_REFERENCES.md](OPEN_SOURCE_REFERENCES.md)。
4. **需求和架构冻结**：将用户要求拆成六个子系统、阶段目标、硬件边界和验收规则，见 [REQUIREMENTS.md](REQUIREMENTS.md)、[ARCHITECTURE.md](ARCHITECTURE.md)。
5. **产品文档基线**：创建构建、模型转换、Benchmark、优化、测试、结果和面试文档；未实测功能均明确标为未验证。

## 边界和下一阶段门槛

- 当前通过的是官方 MobileNet 分类冒烟测试。项目自己的 YOLO 检测模型、前处理、解码、跟踪、事件逻辑和视频输入尚未实现/验收。
- MPP/RGA 目前只确认软件、设备节点和 GStreamer decoder factory 存在；真实解码、缩放和持续吞吐测试仍在 P1/P2。
- 没有识别到物理摄像头。P1 先以本地 MP4 为输入；MIPI/USB 摄像头未通过验收。
- 本项目可在 RK3576 板上完成应用编译和运行。Toolkit2 仅用于生成或转换项目模型；如果拿到许可允许使用的预转换 RKNN 模型，主机无需 Rockchip 硬件或 RK SDK。当前官方 smoke test 已采用板端预转换模型。

P0 的官方 NPU 示例门槛通过，可以进入 P1。不得因本次样例通过而宣称产品精度、持续 FPS、零拷贝或 24 小时稳定性。
