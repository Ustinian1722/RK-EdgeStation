# Open-source and upstream review

Reviewed 2026-11-05. Commit IDs below were read from public Git refs or GitHub metadata on that date. Imported files retain their upstream attribution; exact paths and scope are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

| Source | Revision / license | Findings | Project decision |
|---|---|---|---|
| [airockchip/rknn_model_zoo](https://github.com/airockchip/rknn_model_zoo) | `v2.0.0` `9bed8a56de595046651d20fa6e107884f2011e65`; Apache-2.0 | Release 2.0.0 adds RK3576. Its C++/C API examples are the official first-demo source. YOLOv8 C++ postprocess/types are imported with original license headers retained. The local integration redirects diagnostic `printf` output to stderr so worker stdout remains JSONL. Current `v2.5.0` requires RKNPU2 SDK >=2.5.0. | Use the v2.0.0-compatible YOLOv8n output graph with board Runtime 2.0.0b0. Keep imports scoped to postprocessing/types and retain attribution. Capture, tracking, events, service layers, and the logging integration patch are project work. |
| [airockchip/rknn-toolkit2](https://github.com/airockchip/rknn-toolkit2) | `v2.0.0-beta0` `77b71094e08391c543d9c65fea5f7cf98cc16eee`; `v2.5.0` `2c723177a94fac65cbc036d722907ddbc91980cf`; Rockchip RKNN SDK License | Supports RK3576. The license is a vendor agreement, not a general OSI open-source license; it limits use to development/testing of compatible Rockchip applications and controls documentation/software redistribution. v2.0 beta wheels are Linux x86-64 (Python 3.6–3.11). The v2.0.0-beta0 tree includes a preconverted RK3576 MobileNet sample model, which passed the board C API smoke test. | The user authorized Toolkit2 use under its license. Toolkit2 2.0.0b0 was installed in the isolated WSL x86-64 environment and converted the local YOLOv8n model for RK3576. SDK wheels, runtime binaries, documentation, and weights remain outside this repo. |
| [ruisv/rcdl](https://github.com/ruisv/rcdl) | `main` `6e03b0714fd8fae3170373409b5f4fb4f3329028`; Apache-2.0 | Modular C++17/Python facade over RKNN, RGA, and MPP; documents frame ownership and accelerator-specific limits. Its README says RK3588 requirements and explicitly says RK3576/RK356x are not yet verified, despite repository metadata mentioning those chips. | Read architecture and test organization only. Do not adopt its “zero-copy” or performance claims for this K7; do not import it as a runtime dependency until RK3576 validation and API/license review. |
| [rockchip-linux/mpp](https://github.com/rockchip-linux/mpp) | `develop` `9b56348cc61393c84662e6d6dd78feae67773a36`; GitHub license detector returns none. Board Debian copyright file declares Apache-2.0 for upstream files and GPL-2+ for packaging; current source has per-file SPDX notices. | Rockchip MPP provides a unified media API, but supported codecs and behavior depend on chip and kernel drivers. Recent source history has mixed per-file licensing; repo-level metadata alone is insufficient. | Link to the installed MPP library and API headers. Do not copy MPP source into RK-EdgeStation until each file's license and provenance are checked. Do not redistribute the board library. |
| [airockchip/librga](https://github.com/airockchip/librga) | `main` `2b32edcb97b601b25683e2941d888c8515da6d55`; Apache-2.0. Board package copyright file declares Apache-2.0 for upstream files. | RGA exposes 2D scale, rotation, blend, and color operations; API/driver compatibility is version-specific. Board has `librga` 2.2.0 and `/dev/rga`. | Use the installed library and public API; add a RGA backend only after format/stride operations are tested on RK3576. No prebuilt vendor library will be committed. |

## Community cross-checks

- Reddit's [RK3576 vendor-stack discussion](https://www.reddit.com/r/RockchipNPU/comments/1ut8p1x/got_the_vendor_rknllmrknn_stack_running_on/) describes kernel/runtime alignment and driver-clock issues. This is anecdotal context, not a source of hardware facts or a reason to modify this board.
- The [MPP device-node discussion](https://www.reddit.com/r/RockchipNPU/comments/1lwrx9x/listing_of_dev_mpi_device_nodes/) highlights that kernel image and MPP service nodes affect userspace behavior. Our own P2 probe confirmed that the benchmark process held `/dev/mpp_service` while `mppvideodec` decoded the test clip; it does not establish an end-to-end worker gain or zero-copy behavior.
- Reddit reports do not establish compatibility for this KICKPI K7. Board observations and official documentation remain the acceptance evidence.

## Pretrained model license boundary

- The Model Zoo YOLOv8 example names `airockchip/ultralytics_yolov8` as its model source. That upstream describes AGPL-3.0 and Enterprise options. The `.onnx` and converted `.rknn` weights are local validation artifacts and are not committed. A commercial or redistributed release needs a model license review or a different model; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## License and redistribution policy

- RK-EdgeStation's own source is Apache-2.0.
- Keep Rockchip SDK/Runtime, toolchain wheels, model weights, sample videos, and private user models out of Git. Record each model's upstream license before redistributing it.
- Use dynamic system libraries where possible. Include third-party notices for code actually copied into this repository.
- Never treat public availability of a binary as permission to redistribute it.
