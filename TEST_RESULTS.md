# Test results

Status: P0 official RKNN smoke test passed; updated 2026-11-05.

| Check | Result | Evidence |
|---|---|---|
| Host inspection | Pass: Windows 11 x64, Git and GitHub CLI available; Python 3.12/3.13; no host CMake or C++ compiler; no WSL distribution registered | P0 command output and environment inspection |
| SSH network reachability | Pass: board reachable at configured LAN address | OpenSSH connection |
| Initial password login | Pass; password was entered interactively and was not written to a command, source, repository, config file, or audit log | SSH session |
| Public-key setup | Pass: task-specific Ed25519 key installed at board account with `.ssh` mode 700 and `authorized_keys` mode 600; batch-mode key login succeeded | `ssh -o BatchMode=yes ... id`; private key is outside repository |
| Board identity/hardware inventory | Pass: KICKPI K7 / RK3576 / Ubuntu 24.04.3 / Linux 6.1.75 / aarch64 / 8 GiB | [hardware-audit.txt](evidence/p0/hardware-audit.txt) |
| MPP/RGA software/device inventory | Pass for presence only: MPP and RGA packages/libraries/nodes; GStreamer MPP decoder factory | [hardware-audit.txt](evidence/p0/hardware-audit.txt) |
| Official RKNN API demo build | Pass: official RK3576 C++ sample compiled natively with G++ 13.3 and linked to installed Runtime 2.0.0b0 | [official-rknn-demo-meta.txt](evidence/p0/official-rknn-demo-meta.txt) |
| Official RKNN inference | Pass: context/model loaded, `rknn_run` completed, runtime reported driver 0.9.7, Top-1 score 0.927734 / class 156 | [official-rknn-demo.log](evidence/p0/official-rknn-demo.log) |
| Official sample latency | One sample iteration reported 11.06 ms (90.39 FPS); smoke-test observation only | [official-rknn-demo.log](evidence/p0/official-rknn-demo.log) |
| Physical camera input | Not available in this audit: no camera sensor/USB source identified; `/dev/video0` is not usable | [hardware-audit.txt](evidence/p0/hardware-audit.txt) |
| `/dev/rknpu*` visibility | Not present in the initial audit or after smoke test; RKNN API still succeeded. Runtime access path not traced and no system changes made | [hardware-audit.txt](evidence/p0/hardware-audit.txt), [official-rknn-demo.log](evidence/p0/official-rknn-demo.log) |
| Project detection model | Not tested; official smoke test used a bundled MobileNet classifier model | [MODEL_CONVERSION.md](MODEL_CONVERSION.md) |
| MPP/RGA actual operation | Not tested; presence only | [hardware-audit.txt](evidence/p0/hardware-audit.txt) |
| Project build and automated tests | Not run; implementation has not started | P1 work |

No project throughput, video FPS, detection accuracy, zero-copy, stability, or optimization result is claimed. The single official sample timing is not a product benchmark.
