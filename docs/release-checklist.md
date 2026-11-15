# Release gates — v0.1.0 RC1

This is a release candidate, not a published or safety-certified product.

## Completed in the candidate

- [x] Start from the latest multistream branch containing main and DMA-BUF research.
- [x] Preserve historical reports and evidence, with an index separating final and superseded runs.
- [x] Bilingual README, architecture, model-aware quick start, benchmark index and changelog.
- [x] Distinguish connection count, inference throughput, explicit input copies and model accuracy.

## Required before public release

- [ ] Review the complete Git history and all branches/tags for secrets, SSH details, RTSP credentials, private paths, logs and unpublished assets. Rotate any exposed secret; deletion from the latest tree is not history removal.
- [ ] Review tracked historical `.env` evidence, screenshots, video-derived evidence and compressed logs; redact or remove unauthorized material while preserving reproducibility.
- [ ] Verify all third-party source, weights, dataset, media, SDK and derivative-asset terms. Do not redistribute restricted weights. Check AGPL obligations separately from Apache-2.0 source.
- [ ] Run host tests and native RK3576 build/CTest/Python tests on the exact candidate SHA.
- [ ] Run single-stream smoke, RTSP reconnect, PPE event persistence and multistream smoke with authorized inputs; check SQLite integrity and teardown.
- [ ] Capture a current dashboard screenshot and short demo with authorized footage; document source and license.
- [ ] Verify README commands on a fresh clone and confirm all relevant model/setup instructions.
- [ ] Review PR, merge approved candidate, create release tag and notes, and explicitly authorize changing repository visibility.

Optional future work: independent dynamic PPE labels, real-camera/long-run validation, quantized multistream model and parallel-context scheduling experiments. These are not prerequisites for describing the existing research honestly, but are prerequisites for stronger production claims.
