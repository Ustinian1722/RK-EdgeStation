# Release checklist — v0.1.0 experimental release

This project is an experimental open-source engineering prototype, not a safety-certified industrial product.

## Scope

The first public version is a source release with historical board evidence. A new physical-board acceptance run, a current dashboard screenshot, a video demonstration and a long-duration soak are not required for this release.

## Completed

- [x] Consolidate the single-stream, DMA-BUF and multistream implementation.
- [x] Preserve reports and experimental evidence; separate final and superseded results in the benchmark index.
- [x] Bilingual README, architecture diagram, model-aware quick start, benchmark index and changelog.
- [x] Explicit limitations: connection count versus actual inference capacity; partial low-copy versus full zero-copy; developmental accuracy versus independent evaluation.
- [x] Host CI succeeded twice on candidate a6a5b84 (runs 43707550142 and 43707541268).

## Before changing visibility

- [ ] Review automated history audit results, including historical refs and any findings; review anything the scanner cannot inspect. This is a best-effort screening, not proof of absence of secrets.
- [ ] Confirm permission to publish project-authored source and retained third-party assets under their respective terms. Model weights, original datasets/videos and proprietary SDKs are not distributed.
- [ ] Publish the approved main branch and version notes. Changing visibility exposes existing branches, tags and Git history, not only the latest main tree.

## Optional follow-up (not release blockers)

- Fresh native RK3576 regression, physical-camera tests, long-duration testing and independent dynamic PPE event evaluation.
- A new demo video, dashboard screenshot and production documentation.
- Higher inference throughput, multiple RKNN contexts and additional model families.

The historical results in the benchmark index were collected on prior commits. The host checks do not validate native RKNN inference on this exact release.
