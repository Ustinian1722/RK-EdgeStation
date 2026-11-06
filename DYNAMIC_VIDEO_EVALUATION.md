# Dynamic video and event evaluation

## Source and preparation

The test uses one real moving-camera construction-site video by Acabashi, sourced from Wikimedia Commons. The selected source file is `Building construction Moira Close Broadwater Farm Haringey 2025 16.webm`, recorded 12 May 2025 in Haringey, London, and licensed CC BY-SA 4.0. The clip metadata records the author, license, original SHA-256, source time range, FFmpeg command and derivative hash in [`manual-annotations.json`](evidence/dynamic-video/manual-annotations.json). The Commons page identifies the creator/license and requires attribution and ShareAlike for redistributed derivatives ([source and license](https://commons.wikimedia.org/wiki/File:Building_construction_Moira_Close_Broadwater_Farm_Haringey_2025_16.webm)).

The 34-second H.264 derivative covers source seconds 236–270 and is 1280×720 at 15 FPS (510 frames). The separate 8-second performance segment covers derivative seconds 18–26 (120 frames). All 510 frames have distinct decoded frame hashes, and visual samples show workers moving, carrying materials and becoming partly occluded. This is a London construction site, not a factory production line. No clip or model weight is committed; reports contain provenance, hashes and derived logs only.

## Manual annotation protocol and observations

The review was intentionally small: inspect frames at derivative seconds 0, 4, 8, 12, 16, 20, 24, 28 and 32, then inspect every PPE alert image emitted by the Worker run. Times below are derivative PTS; add 236 seconds for approximate source time. Only visible facts were labelled. Frames between those points were left unlabeled except for the system's raw track/event records.

| Interval (clip PTS) | Manual observation | Label confidence |
|---|---|---|
| 0–4.667 s | A visible person in the lower work area starts just to the right of the configured ROI boundary and is inside it by the Worker’s 4.667 s ROI-enter record. | Coarse entry interval; onset is bounded by sparse processed frames. |
| 14.7–15.467 s | A worker appears at the left edge of the lower work area and is inside the ROI by the 15.467 s processed frame. | Coarse entry; part of the body is near the image edge. |
| 18–20 s | The worker carrying material moves right; the same tracker ID’s box center is outside the ROI by 19.467 s and the 20 s visual sample shows the worker farther right. | Coarse exit interval, with a clear per-track enter/exit cycle in this zone. |
| 0, 4, 8, 12, 16, 20, 24, 28, 32 s | Visible workers wear hard hats and high-visibility clothing at these sampled points. | Positive visible PPE evidence at samples only; not dense frame truth. |
| 5–8 s and 20–29 s | Workers are partly hidden by the concrete wall/platform edge, stacked material or boards they carry. | Mark obscured/unknown where helmet or torso is not visible. |
| 0–34 s | No forklift is visible. | Forklift/person conflict cannot be evaluated on this clip. |
| 0–34 s | No visually confirmed continuous no-helmet or no-vest violation interval was found in this brief review. | PPE violation recall is not measurable from this clip. |

The exact per-alert manual notes are in [`manual-annotations.json`](evidence/dynamic-video/manual-annotations.json). They do not convert this small spot-check into an independent PPE accuracy estimate.

## RK3576 Worker run

The final event run used Worker SHA `617a2dc21d1bb4e3155c90ea1cbc16f23071011f3a61504253588fc587fdeab0`, PPE INT8 model SHA `88b10ac2711f4b00d2859dfa0e1089ccb34ebd78a73b7dc46bd51a56c92c502c`, MPP/GStreamer + RGA + retained MPP DMA-BUF + RKNN shared input, bidirectional Runtime cache sync, media pacing, `Latest` queue capacity 1, confidence 0.35, PPE confirmation 3 frames, 30 s cooldown, and ROI `0,0.45,0.30,0.55`. The Worker reached EOS without error and wrote event images plus a per-frame track log. This was a functional event/track review, not a performance run: evidence JPEG encoding and track JSONL writing were enabled.

| Worker output | Result |
|---|---:|
| Decoded / processed / dropped | 510 / 233 / 277 frames |
| Event-run effective throughput | 6.831 FPS |
| ROI events | 191 enter, 39 dwell, 5 leave (class agnostic) |
| PPE violation events | 4 |
| Danger-zone intrusion / forklift co-occurrence | 0 / 0 |
| Worker explicit inference ingress copy bytes | 0 B; event-evidence copies still occur and are excluded from this counter |
| Unique `Person` track IDs | 19 across 233 inference frames |

The generic ROI event layer consumes every tracked PPE-class detection. Of its 235 records, only 20 are labelled `Person`; the rest include helmet, vest, gloves, boots and violation boxes. This class-agnostic `RoiEventMachine` input makes aggregate ROI counts unsuitable as person-entry counts. One manually reviewed person track entered at 16.0 s and exited at 19.333 s while the worker carried material across the configured zone; those times fall within the coarse visual interval, but sparse Latest-frame inference does not support a precise delay estimate.

The tracker maintained long-lived IDs for two moving workers: Person track 16 had 196 observations from 4.733–33.933 s (maximum observed source-frame gap 3), and track 1 had 183 observations from 0–27.133 s (maximum gap 8). A third track was observed for 15 s. Nineteen Person IDs were created overall, including short tracks, indicating fragmentation around low-confidence detections, partial occlusions and Latest-frame gaps. This is a continuity diagnostic, not IDF1 or a ground-truth identity score. Parsed metric/event JSONL, track JSONL, and the complete Worker stdout (including librga diagnostic lines) are preserved under [`evidence/dynamic-video/event-run/raw`](evidence/dynamic-video/event-run/raw/).

## PPE alert review and known behavior

All four emitted PPE alerts were manually inspected at their corresponding evidence images. Three were visually inconsistent with the visible high-visibility clothing; one helmet alert was partly occluded and remains unknown:

| PTS (s) | Model alert | Confidence | Review |
|---:|---|---:|---|
| 1.667 | `no_helmet` | 0.6999 | Head/equipment partly obscured by wall and overlapping annotations; status unknown. |
| 6.600 | `no_vest` | 0.8659 | Visible high-visibility clothing; apparent false alert. |
| 12.133 | `no_vest` | 0.7782 | Visible workers wear high-visibility clothing; apparent false alert. |
| 15.200 | `no_vest` | 0.7652 | Visible high-visibility clothing; apparent false alert. |

This alert-only review found three apparent false alerts and one unknown result. It is not a full-video or deployment precision estimate, and recall cannot be estimated because the clip contains no confirmed violation interval. The three no-vest alerts are associated to the same person but use changing violation-box track IDs; because cooldown is keyed by that violation track, repeated alerts occurred within 30 seconds. The clip contains no human-confirmed violation to test true-positive behavior.

## Conclusion and limits

- Dynamic H.264 decode, RGA/RKNN inference, tracking output, ROI transitions, PPE event creation and evidence-image writing all executed on RK3576.
- The selected footage provides real motion, hard-hat/high-visibility samples, partial occlusion, a coarse person entry/exit interval and track-continuity diagnostics.
- The test found noisy class-agnostic ROI events, fragmented Person IDs, three apparent false PPE alerts and one unknown helmet alert. These are recorded as current business/model limitations; no attempt was made to “correct” them by relabelling predictions or retuning against this single scene.
- The clip has no forklift and no confirmed PPE violation. Independent dynamic violation recall, multi-scene PPE precision, real factory footage and live-camera RTSP DMA-BUF reconnect remain unverified.
- Only this one 34-second source clip was curated for the task. It is enough to exercise motion and the full Worker path, but it is not a diverse evaluation collection.

The benchmark runs that disable events use high confirmation thresholds and are intentionally excluded from the event counts above. Their matched hardware performance results are in [ZERO_COPY_BENCHMARK_REPORT.md](ZERO_COPY_BENCHMARK_REPORT.md).
