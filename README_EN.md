# Good-Badminton · A Research-grade Badminton Broadcast Analysis Pipeline 🏸

> **A secondary development based on [yo-WASSUP/Good-Badminton](https://github.com/yo-WASSUP/Good-Badminton).**
>
> The upstream project is a "frame-by-frame detection + trajectory visualization" badminton match video-analysis tool.
> On top of its computer-vision capabilities, this project rebuilds it into a
> **reproducible, quality-gated** data-processing pipeline for publication-grade visual analysis.

[中文](README.md) · [English](README_EN.md)

---

## Relationship to Upstream

This project is forked from [yo-WASSUP/Good-Badminton](https://github.com/yo-WASSUP/Good-Badminton) (upstream author yo-WASSUP, Apache License 2.0).

- **Kept** the upstream CV capabilities and interactions: RTMPose / RTMO / YOLO Pose models, YOLO shuttlecock detection, the court coordinate-mapping approach, and the Gradio WebUI.
- **Added** the `badmintondataprocess/` research pipeline: from an **uncleaned full broadcast** it automatically cleans usable rallies, then runs calibration, tracking, smoothing and tactical analysis, all while producing auditable, reproducible run records.
- **Frozen** the upstream `main.py` demo pipeline, kept only as a compatibility entry point with no new algorithmic responsibility.

## Our Advantages (vs Upstream)

> The comparison below follows the upstream's current [README](https://github.com/yo-WASSUP/Good-Badminton) and its development plan.

| Dimension | Upstream yo-WASSUP | This Project |
| --- | --- | --- |
| **Input / annotation** | Requires a court template image or manual four-corner annotation (auto-detects white/yellow court lines) | Uncleaned full broadcast, auto-cleaned + automatic Validated Calibration |
| **Rally detection** | Template-matching "continuous court view", unable to tell replays/close-ups from play | Main View gate + Usable Rally segmentation, automatically dropping interviews/replays/close-ups/scoreboards/camera cuts |
| **Calibration validation** | Auto-matches white/yellow court lines + WebUI manual four-corner correction; candidates are not independently validated | Hough white-line + 13-line support + geometric / reprojection / convexity / temporal-stability multi-frame validation, **explicitly rejecting rather than guessing** when evidence is insufficient |
| **Shuttlecock detection** | Single-frame YOLO detection | TrackNet multi-frame deep detection, ~93% dense visible tracks |
| **Player localization** | Detection box / keypoints projected directly onto the court | RTMPose pose + **dual anchors** (body center / ground contact separated); official metric coordinates come only from ground contact |
| **Movement statistics** | Distance, speed, rally count (hit-point analysis is experimental) | Hit/landing (physical rule: the shuttle touches the floor exactly once per rally), distance, coverage, court-area occupancy, each metric gated by eligibility |
| **Batch analysis** | Not implemented (an open item in the upstream plan) | `bdp pipeline batch` workflow |
| **Failure semantics / reproducibility** | No structured stage results; ad-hoc output paths; no run manifest | Five distinct states — missing / rejected / failed / empty / success — + Run Manifest (config/input/model fingerprints, breakpoint resume) |

**Completed the upstream's own open items.** The four unchecked items in the upstream README's development plan are all implemented here:

- [ ] More stable hit-point recognition → physical-rule hit / landing classification (the shuttle touches the floor exactly once per rally)
- [ ] More accurate shuttlecock detection model → integrated TrackNet multi-frame deep detection
- [ ] More complete technique statistics → eligibility-gated tactical analysis (hit/landing/distance/coverage/area occupancy)
- [ ] Batch video analysis workflow → `bdp pipeline batch`

Engineering: 122 tests, CUDA acceleration (RTMPose player stage ~603s CPU → ~41s CUDA), H.264 browser-compatible export.

## Quick Start

### Environment

Requires CUDA PyTorch + ONNX Runtime. The repo ships a reproducible Conda environment:

```powershell
conda env create -f badmintondataprocess/environment.yml
conda activate good-badminton
python -m pip install -e "badmintondataprocess/."
python -m pip install rtmlib==0.0.16 --no-deps   # avoid clobbering onnxruntime-gpu
bdp verify
```

### One-command analysis of an uncleaned broadcast

A single command turns a full broadcast — interviews, replays, close-ups and camera cuts included — into cleaned rallies, calibration, tracks, charts, and a final analysis video:

```powershell
bdp analyze F:\material\match.mp4 --run-id match_full_analysis
```

Or use the PowerShell thin entry point (auto-locates the `good-badminton` environment):

```powershell
.\badmintondataprocess\scripts\run_full_analysis.ps1 `
  -InputVideo "F:\Good-Badminton\material\example.mp4" `
  -RunId example_full_analysis
```

Before the first run on a new machine or new material, do a read-only preflight (checks decoding, config, RTMPose/TrackNet weights and CUDA without creating a run directory):

```powershell
bdp analyze F:\material\match.mp4 --preflight-only
```

### Staged pipeline

```bash
bdp pipeline run raw_videos/match.mp4 --run-id match_manual --config configs/experiments/synthetic_smoke.yaml
```

The same `run-id` can safely resume an interrupted run; if the source video or config changes, stale artifacts are rejected and `--force` is required to rerun.

## Pipeline Flow

```text
raw broadcast -> Main View cleaning -> Usable Rally segmentation -> automatic Validated Calibration
         -> RTMPose CUDA dual-side skeleton -> TrackNet CUDA -> trajectory smoothing / stats
         -> charts & tactical diagnostics -> merged H.264 analysis video
```

Key artifacts:

```text
runs/<run-id>/analysis_summary.json
runs/<run-id>/rallies/                          # auto-cleaned usable rallies
runs/<run-id>/annotations/court_calibration/    # per-rally validated calibration
runs/<run-id>/annotations/*_tracks_smoothed.csv
runs/<run-id>/outputs/tracking_charts/
runs/<run-id>/outputs/demo/badminton_full_analysis.mp4
```

## Unified CLI

The CLI, batch runner and WebUI share a single pipeline interface — there is no second, semantically different implementation:

```text
bdp analyze <video>             # one-command full-broadcast analysis
bdp pipeline run / batch        # staged / batch runs
bdp rally segment               # rally segmentation
bdp calibrate                   # court calibration
bdp track players / shuttle     # player / shuttle tracking
bdp smooth                      # trajectory smoothing
bdp tactics analyze             # tactical analysis
bdp render demo                 # demo-video re-render
bdp compare trackers            # tracker comparison
bdp webui                       # browser interface
bdp verify                      # environment self-check
```

## Repository Layout

```text
badmintondataprocess/
├── src/badminton_data_process/   # research pipeline package (core / main_view / rally /
│                                 #   calibration / tracking / smoothing /
│                                 #   tactics / visualization / media / webui)
├── scripts/                      # compat entry points + one-click PowerShell script
├── configs/                      # default / experiments / production (full_video_gpu)
├── tests/                        # 122 tests
└── docs/                         # architecture, migration and implementation plans
```

## Scope and Limits (an honest statement)

- Dual-side player localization (near/far court roles) and hit/landing events are **experimental**; they do not imply stable athlete identity or full tactical conclusions. Near-side localization is the currently prioritized, validated scope.
- The demo video is a **Diagnostic Demo** for debugging and displaying validated artifacts, not evidence of model correctness.
- The frozen annotation set (Main View / rally / court corners / player foot points / shuttle ground truth) is still under construction; precision / recall / ID-switch metrics have not yet been baselined against real annotations.

## Acknowledgements and License

Thanks to upstream [yo-WASSUP/Good-Badminton](https://github.com/yo-WASSUP/Good-Badminton) and its contributors, and to RTMPose / RTMO / OpenMMLab, [rtmlib](https://github.com/Tau-J/rtmlib), [Ultralytics](https://github.com/ultralytics/ultralytics) and [TrackNet](https://github.com/yastrebksv/TrackNet) for the algorithmic and data foundations.

This project follows the upstream Apache License 2.0.
