# BBA · Badminton Biomechanics Analytics

> A research-oriented system for badminton biomechanics and intelligent match-video analysis.

[中文](README.md) · [English](README_EN.md)

BBA turns a badminton match video into a reviewable analysis video, player poses and court coordinates, shuttle trajectories, per-rally statistics, heatmaps, and structured data. It accepts both uncleaned broadcasts containing interviews, replays, close-ups, and camera cuts, and clips that already contain only active play.

The project focuses on a unified, resumable, quality-gated research pipeline rather than disconnected demo effects. The CLI, batch runner, and WebUI all call the same Pipeline Interface; rejected, failed, empty, and successful results have distinct meanings; every formal output is recorded in an auditable Run Manifest.

## Current capabilities

| Capability | Status | Notes |
| --- | --- | --- |
| Automatic cleaning of full broadcasts | Available | Main View detection and Usable Rally segmentation filter interviews, replays, close-ups, scoreboards, and obvious camera cuts |
| Analysis of pre-cut clips | Available | The complete clip can enter the analysis stages without broadcast cleaning |
| Overhead / standard broadcast view | Available | The primary validated camera range |
| Low / side fixed-camera view | Experimental | Uses a dedicated profile; severe perspective and occlusion should be checked manually |
| Automatic court calibration | Available | Uses white regulation lines plus geometry, reprojection, convexity, and temporal-stability validation |
| Model-based manual court correction | Available | Two longitudinal and two transverse lines determine the standard 6.10 m × 13.40 m court, including corners outside the frame |
| Dual-side player detection and skeletons | Available / experimental | RTMPose with CUDA; near/far are court-side roles, not persistent athlete identities |
| Metric player localization | Available | Uses ankle / ground-contact anchors; the torso center is reserved for pose analysis and display |
| Shuttle tracking | Available | TrackNet multi-frame detection with explicit observed / interpolated / missing states and guarded gap interpolation |
| Movement data and charts | Available | Distance, speed, coverage, relative center-of-mass height, court-zone occupancy, trajectories, scatter plots, and heatmaps |
| Detailed stroke biomechanics | In development | Planned stroke classification, swing phases, joint angles, stability, and footwork analysis |

The current automated test suite contains **147 tests**. Production profiles support NVIDIA CUDA, RTMPose, and TrackNet GPU inference and export browser-compatible H.264 analysis video.

## One-click BBA WebUI

### Windows CMD

From CMD, change to the cloned repository and run:

```cmd
cd /d F:\Good-Badminton
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File F:\Good-Badminton\start_webui.ps1
```

Replace `F:\Good-Badminton` with the actual absolute path if the repository was cloned elsewhere. When startup completes, open:

```text
http://127.0.0.1:7860
```

The launcher locates the `good-badminton` Conda environment or creates it on first use, installs missing WebUI / RTMPose dependencies, and starts the local browser workspace. Creating the environment for the first time downloads the CUDA and model runtime dependencies and therefore takes considerably longer than later launches. Uploaded videos and analysis results remain on the local machine by default.

### WebUI workflow

1. Upload a match video.
2. Choose uncleaned / pre-cut input and overhead / low camera view.
3. Inspect the representative frame and automatic court calibration.
4. Accept the result or correct it with the standard-court model lines.
5. Start the complete analysis.
6. Follow the actual stage, frame-level progress, estimated remaining time, and run log.
7. Review and download the analysis video, charts, JSON, CSV, and Run Manifest.

The ETA is not a fixed countdown. Player tracking and TrackNet report processed frames against total frames. BBA calculates an ETA only after measuring real throughput for the active stage; otherwise it displays that the stage is still being measured instead of inventing a number.

Reports are organized into match overview, whole-match player statistics, per-rally player statistics, per-rally shuttle statistics, and quality diagnostics. The interface exposes coverage, calibration eligibility, and capability limits; unsupported detailed biomechanics are consistently marked as in development.

## Court calibration

Formal metric measurements require a **Validated Calibration**. A detector returning four points is not sufficient evidence by itself.

Automatic calibration checks:

- support from the main white regulation lines;
- quadrilateral area, ordering, convexity, and relation to image boundaries;
- reprojection error against the standard court and the Homography condition number;
- corner stability across representative timestamps;
- visible-line and perspective constraints for low-angle views.

If the automatic result is wrong, the WebUI lets the user mark two longitudinal lines and two transverse lines by clicking two well-separated points on each line. BBA fits infinite lines, intersects them, and maps them to the standard doubles-court model. The resulting court corners may therefore lie outside the video frame.

One confirmation applies only to the same fixed camera. If active-play footage itself uses multiple camera geometries, split the material or confirm each geometry separately; one Homography must not be reused across different views.

## Complete analysis pipeline

```text
Source Match
  -> Main View cleaning
  -> Usable Rally segmentation
  -> Validated Calibration
  -> RTMPose player detection / skeletons / ground contact
  -> TrackNet shuttle trajectory
  -> constrained smoothing and unsafe-interpolation rejection
  -> heatmaps, scatter plots, and trajectory charts
  -> eligibility-gated statistics / tactical diagnostics
  -> H.264 analysis video and structured report
```

All nine stages write to the Run Manifest. The WebUI progress bar reads real Stage Results, with frame-level progress inside player and shuttle tracking. An interrupted run can resume under the same `run-id`; changes to the input or configuration prevent stale artifacts from being mixed into a new result.

## Analysis results

### Player metrics

- near / far court-side role;
- valid frames, tracking coverage, and valid-pose ratio;
- whole-match and per-rally movement distance;
- current, average, and robust maximum movement speed;
- average relative center-of-mass height;
- front-, mid-, and back-court occupancy;
- metric court trajectories, scatter plots, and heatmaps.

Player ground position uses the midpoint of both ankles when possible, falls back to one valid ankle, and uses the detection-box bottom only when pose data is unavailable. The anchor source, confidence, and validity are written to the track data.

### Shuttle metrics

- valid observation frames and visibility ratio;
- observed / interpolated / missing states;
- current, average, and robust maximum image-plane speed;
- screen-diagonal-normalized speed;
- per-rally trajectories and debug video.

Because the shuttle travels through the air, a monocular ground-plane Homography cannot recover trustworthy 3D metric speed or an official landing point by itself. BBA therefore reports image-plane speed explicitly. Stroke events, landing points, and complete tactical conclusions remain experimental research areas and are not presented as falsely precise formal results.

## Command-line usage

### Environment setup

```powershell
conda env create -f badmintondataprocess/environment.yml
conda activate good-badminton
python -m pip install -e "badmintondataprocess/.[ui,yaml]"
python -m pip install rtmlib==0.0.16 --no-deps
bdp verify
```

`rtmlib` is installed with `--no-deps` to avoid replacing the existing `onnxruntime-gpu` package with a CPU build.

### One-command analysis of an uncleaned broadcast

```powershell
bdp analyze F:\material\match.mp4 --run-id match_full_analysis
```

Before the first run on a new machine or new material, a read-only preflight can verify decoding, configuration, model weights, and CUDA without creating a run directory:

```powershell
bdp analyze F:\material\match.mp4 --preflight-only
```

### Unified CLI

```text
bdp analyze <video>             # one-command full analysis
bdp pipeline run / batch        # staged / batch execution
bdp rally segment               # rally segmentation
bdp calibrate                   # court calibration
bdp track players / shuttle     # player / shuttle tracking
bdp smooth                      # trajectory smoothing
bdp tactics analyze             # tactical diagnostics
bdp render demo                 # re-render the analysis video
bdp compare trackers            # tracker comparison
bdp webui                       # BBA browser workspace
bdp verify                      # environment self-check
```

## Run artifacts

```text
runs/<run-id>/manifest.json
runs/<run-id>/analysis_summary.json
runs/<run-id>/webui_report.json
runs/<run-id>/rallies/
runs/<run-id>/annotations/court_calibration/
runs/<run-id>/annotations/player_tracks*.csv
runs/<run-id>/annotations/shuttle_tracks*.csv
runs/<run-id>/outputs/tracking_charts/
runs/<run-id>/outputs/demo/badminton_full_analysis.mp4
```

## Repository layout

```text
badmintondataprocess/
├── src/badminton_data_process/
│   ├── core / pipeline / main_view / rally
│   ├── calibration / tracking / smoothing / tactics
│   ├── visualization / media / review
│   └── webui
├── configs/                    # default / production / experiments / webui
├── scripts/                    # PowerShell and compatibility entry points
├── tests/                      # 147 automated tests
├── docs/                       # architecture and migration plans
└── runs/                       # isolated artifacts for each run
```

## Scope and limitations

- Low-angle calibration, persistent dual-side identity, stroke attribution, landing points, and complete tactical conclusions remain experimental.
- near / far identify sides of the net, not stable athlete identities across rallies.
- A Diagnostic Demo is a visualization and debugging artifact, not proof of model accuracy.
- Data that does not pass calibration quality gates cannot produce formal metric measurements.
- A frozen real-world annotation set is still under construction; complete precision, recall, and ID-switch benchmarks have not yet been published.

## Relationship to upstream

This project continues development from [yo-WASSUP/Good-Badminton](https://github.com/yo-WASSUP/Good-Badminton) by upstream author yo-WASSUP under the Apache License 2.0.

- It preserves and migrates experience from RTMPose / RTMO / YOLO Pose, shuttle detection, and court coordinate mapping.
- It adds the unified `badmintondataprocess/` research pipeline, quality gates, Run Manifest, batch processing, and BBA WebUI.
- The legacy root `main.py` demo entry point is frozen for compatibility and no longer carries new algorithmic responsibilities.

Thanks to the upstream project and its contributors, and to RTMPose / RTMO / OpenMMLab, [rtmlib](https://github.com/Tau-J/rtmlib), [Ultralytics](https://github.com/ultralytics/ultralytics), and [TrackNet](https://github.com/yastrebksv/TrackNet) for their algorithmic and engineering foundations.

This project follows the upstream Apache License 2.0.
