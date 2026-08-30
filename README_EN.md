# BBA · Badminton Biomechanics Analytics

**Turn an ordinary match video into a watchable, measurable, and reviewable badminton performance report.**

[中文](README.md) · [English](README_EN.md)

BBA is a local video-analysis workspace for badminton coaching, research, and match review. Upload a full broadcast or a pre-cut rally video, choose an overhead or low-angle camera, confirm the court once, and let BBA clean usable rallies, track both players and their skeletons, follow the shuttle, map movement onto a regulation court, and assemble the results into one report.

Your footage does not need to leave your machine. Models, processing, and results stay local by default and are available through a browser-based interface.

![BBA overhead dual-player pose, shuttle trajectory, and regulation-court mapping](assets/readme/analysis-overhead-china2018.png)

> An actual frame produced by BBA. Orange and blue denote the near/far court-side roles, green shows the shuttle trajectory, and the mini court places both players on a regulation court model.

## What BBA can already do

- **Clean full broadcasts automatically** by extracting usable play from interviews, replays, close-ups, and camera cuts.
- **Track both players and their poses** with GPU-accelerated RTMPose, including ground-contact positions for the near and far sides.
- **Track the shuttle with TrackNet**, preserving observed, short-gap interpolated, and missing states while rejecting unsafe long jumps.
- **Produce metric court movement** by projecting player contact points onto a 6.10 m × 13.40 m regulation doubles court.
- **Handle overhead and low-angle footage**, with an experimental profile for strong perspective and small far-side players.
- **Deliver a complete result set**: browser-compatible H.264 video, match and rally metrics, CSV, JSON, charts, and a resumable Run Manifest.

## See it in action

### Overhead / standard broadcast view

For a conventional broadcast camera, BBA combines two-player skeletons, player boxes, shuttle trajectory, the full court outline, and regulation-court coordinates in one frame.

![BBA overhead analysis on the 2011 BWF World Superseries Finals](assets/readme/analysis-overhead-bwf2011.png)

### Low / side fixed-camera view

Low-angle footage makes the far player smaller, increases occlusion, and can place modeled court corners outside the image. BBA uses a dedicated profile and regulation-court model to retain two-player poses, shuttle motion, and court geometry; users can correct the court when automation is uncertain.

![BBA low-angle analysis and results interface](assets/readme/analysis-low-angle-lindan2026.png)

## From upload to report

1. **Upload footage** as either an uncleaned full broadcast or a pre-cut match clip.
2. **Choose the camera**: overhead / standard broadcast or low / side fixed camera.
3. **Check the court** by accepting automatic calibration or correcting the representative frame with the regulation-court model.
4. **Run the analysis** while the UI reports the real stage, processed frames, elapsed time, and an ETA based on measured throughput.
5. **Review and export** the annotated video, match and rally metrics, charts, and raw structured data.

<table>
  <tr>
    <td width="50%"><img src="assets/readme/webui-home.png" alt="BBA WebUI home and workflow"></td>
    <td width="50%"><img src="assets/readme/webui-analysis-progress.png" alt="BBA WebUI real analysis progress"></td>
  </tr>
  <tr>
    <td align="center">A clear four-step workflow</td>
    <td align="center">Real stages, frame progress, and dynamic ETA</td>
  </tr>
</table>

## Start the WebUI with one command

### Windows

After installing [Conda / Miniconda](https://docs.conda.io/projects/miniconda/en/latest/), open CMD in the cloned repository and run:

```cmd
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File .\start_webui.ps1
```

You can also double-click `start_webui.bat`. When startup completes, open:

```text
http://127.0.0.1:7860
```

The first launch creates the `good-badminton` Conda environment and installs the WebUI, CUDA, RTMPose, and related dependencies, so it takes noticeably longer than later starts. Analysis time depends on video duration, resolution, frame rate, GPU, and the number of usable rallies.

## More than an annotated video

| Output | What it helps answer |
| --- | --- |
| Dual-player pose and shuttle video | When did each player move, jump, or lower their center of mass, and where did the shuttle travel? |
| Match and per-rally player metrics | What were the distance, average / maximum speed, tracking coverage, and valid-pose ratio? |
| Center-of-mass and court-zone distribution | How low was the player's average posture, and how much time was spent in the front, mid, and back court? |
| Regulation-court trajectories, scatter plots, and heatmaps | Which areas were occupied most often, and how did movement patterns change across rallies? |
| Shuttle observations and image-plane speed | Which frames genuinely observed the shuttle, how continuous was the path, and how did image-plane speed change? |
| CSV, JSON, and Run Manifest | How can results be audited, extended, re-plotted, or resumed after interruption? |

Detailed stroke biomechanics—including stroke classification, swing phases, joint angles, stability, and footwork evaluation—already has a place in the report and is currently marked **in development**.

## Built for research and coaching workflows

- **Validate before measuring**: formal metric results require a court calibration that passes quality checks.
- **Automate without removing control**: manual calibration uses a regulation-court model and supports corners beyond the video boundary.
- **Keep results traceable**: each stage records inputs, state, quality summary, and artifacts; failure, no data, and success remain distinct.
- **Resume long runs safely**: interrupted jobs can continue, while input or configuration changes prevent stale artifacts from being reused.
- **Use the local GPU**: production profiles support NVIDIA CUDA, RTMPose, and TrackNet GPU inference.

## Current scope

BBA demonstrates the complete journey from raw video to analysis report, but it remains an active research project:

- Overhead / standard broadcast footage is the primary validated range. Low angles, severe occlusion, and frequently changing cameras remain experimental.
- `near` and `far` are court-side roles, not persistent athlete identities across rallies.
- A monocular video and ground-plane Homography cannot reliably recover the shuttle's 3D metric speed or an official landing point. Formal reports currently focus on image-plane speed and visibility.
- Stroke attribution, movement-quality scoring, and complete tactical conclusions still require a larger frozen real-world annotation set.

When evidence is insufficient, BBA prefers to show “not enough data” or “in development” instead of presenting false precision.

## Building on open-source work

BBA **was not created from scratch**. It continues development from [yo-WASSUP/Good-Badminton](https://github.com/yo-WASSUP/Good-Badminton). We thank upstream author yo-WASSUP and contributors for the original project foundation and their exploration of player pose, shuttle detection, and court mapping.

On top of that foundation, BBA adds the unified and resumable `badmintondataprocess/` research pipeline, full-broadcast cleaning, quality gates, Run Manifest, batch processing, regulation-court manual calibration, overhead / low-angle profiles, and a one-command WebUI for new users. The legacy root demo remains frozen for compatibility and no longer carries new algorithm work.

We also thank RTMPose / RTMO / OpenMMLab, [rtmlib](https://github.com/Tau-J/rtmlib), [Ultralytics](https://github.com/ultralytics/ultralytics), and [TrackNet](https://github.com/yastrebksv/TrackNet) for their algorithmic and engineering foundations.

<details>
<summary><strong>For developers: environment, CLI, and run artifacts</strong></summary>

### Manual setup

```powershell
conda env create -f badmintondataprocess/environment.yml
conda activate good-badminton
python -m pip install -e "badmintondataprocess/.[ui,yaml]"
python -m pip install rtmlib==0.0.16 --no-deps
bdp verify
```

### Analyze a full broadcast with one command

```powershell
bdp analyze F:\material\match.mp4 --run-id match_full_analysis
```

### Main commands

```text
bdp analyze <video>             # one-command full analysis
bdp pipeline run / batch        # staged / batch execution
bdp calibrate                   # court calibration
bdp track players / shuttle     # player / shuttle tracking
bdp render demo                 # re-render the analysis video
bdp webui                       # start the browser workspace
bdp verify                      # verify the environment
```

Core artifacts for each run live in `badmintondataprocess/runs/<run-id>/`, including `manifest.json`, `analysis_summary.json`, trajectory CSV files, charts, and `outputs/demo/badminton_full_analysis.mp4`.

The current automated suite contains **149 tests**.

</details>

## License

This project follows the upstream [Apache License 2.0](LICENSE). Please preserve the license and upstream attribution when using, modifying, or redistributing the project.
