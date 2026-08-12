# Refactored Architecture

本项目现在采用分层包结构，同时保留原有 `scripts/*.py` 命令。

## 入口

统一 CLI：

```bash
PYTHONPATH=src python -m badminton_data_process --help
PYTHONPATH=src python -m badminton_data_process metadata validate --summary
PYTHONPATH=src python -m badminton_data_process pipeline run raw_videos/synthetic_match.mp4 \
  --config configs/experiments/synthetic_smoke.yaml \
  --run-id smoke_refactor \
  --skip-visualize
```

安装 editable 包后也可以直接使用：

```bash
bdp metadata validate --summary
bdp rally segment raw_videos/example.mp4
bdp pipeline run raw_videos/synthetic_match.mp4 --config configs/experiments/synthetic_smoke.yaml
```

严格俯视主视角流程：

```bash
bdp main-view analyze raw_videos/match.mp4 --run-id match_main_view
bdp main-view export raw_videos/match.mp4 \
  --timeline runs/match_main_view/main_view/main_view_timeline.json
bdp rally segment raw_videos/match.mp4 \
  --timeline runs/match_main_view/main_view/main_view_timeline.json \
  --output-dir runs/match_main_view/rallies \
  --metadata-csv runs/match_main_view/rallies.csv
bdp review main-view --run runs/match_main_view
```

## 包结构

```text
src/badminton_data_process/
├── core/             # config, schemas, paths, run reports, CSV/JSON I/O
├── metadata/         # match metadata validation and download adapter
├── preprocess/       # main-view timeline smoothing utilities
├── main_view/        # strict birdseye main-view scoring, timeline, and export
├── rally/            # rally segmentation and manual review template
├── calibration/      # court calibration and homography helpers
├── tracking/
│   ├── player/       # player tracking adapter and future detector/tracker interfaces
│   └── shuttle/      # shuttle tracking adapter and future TrackNet interface
├── smoothing/        # trajectory smoothing adapter and smoother interface
├── visualization/    # chart/debug visualization adapter
├── evaluation/       # CSV-level tracking statistics
├── review/           # post-hoc accepted/rejected quality filtering
└── pipeline/         # end-to-end run orchestration
```

## 运行产物

单次实验输出进入：

```text
runs/{run_id}/
├── manifest.json
├── report.json
├── rallies.csv
├── rallies/
├── annotations/
└── outputs/
```

`manifest.json` 和 `report.json` 记录每个 stage 的输入、输出、参数、状态和耗时。

## 迁移策略

当前包内阶段模块通过 legacy adapter 调用原有脚本实现，避免重构时改变算法行为。后续可以逐个模块把脚本内核心逻辑迁入包内实现，并保持 `scripts/*.py` 作为兼容入口。

## 主视角质量控制

`main_view` 模块把“俯视主视角识别”和“rally 切分”拆开处理。它会输出：

```text
runs/{run_id}/main_view/
├── main_view_timeline.json
├── main_view_segments.csv
├── main_view_frame_scores.csv
├── main_view_quality.csv
├── rejected_segments.csv
├── clean_main_view.mp4
└── frame_index.csv
```

`review main-view` 会对已有 run 做后验质量过滤，重点检查球员投影到标准场地后的越界比例、边界卡死比例和异常 `court_y`。这一步用于自动剔除非主视角、标定异常或误切 rally。
