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

所有上述路径均由 `core.paths.RunLayout` 这一统一 Interface 生成。主管线、Main View 独立入口和批处理不得自行拼接 `runs/`、`annotations/` 或 `outputs/`；`data.runs_dir` 是默认运行根，命令行 `--runs-dir` 可显式覆盖，且相对路径统一以项目根目录解析。这样把路径规则集中在一个高 Leverage、强 Locality 的 Module 中，并防止 `run_id` 通过路径片段逃逸运行根目录。

## 严格配置契约

`core.config_schema.parse_config` 是研究管线唯一的配置校验 Interface。它在任何 Artifact 写入前完成类型和语义校验，并一次性报告所有带完整字段路径的问题，包括未知顶层键、未知分区键、负数、越界比例、非法 ROI、无效模型名和不受支持的角色组合。

配置文件只保留已有 Implementation 真正消费的字段。阶段自己的输出路径、未接入的 tracker/pose 开关、未来模型占位符以及通用 `outputs` 开关已经删除；旧配置继续携带这些字段时会明确失败，不再静默制造“已经启用”的错觉。角色组合支持已验证回退模式 `roles: [near]` 和默认实验双端模式 `roles: [near, far]`；`[far]`、重复或乱序组合均拒绝。

## 综合演示视频

`demo_rendering` 是默认流水线的最后一个 stage。它消费 rally 视频、平滑后的球员/羽毛球轨迹、球场标定和战术分析结果，输出：

```text
runs/{run_id}/outputs/demo/badminton_analysis_demo.mp4
```

渲染器使用 `(video_stem, rally_id, frame_id)` 作为帧数据身份，避免不同 rally 从零开始计帧造成的数据串线。独立入口为 `bdp render demo`；流水线可通过 `--skip-demo` 跳过该阶段。

球员追踪默认采用 YOLO 双端场地角色模式：`detector: yolo`、`roles: [near, far]`、`yolo_image_size: 640`。角色在 Homography 投影后的标准球场坐标中判断，near 候选先分配，far 只能消费远半场剩余候选，两个角色维护独立轨迹状态。远端仍属于 experimental Dual-side Observation；如需稳定近端基线可显式配置 `[near]`。战术事件和位置图表仅输出实际启用的球员角色，启发式运动前景检测仅保留用于快速 smoke test。

Diagnostic Demo 的右上角俯视图使用 6.10m × 13.40m 标准双打外框，并同时绘制 5.18m 单打边线、距球网 1.98m 的前发球线、距底线 0.76m 的双打后发球线、两侧分区中线和球网。球员圆点继续消费 Artifact 中的标准场地坐标，越界 Observation 只计为 rejected，不会钳制到边线上。

## 迁移策略

部分包内阶段模块仍通过 legacy adapter 调用原有脚本实现，避免重构时一次改变全部算法行为。Usable Rally 的画面活动分析已经迁入 `rally/activity.py`，正式管线不再依赖旧脚本中的全画面运动判定；其余模块继续逐个迁移，并保持 `scripts/*.py` 作为兼容入口。

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

正式 `pipeline run` 现在先执行 Main View Stage；只有 `main_view_segments.csv`、质量表和 timeline 均有效且至少存在一个接受区间时，才会进入 rally segmentation。rally Module 消费原视频和 `main_view_timeline.json`，所有采样与导出区间均被限制在接受的 Main View 内。内部标签统一为枚举值 `MAIN_VIEW`；`MAIN_LIVE_VIEW` 与 `MAIN_BIRDSEYE_LIVE` 只在迁移期 Adapter 入口被归一化。

Usable Rally Module 会把活动/球场结构证据按真实帧间隔分组，并输出 `rally_decisions.csv`。活动分数使用归一化球场主体区域的局部帧差，审计 CSV 同时保留 `global_motion_score` 和 `play_area_motion_score`；远景小球员的运动不会再被观众席等大面积静态背景稀释。每个候选都具有 `accepted/rejected`、reason、证据样本数和来源 Main View 区间；只有 accepted 行会进入 `rallies.csv` 和视频导出。所有区间统一为 `[start_frame, end_frame)`，最短/最长时长先换算为整数帧门槛，避免 30.000033 FPS 把恰好 60 帧误判为不足 2 秒。导出视频必须严格包含 `end_frame - start_frame` 帧。零 accepted 结果会把 Stage 记录为 `rejected` 并停止下游。

`review main-view` 会对已有 run 做后验质量过滤，重点检查球员投影到标准场地后的越界比例、边界卡死比例和异常 `court_y`。这一步用于自动剔除非主视角、标定异常或误切 rally。
