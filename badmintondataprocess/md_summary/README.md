# Badminton Data Process

基于职业羽毛球比赛官方录播视频的运动员与羽毛球轨迹识别及战术分析项目。

## 项目目标

本项目面向论文级视觉分析任务，目标包括：

- 运动员轨迹识别与映射
- 羽毛球轨迹检测、补全与平滑
- rally 切分
- 击球点与落点分析
- 站位热力图与战术统计输出

## 当前落地内容

当前仓库已完成首批基础搭建：

- `docs/implementation_plan.md`：可执行实施计划
- `docs/official_replay_sources.md`：官方比赛录像候选说明
- `metadata/matches.csv`：比赛元数据模板
- `metadata/rallies.csv`：rally 切分结果模板
- `metadata/official_replay_candidates.csv`：官方录像候选链接清单
- `metadata/video_sources.json`：视频来源模板
- `configs/project_config.yaml`：项目配置模板
- `scripts/`：阶段脚本入口与原型实现
- `raw_videos/`、`rallies/`、`annotations/`、`outputs/`：数据与结果目录

## 建议开发顺序

1. 填充 `metadata/matches.csv`
2. 下载并整理官方录播到 `raw_videos/`
3. 完成 `scripts/rally_segmentation.py`
4. 完成 `scripts/court_calibration.py`
5. 完成 `scripts/player_tracking.py`
6. 完成 `scripts/shuttle_tracking.py`
7. 完成 `scripts/trajectory_smoothing.py`
8. 完成 `scripts/visualize_results.py`

## 目录结构

```text
badmintondataprocess/
├── annotations/
├── configs/
├── docs/
├── metadata/
├── outputs/
│   ├── heatmaps/
│   └── trajectory_videos/
├── rallies/
├── raw_videos/
├── scripts/
├── promptplan.md
└── README.md
```

## 下一步建议

如果你希望我继续推进，我可以直接做下面任一项：

- 实现 `matches.csv` 自动校验与统计脚本
- 实现 rally 切分模块的第一版代码
- 实现球场标定模块的交互式原型
- 实现完整 Python 包结构与依赖文件

## 当前可用命令

### 未清洗完整视频一键分析

Windows 下使用项目专用 Conda GPU 环境，一条命令即可从包含采访、回放、特写和切视角的完整转播视频生成清洗回合、自动场地标定、双端骨骼、TrackNet 羽毛球轨迹、统计图和最终分析视频：

```powershell
.\scripts\run_full_analysis.ps1 `
  -InputVideo "F:\Good-Badminton\material\example.mp4" `
  -RunId example_full_analysis
```

第一次处理新机器或新素材时，可先执行只读预检；它会核对视频解码、生产配置、RTMPose、TrackNet 权重、PyTorch CUDA 和 ONNX Runtime CUDA，不创建运行目录：

```powershell
.\scripts\run_full_analysis.ps1 `
  -InputVideo "F:\Good-Badminton\material\example.mp4" `
  -ValidateOnly
```

脚本自动定位 `good-badminton` Conda 环境并执行以下唯一主流程：

```text
原始转播 -> Main View 清洗 -> Usable Rally 切分 -> 自动 Validated Calibration
         -> RTMPose CUDA 双端骨骼 -> TrackNet CUDA -> 轨迹平滑/统计
         -> 图表与战术诊断 -> 合并 H.264 分析视频
```

默认使用 `configs/production/full_video_gpu.yaml`。它不包含特定比赛的手工场地点，并显式要求 RTMPose/TrackNet 使用 CUDA；GPU 环境错误时在长任务开始前直接失败。TrackNet 对缺失帧不做盲目外推，轨迹平滑只允许二维端点位移不超过 80 像素的短缺口插值；羽毛球飞出画面时轨迹会断开，重新出现后从新观测点开始。Usable Rally 使用归一化球场主体区域内的局部运动证据，不再以整幅画面平均变化过滤远景小球员；前后上下文默认为 2.2/1.4 秒，`rallies_analysis.csv` 同时保留全画面和球场区域运动分数用于审计。生产 `hybrid` 场地标定只允许白色规则线生成正式 Homography 四角：绿色场地轮廓仅用于场地区域、Main View 或诊断证据，不会被当作双打边线。候选由完整 13 条标准场地线的投影支持率排序和验证，证据不足时明确拒绝，不回退到绿色边缘。执行结束后重点产物为：

```text
runs/<run-id>/analysis_summary.json
runs/<run-id>/rallies/                          # 自动清洗后的可用回合
runs/<run-id>/annotations/court_calibration/   # 每回合已验证标定
runs/<run-id>/annotations/*_tracks_smoothed.csv
runs/<run-id>/outputs/tracking_charts/
runs/<run-id>/outputs/demo/badminton_full_analysis.mp4
```

同一 `run-id` 可安全恢复中断任务；如果源视频或配置不同，工作流会拒绝混用旧产物。只有确认需要全部重跑时才添加 `-Force`。

### 一键分析与演示视频

完整流水线会在战术分析结束后生成综合演示视频：

```bash
bdp pipeline run raw_videos/example.mp4 --run-id example
```

默认输出：

```text
runs/example/outputs/demo/badminton_analysis_demo.mp4
```

演示视频按 `video_stem + rally_id + frame_id` 关联数据，并同时绘制球场边界、球员框、羽毛球轨迹、已启用球员的战术统计和标准俯视球场。俯视图包含单双打边线、前发球线、双打后发球线、分区中线、球网以及 near/far 位置标签。使用 `--skip-demo` 可跳过最终渲染。

人物轨迹采用双锚点契约：画面标签优先位于 `body_center`（躯干中心），米制场地坐标、距离和速度只由 `ground_contact`（球员地面接触点）产生。旧 CSV 的 `image_x/image_y` 通过兼容 Adapter 保持“框底中心”语义，不会被静默解释为躯干中心。

默认最终视频由独立 Media Export Module 转为 H.264、`yuv420p`、MP4 `faststart`；OpenCV 生成的中间视频与浏览器兼容视频都保存在当前 Run Layout 并登记为 Artifact。

球员追踪默认使用 YOLO 人体检测并恢复近端、远端两个场地角色：

```yaml
player_tracking:
  detector: yolo
  roles:
    - near
    - far
  yolo_confidence: 0.25
  yolo_image_size: 640
```

角色关联先使用 Validated Calibration 的 Homography 将候选严格分到球网两侧，再按 near、far 独立维护轨迹；远端恢复不会复用近端框。双端结果当前仍属于 experimental Dual-side Observation，不代表稳定人物身份或完整战术结论。需要退回已验证的近端基线时，显式配置 `roles: [near]`；`heuristic` 检测器只建议用于不加载模型的快速 smoke test。

### 姿态与骨骼识别

CPU 环境可直接安装 RTMPose 可选依赖：

```bash
python -m pip install -e ".[pose]"
```

RTX 50 系显卡建议使用仓库的 CUDA 12.8 Conda 环境。`rtmlib` 的包元数据会依赖 CPU 版 `onnxruntime`，因此 GPU 环境先由 `environment.yml` 安装 `onnxruntime-gpu`，再无依赖安装 `rtmlib`，避免两个 ONNX Runtime 包互相覆盖：

```powershell
conda env create -f environment.yml
conda activate good-badminton
python -m pip install rtmlib==0.0.16 --no-deps
bdp verify
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

最后一条命令必须包含 `CUDAExecutionProvider`。高质量 GPU 验证配置应显式要求 CUDA，环境错误时直接失败，禁止静默回退 CPU：

使用 rtmlib 自动缓存的 balanced RTMPose：

```yaml
player_tracking:
  detector: rtmpose
  roles: [near, far]
  pose_keypoint_confidence: 0.35
  pose_min_keypoints: 5
  rtmpose_mode: balanced
  rtmpose_backend: onnxruntime
  rtmpose_device: cuda

shuttle_tracking:
  model: tracknet
  tracknet_weights: weights/TrackNet_best.pt
  tracknet_device: cuda
```

如需完全离线运行，可同时配置 `rtmpose_detector_model` 与 `rtmpose_pose_model` 的本地 ONNX 路径；只配置其中一个会在运行前被拒绝。也可以设置 `detector: yolo_pose` 和 `pose_model: yolo11n-pose.pt` 使用较轻的 YOLO Pose Adapter。

姿态结果以 COCO-17 具名关键点写入 `player_tracks.csv`。骨架只连接达到置信度阈值的端点；躯干中心优先来自肩和髋，球员地面接触点优先来自双脚踝或单脚踝，正式米制坐标不使用躯干投影。

也可以基于已有 run 单独重新渲染：

```bash
bdp render demo \
  --rallies-csv runs/example/rallies.csv \
  --player-tracks-csv runs/example/annotations/player_tracks_smoothed.csv \
  --shuttle-tracks-csv runs/example/annotations/shuttle_tracks_smoothed.csv \
  --calibration-dir runs/example/annotations/court_calibration \
  --tactics-events-csv runs/example/outputs/tactics/tactics_events.csv \
  --tactics-summary-csv runs/example/outputs/tactics/tactics_summary.csv \
  --output runs/example/outputs/demo/badminton_analysis_demo.mp4
```

可选研究 WebUI 与 CLI 调用同一个流水线 Interface：

```bash
python -m pip install -e ".[ui]"
bdp webui
```

WebUI 从 Run Manifest 读取 `success`、`rejected`、`failed` 和 `empty`，不会维护另一条分析调用链，也不会自动清理历史运行目录。

校验比赛元数据：

```bash
python scripts/prepare_matches.py --summary
```

从完整比赛视频生成候选 rally 片段：

```bash
python scripts/rally_segmentation.py raw_videos/example.mp4
```

自定义切分参数并输出 metadata：

```bash
python scripts/rally_segmentation.py raw_videos/example.mp4 \
  --output-dir rallies \
  --metadata-csv metadata/rallies.csv \
  --sample-every 15 \
  --min-rally-seconds 3.0 \
  --max-gap-seconds 1.2
```

## 依赖说明

当前已实现脚本的基础依赖包括：

- Python 3.10+
- `opencv-python`
- `numpy`
- `yt-dlp`

安装依赖：

```bash
python -m pip install -r requirements.txt
```

## venv 环境

推荐使用项目内的 `.venv`：

```bash
chmod +x setup_venv.sh
./setup_venv.sh
source .venv/bin/activate
```

不激活环境时，也可以直接这样运行：

```bash
.venv/bin/python scripts/prepare_matches.py --summary
.venv/bin/python scripts/rally_segmentation.py --help
```

下载 `matches.csv` 中的比赛录像：

```bash
.venv/bin/python scripts/download_matches.py
```

只下载指定比赛：

```bash
.venv/bin/python scripts/download_matches.py --match-id BWF_2025_KoreaOpen_MS_Final_001
```

使用 cookies 进行受限视频下载：

```bash
.venv/bin/python scripts/download_matches.py --cookies /path/to/cookies.txt
```

如果本机浏览器可直接读取 cookies：

```bash
.venv/bin/python scripts/download_matches.py --cookies-from-browser chrome
```

按站点分别指定 cookies：

```bash
.venv/bin/python scripts/download_matches.py \
  --youtube-cookies /mnt/d/badmintondataprocess/www.youtube.com_cookies.txt \
  --olympics-cookies /mnt/d/badmintondataprocess/www.olympics.com_cookies.txt
```
