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
