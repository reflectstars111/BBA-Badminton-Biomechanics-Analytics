# Good-Badminton · 研究级羽毛球转播分析管线 🏸

> **基于 [yo-WASSUP/Good-Badminton](https://github.com/yo-WASSUP/Good-Badminton) 的二次开发。**
>
> 上游是一个「逐帧检测 + 轨迹可视化」的羽毛球比赛视频分析工具；本项目在其计算机视觉能力之上，
> 重构为一条面向论文级视觉分析、**可复现、质量门控**的研究数据处理管线。

[中文](README.md) · [English](README_EN.md)

---

## 与上游的关系

本项目 fork 自 [yo-WASSUP/Good-Badminton](https://github.com/yo-WASSUP/Good-Badminton)（上游作者 yo-WASSUP，Apache License 2.0）。

- **保留**上游的 CV 能力与交互：RTMPose / RTMO / YOLO Pose 多姿态模型、YOLO 羽毛球检测、球场坐标映射思路、Gradio WebUI。
- **新增** `badmintondataprocess/` 研究管线：从**未剪辑的完整转播**自动清洗出可用回合，再完成标定、跟踪、平滑与战术分析，并全程产出可审计、可复现的运行记录。
- **冻结**上游的 `main.py` 演示管线，仅作为兼容入口保留，不再新增算法职责。

## 我们的优势（相对上游）

> 以下对比依据上游当前 [README](https://github.com/yo-WASSUP/Good-Badminton) 及其「开发计划」。

| 维度 | 上游 yo-WASSUP | 本项目 |
| --- | --- | --- |
| **输入 / 标定** | 需选球场模板图或手动四点标定（自动检测白/黄球场线） | 未剪辑完整转播，自动清洗 + 自动 Validated Calibration |
| **回合识别** | 基于球场模板匹配的「连续比赛画面」，无法区分回放/特写 | Main View 门禁 + Usable Rally 切分，自动剔除采访/回放/特写/计分牌/切视角 |
| **球场标定验证** | 自动匹配白/黄球场线 + WebUI 手动四点修正，候选无独立验证 | Hough 白线 + 13 条规则线支持率 + 几何/重投影/凸性/时序稳定性多帧验证，证据不足**明确拒绝而非猜测** |
| **羽毛球检测** | 单帧 YOLO 检测 | TrackNet 多帧深度检测，约 93% 密集可见轨迹 |
| **球员定位** | 检测框 / 关键点直接投影到球场 | RTMPose 姿态 + **双锚点**（躯干中心 / 地面接触点分离），正式米制坐标只来自地面接触点 |
| **运动统计** | 移动距离、速度、回合数（击球点为实验功能） | 击球/落点（物理规则：每拍羽毛球落地恰好一次）、跑动距离、覆盖面积、站位区域占比，每项指标带资格门控 |
| **批量分析** | 未实现（上游开发计划中的待办） | `bdp pipeline batch` 批量工作流 |
| **失败语义 / 可复现** | 无结构化阶段结果，输出目录拼接、无运行清单 | missing / rejected / failed / empty / success 五态分离 + Run Manifest（配置/输入/模型指纹、断点 resume） |

**补全了上游自身的待办。** 上游 README「开发计划」中未勾选的四项，本项目均已落地：

- [ ] 更稳定的击球点识别 → 物理规则分类击球 / 落点（每拍羽毛球落地恰好一次）
- [ ] 更精确的羽毛球检测模型 → 集成 TrackNet 多帧深度检测
- [ ] 更完整的技术动作统计 → 资格门控的战术分析（击球/落点/跑动/覆盖/区域占比）
- [ ] 批量视频分析工作流 → `bdp pipeline batch`

工程方面：122 项测试、CUDA 加速（RTMPose 人物阶段 CPU ~603s → CUDA ~41s）、H.264 浏览器兼容导出。

## 快速开始

### 环境

需要 CUDA 版 PyTorch + ONNX Runtime。项目自带可复现的 Conda 环境：

```powershell
conda env create -f badmintondataprocess/environment.yml
conda activate good-badminton
python -m pip install -e "badmintondataprocess/."
python -m pip install rtmlib==0.0.16 --no-deps   # 避免覆盖 onnxruntime-gpu
bdp verify
```

### 一键分析未剪辑转播

一条命令，从包含采访、回放、特写和切视角的完整转播视频，产出清洗后的回合、标定、轨迹、统计图和最终分析视频：

```powershell
bdp analyze F:\material\match.mp4 --run-id match_full_analysis
```

也可用 PowerShell 薄入口（自动定位 `good-badminton` 环境）：

```powershell
.\badmintondataprocess\scripts\run_full_analysis.ps1 `
  -InputVideo "F:\Good-Badminton\material\example.mp4" `
  -RunId example_full_analysis
```

首次在新机器/新素材上运行前，先做只读预检（核对解码、配置、RTMPose/TrackNet 权重与 CUDA，不创建运行目录）：

```powershell
bdp analyze F:\material\match.mp4 --preflight-only
```

### 分阶段管线

```bash
bdp pipeline run raw_videos/match.mp4 --run-id match_manual --config configs/experiments/synthetic_smoke.yaml
```

同一 `run-id` 可安全恢复中断任务；源视频或配置变化时会拒绝混用旧产物，确认重跑才加 `--force`。

## 管线流程

```text
原始转播 -> Main View 清洗 -> Usable Rally 切分 -> 自动 Validated Calibration
         -> RTMPose CUDA 双端骨骼 -> TrackNet CUDA -> 轨迹平滑/统计
         -> 图表与战术诊断 -> 合并 H.264 分析视频
```

重点产物：

```text
runs/<run-id>/analysis_summary.json
runs/<run-id>/rallies/                          # 自动清洗后的可用回合
runs/<run-id>/annotations/court_calibration/    # 每回合已验证标定
runs/<run-id>/annotations/*_tracks_smoothed.csv
runs/<run-id>/outputs/tracking_charts/
runs/<run-id>/outputs/demo/badminton_full_analysis.mp4
```

## 统一 CLI

CLI、批处理与 WebUI 共用同一条管线 Interface，不存在语义不同的平行实现：

```text
bdp analyze <video>             # 一键完整转播分析
bdp pipeline run / batch        # 分阶段 / 批量运行
bdp rally segment               # 回合切分
bdp calibrate                   # 球场标定
bdp track players / shuttle     # 球员 / 羽毛球跟踪
bdp smooth                      # 轨迹平滑
bdp tactics analyze             # 战术分析
bdp render demo                 # 演示视频重渲染
bdp compare trackers            # 跟踪器对照
bdp webui                       # 浏览器界面
bdp verify                      # 环境自检
```

## 目录结构

```text
badmintondataprocess/
├── src/badminton_data_process/   # 研究管线包（core / main_view / rally /
│                                 #   calibration / tracking / smoothing /
│                                 #   tactics / visualization / media / webui）
├── scripts/                      # 兼容命令入口 + 一键分析 PowerShell 脚本
├── configs/                      # default / experiments / production(full_video_gpu)
├── tests/                        # 122 项测试
└── docs/                         # 架构、迁移与实施计划
```

## 能力边界（诚实声明）

- 双端球员定位（near/far 场地角色）与击球/落点事件属于 **experimental**，不等于稳定运动员身份或完整战术结论；近端定位是当前优先验证的范围。
- 演示视频是 **Diagnostic Demo**，用于排错与展示已验证产物，不作为模型正确性的证据。
- 冻结标注集（Main View / rally / 球场角点 / 球员脚点 / 羽毛球真值）仍在建立中，precision / recall / ID switch 等精度指标尚未在真实标注上形成基线。

## 致谢与许可

感谢上游 [yo-WASSUP/Good-Badminton](https://github.com/yo-WASSUP/Good-Badminton) 及其贡献者，以及 RTMPose / RTMO / OpenMMLab、[rtmlib](https://github.com/Tau-J/rtmlib)、[Ultralytics](https://github.com/ultralytics/ultralytics)、[TrackNet](https://github.com/yastrebksv/TrackNet) 提供的算法与数据基础。

本项目沿用上游 Apache License 2.0。
