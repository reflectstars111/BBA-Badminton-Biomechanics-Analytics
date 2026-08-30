# BBA · Badminton Biomechanics Analytics

> 羽毛球生物力学与比赛视频智能分析系统

[中文](README.md) · [English](README_EN.md)

BBA 将一段羽毛球比赛视频处理为可复核的分析视频、球员姿态与场地坐标、羽毛球轨迹、逐回合统计、热力图和结构化数据。系统支持包含采访、回放、特写与切镜头的未清洗转播，也支持用户已经裁切好的比赛片段。

项目当前重点不是堆叠演示效果，而是建立一条统一、可恢复、带质量门控的研究管线：CLI、批处理和 WebUI 消费同一套 Pipeline Interface；失败、拒绝、无数据与成功具有明确语义；所有正式结果都保留在可审计的 Run Manifest 中。

## 当前能力

| 能力 | 当前状态 | 说明 |
| --- | --- | --- |
| 未清洗转播自动清洗 | 已实现 | Main View 检测与 Usable Rally 切分，过滤采访、回放、特写、计分牌和明显切镜头 |
| 已裁切片段分析 | 已实现 | 可跳过素材清洗判断，将整段作为比赛素材进入后续流程 |
| 俯视 / 标准转播视角 | 已实现 | 当前主要验证范围 |
| 低视角 / 侧面固定机位 | 实验性 | 使用单独配置；严重透视和遮挡时建议手动确认场地 |
| 自动球场标定 | 已实现 | 基于白色规则线、几何关系、重投影、凸性和稳定性验证 |
| 标准球场模型手动修正 | 已实现 | 标注两条纵线和两条横线，可由 6.10 m × 13.40 m 模型推导画外角点 |
| 双端球员检测与骨骼 | 已实现 / 实验性 | RTMPose CUDA；near/far 是场地角色，不等于稳定运动员身份 |
| 球员米制场地定位 | 已实现 | 使用脚踝 / 地面接触锚点投影；躯干中心仅用于姿态与展示 |
| 羽毛球轨迹 | 已实现 | TrackNet 多帧检测，保留 observed / interpolated / missing 语义并限制大跨度插值 |
| 运动数据与图表 | 已实现 | 移动距离、速度、覆盖率、重心相对高度、站位区域、轨迹、散点和热力图 |
| 骨骼动作细节分析 | 开发中 | 计划加入击球动作分类、挥拍阶段、关节角度、稳定性和步法分析 |

当前自动化测试：**147 项**。生产配置支持 NVIDIA CUDA、RTMPose 和 TrackNet GPU 推理，并输出浏览器兼容的 H.264 分析视频。

## 一键启动 BBA WebUI

### Windows CMD

在 CMD 中进入仓库目录后运行：

```cmd
cd /d F:\Good-Badminton
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File F:\Good-Badminton\start_webui.ps1
```

如果仓库不在 `F:\Good-Badminton`，请替换为实际绝对路径。启动完成后访问：

```text
http://127.0.0.1:7860
```

启动脚本会查找或首次创建 `good-badminton` Conda 环境，安装缺少的 WebUI / RTMPose 依赖，然后启动本地浏览器工作区。首次创建环境需要下载 CUDA 与模型运行依赖，耗时会明显长于后续启动。视频与分析结果默认只保留在当前计算机。

### WebUI 工作流

1. 上传比赛视频；
2. 选择“未裁切 / 已裁切”和“俯视 / 低视角”；
3. 检查系统挑选的代表帧与自动球场标定；
4. 接受自动结果，或使用标准球场模型线进行手动修正；
5. 启动完整分析；
6. 查看实时阶段、帧级进度、预计剩余时间和运行日志；
7. 下载分析视频、图表、JSON、CSV 与 Run Manifest。

WebUI 的预计用时不是固定倒计时。球员跟踪和 TrackNet 会实时上报“已处理帧 / 总帧数”；系统只有获得当前阶段真实吞吐后才计算 ETA。无法获得可靠阶段内进度时会显示“正在测量当前阶段”，不会输出没有依据的数字。

分析报告按比赛概览、球员全局统计、球员逐回合统计、羽毛球逐回合统计和质量诊断组织。界面会明确显示数据覆盖率、标定资格与能力边界；尚未具备可靠依据的骨骼动作细节分析统一标记为“开发中”。

## 场地标定

正式米制指标依赖 **Validated Calibration**。系统不会仅因为算法返回了四个点就认定标定成功。

自动标定会综合检查：

- 主要白色场地边线及其线段支持率；
- 四边形面积、顺序、凸性和画面边界关系；
- 标准球场模型重投影误差与 Homography 条件数；
- 多个代表时间点之间的角点稳定性；
- 低视角下的透视与可见边线约束。

自动结果不正确时，可在 WebUI 中选择实际可见的两条纵线和两条横线，并在每条线上点击两个相距较远的点。系统拟合无限直线、求交点，并通过标准双打球场模型推导完整外框，因此手动角点允许位于视频边界以外。

一次确认只适用于同一固定机位。如果视频中实际比赛画面也频繁更换机位，应拆分素材或分别确认，不能复用单个 Homography。

## 完整分析管线

```text
Source Match
  -> Main View 清洗
  -> Usable Rally 切分
  -> Validated Calibration
  -> RTMPose 球员检测 / 骨骼 / 地面接触点
  -> TrackNet 羽毛球轨迹
  -> 受约束的轨迹平滑与异常插值过滤
  -> 热力图、散点图与轨迹图
  -> 资格门控的统计 / 战术诊断
  -> H.264 分析视频与结构化报告
```

九个阶段均写入 Run Manifest。WebUI 进度条读取真实阶段结果，球员和羽毛球跟踪阶段进一步使用帧级进度；同一 `run-id` 可以安全恢复，输入或配置变化时会拒绝混用旧产物。

## 分析结果

### 球员指标

- near / far 场地角色；
- 有效帧、跟踪覆盖率和骨骼有效率；
- 全场及逐回合移动距离；
- 当前、平均和稳健最高移动速度；
- 平均重心相对高度；
- 前场、中场、后场站位占比；
- 米制球场轨迹、散点图和热力图。

球员地面位置优先使用双脚踝中点，单脚有效时降级为单脚，姿态不可用时才退回检测框底部。锚点来源、置信度和有效性会写入轨迹数据。

### 羽毛球指标

- 有效观测帧与可见率；
- observed / interpolated / missing 状态；
- 当前、平均和稳健最高图像速度；
- 屏幕对角线归一化速度；
- 逐回合轨迹和调试视频。

羽毛球在空中运动，单目视频不能仅通过地面 Homography 恢复可信三维米制球速或正式落点。因此 BBA 当前明确报告图像平面速度；击球、落点和完整战术结论仍属于实验研究范围，不以伪精确数值冒充正式结果。

## 命令行使用

### 环境安装

```powershell
conda env create -f badmintondataprocess/environment.yml
conda activate good-badminton
python -m pip install -e "badmintondataprocess/.[ui,yaml]"
python -m pip install rtmlib==0.0.16 --no-deps
bdp verify
```

`rtmlib` 使用 `--no-deps` 安装，避免把现有 `onnxruntime-gpu` 替换成 CPU 版本。

### 一键分析未清洗转播

```powershell
bdp analyze F:\material\match.mp4 --run-id match_full_analysis
```

首次在新机器或新素材上运行前，可执行只读预检：

```powershell
bdp analyze F:\material\match.mp4 --preflight-only
```

### 统一 CLI

```text
bdp analyze <video>             # 一键完整分析
bdp pipeline run / batch        # 分阶段 / 批量运行
bdp rally segment               # 回合切分
bdp calibrate                   # 球场标定
bdp track players / shuttle     # 球员 / 羽毛球跟踪
bdp smooth                      # 轨迹平滑
bdp tactics analyze             # 战术诊断
bdp render demo                 # 分析视频重渲染
bdp compare trackers            # 跟踪器对照
bdp webui                       # BBA 浏览器界面
bdp verify                      # 环境自检
```

## 运行产物

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

## 目录结构

```text
badmintondataprocess/
├── src/badminton_data_process/
│   ├── core / pipeline / main_view / rally
│   ├── calibration / tracking / smoothing / tactics
│   ├── visualization / media / review
│   └── webui
├── configs/                    # default / production / experiments / webui
├── scripts/                    # PowerShell 与兼容入口
├── tests/                      # 147 项自动化测试
├── docs/                       # 架构与迁移计划
└── runs/                       # 每次运行的独立产物目录
```

## 能力边界

- 低视角、双端身份保持、击球归属、落点与完整战术结论仍是实验能力；
- near / far 只表示球网两侧的场地角色，不等于跨回合稳定运动员身份；
- Diagnostic Demo 用于展示与排错，不是精度证明；
- 没有通过标定质量门控的数据不能产生正式米制指标；
- 冻结真实标注集仍在建设中，precision、recall、ID switch 等基准尚未完整发布。

## 与上游的关系

本项目基于 [yo-WASSUP/Good-Badminton](https://github.com/yo-WASSUP/Good-Badminton) 继续开发（上游作者 yo-WASSUP，Apache License 2.0）。

- 保留并迁移 RTMPose / RTMO / YOLO Pose、羽毛球检测和球场映射相关经验；
- 新增 `badmintondataprocess/` 统一研究管线、质量门控、Run Manifest、批处理和 BBA WebUI；
- 根目录旧 `main.py` 演示入口保持冻结，仅用于兼容，不再承担新算法职责。

感谢上游项目及其贡献者，以及 RTMPose / RTMO / OpenMMLab、[rtmlib](https://github.com/Tau-J/rtmlib)、[Ultralytics](https://github.com/ultralytics/ultralytics) 和 [TrackNet](https://github.com/yastrebksv/TrackNet) 提供的算法与工程基础。

本项目沿用上游 Apache License 2.0。
