# BBA · Badminton Biomechanics Analytics

**把一段普通比赛视频，转化为可观看、可量化、可复核的羽毛球运动表现报告。**

[中文](README.md) · [English](README_EN.md) · [一页倡议书](BBA_INITIATIVE_ZH.md) · [项目简介与合作计划](PROJECT_BRIEF_ZH.md)

BBA（Badminton Biomechanics Analytics）是一套面向羽毛球训练、科研与比赛复盘的本地视频分析系统。上传完整转播或已经裁切好的比赛片段，选择俯视角或低视角，确认一次球场标定，系统即可自动完成有效回合清洗、双端球员骨骼追踪、羽毛球轨迹分析、标准球场映射和数据报告生成。

无需把视频交给云端。分析、模型与结果默认都保留在你的计算机上，并可通过一个浏览器界面完成。

![BBA 俯视角双端骨骼、羽毛球轨迹与标准球场映射](assets/readme/analysis-overhead-china2018.png)

> 上图为 BBA 实际分析成片。橙色与蓝色分别表示球网两侧的 near / far 场地角色，绿色表示羽毛球轨迹，右侧小场地显示球员在标准球场中的位置。

## 现在已经可以做到什么

- **完整转播自动清洗**：从包含采访、回放、特写和切镜头的长视频中提取可用比赛回合。
- **双端球员与骨骼追踪**：使用 GPU 加速的 RTMPose，同时分析近端和远端球员的姿态与地面位置。
- **TrackNet 羽毛球追踪**：保留真实观测、短缺口插值和缺失状态，拒绝跨越大距离的错误插值。
- **真实球场坐标**：将球员落脚点投影到 6.10 m × 13.40 m 标准双打球场，生成移动轨迹、散点图与热力图。
- **俯视角与低视角**：覆盖常见标准转播视角，并提供针对低机位、强透视画面的实验配置。
- **一份可交付的结果集**：输出 H.264 分析视频、逐回合与全场指标、CSV、JSON、图表和可恢复的运行清单。

## 效果展示

### 俯视 / 标准转播视角

在常见转播机位下，BBA 可以同时呈现双端骨骼、球员框、羽毛球轨迹、完整球场边界和标准球场位置。

![2011 世界羽联总决赛俯视角分析](assets/readme/analysis-overhead-bwf2011.png)

### 低视角 / 侧面固定机位

低视角下，远端人物更小、遮挡更强，部分球场角点还会位于画面之外。BBA 使用独立的低视角配置与标准球场模型，仍可保留双端骨骼、球路和场地投影；自动结果不可靠时可由用户手动修正。

![BBA 低视角分析与结果界面](assets/readme/analysis-low-angle-lindan2026.png)

## 从上传视频到查看报告

1. **上传素材**：选择未裁切的完整转播，或已经裁切好的比赛片段。
2. **选择机位**：选择俯视 / 标准转播视角，或低视角 / 侧面固定机位。
3. **核对球场**：接受自动标定，或在代表帧上用标准球场模型手动修正。
4. **启动分析**：界面显示真实阶段、帧级进度、已运行时间和基于实际吞吐量计算的预计剩余时间。
5. **查看结果**：播放带骨骼和轨迹的分析视频，浏览全场与逐回合指标，并下载图表和原始数据。

<table>
  <tr>
    <td width="50%"><img src="assets/readme/webui-home.png" alt="BBA WebUI 首页与分析流程"></td>
    <td width="50%"><img src="assets/readme/webui-analysis-progress.png" alt="BBA WebUI 真实分析进度"></td>
  </tr>
  <tr>
    <td align="center">清晰的四步工作流</td>
    <td align="center">真实阶段、帧进度与动态 ETA</td>
  </tr>
</table>

## 一键启动 WebUI

### Windows

安装 [Conda / Miniconda](https://docs.conda.io/projects/miniconda/en/latest/) 后，在克隆仓库的根目录打开 CMD，运行：

```cmd
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File .\start_webui.ps1
```

也可以双击 `start_webui.bat`。启动完成后访问：

```text
http://127.0.0.1:7860
```

项目统一使用已有的 `good-badminton` Conda 环境。WebUI 启动前会严格检查完整生产套件；若依赖不完整，会调用 `setup_runtime.ps1` 在这个环境内一次性同步 WebUI、CUDA、RTMPose、BST、统计评估和报告依赖，不会创建第二个环境。脚本还会准备并校验固定版本的 BST 上游源码与官方集成权重，避免换机后静默退回无分类。正式分析的耗时取决于视频长度、分辨率、帧率、显卡和可用回合数量。

## 得到的不只是一段标注视频

| 输出 | 可以回答的问题 |
| --- | --- |
| 双端骨骼与球路分析视频 | 球员在什么时候移动、起跳、降低重心，球经过了哪里？ |
| 全场与逐回合球员指标 | 移动距离、平均 / 最高速度、追踪覆盖率、骨骼有效率分别是多少？ |
| 重心与站位分布 | 平均重心相对高度如何？前场、中场、后场的占比是多少？ |
| 标准球场轨迹、散点图与热力图 | 球员最常出现在哪些区域？不同回合的移动模式有何差异？ |
| 羽毛球观测与图像平面速度 | 哪些帧真正观测到球？轨迹是否连续？图像中的速度变化如何？ |
| 骨骼动作细节分析 | 击球候选、准备 / 加速 / 接触窗 / 随挥 / 恢复阶段、二维关节角、稳定性与步法描述如何？ |
| CSV、JSON 与 Run Manifest | 如何复核结果、继续研究、制作自己的统计或恢复中断任务？ |

骨骼动作细节分析基础版已经接入一键管线与 WebUI。系统使用多证据门槛生成击球候选，缺失或低置信数据会保留拒绝原因；结果页可一键导出带骨骼的三帧复核图、待审核 CSV 和 ZIP 复核包。可选的 BST 击球分类后端需要单独配置作者发布的官方权重，详见 [BST 配置说明](badmintondataprocess/docs/bst_setup.md)。

## 为什么 BBA 适合作为研究与训练工具

- **先验证，再计算**：只有通过质量检查的球场标定才能产生正式米制指标。
- **自动化但允许人工接管**：手动标定使用标准球场模型，角点可以自然延伸到视频画面之外。
- **结果可追溯**：每个阶段都记录输入、状态、质量摘要与产物，失败、无数据和成功不会混为一谈。
- **可恢复运行**：长视频中断后可继续处理；输入或配置发生变化时会拒绝混用旧结果。
- **本地 GPU 工作流**：生产配置支持 NVIDIA CUDA、RTMPose 与 TrackNet GPU 推理。

## 当前能力边界

BBA 已能完整演示从原始视频到分析报告的流程，但仍是持续发展的研究项目：

- 俯视 / 标准转播视角是当前主要验证范围；低视角、严重遮挡和频繁换机位仍属于实验场景。
- `near` / `far` 表示球网两侧的场地角色，不等同于跨回合保持不变的运动员身份。
- 单目视频与地面 Homography 无法可靠恢复空中羽毛球的三维米制速度或正式落点；当前正式报告以图像平面速度和可见性为主。
- 击球候选与阶段分解仍需冻结人工标注集验证；当前稳定性和步法是二维描述符，不是医学诊断、三维关节测量或绝对动作优劣评分。

我们宁愿明确显示“数据不足”或“开发中”，也不会用看似精确的数字掩盖证据不足。

## 基于开源成果继续前进

BBA **并非从零开始**。本项目基于 [yo-WASSUP/Good-Badminton](https://github.com/yo-WASSUP/Good-Badminton) 继续开发，感谢上游作者 yo-WASSUP 与贡献者在球员姿态、羽毛球检测和球场映射方向提供的项目基础与探索。

在此基础上，BBA 进一步建设了统一且可恢复的 `badmintondataprocess/` 研究管线、完整转播清洗、质量门控、Run Manifest、批处理、标准球场手动校准、俯视 / 低视角配置以及面向普通用户的一键 WebUI。根目录旧版演示入口只保留用于兼容，不再承载新的算法开发。

同时感谢 RTMPose / RTMO / OpenMMLab、[rtmlib](https://github.com/Tau-J/rtmlib)、[Ultralytics](https://github.com/ultralytics/ultralytics) 与 [TrackNet](https://github.com/yastrebksv/TrackNet) 提供的算法和工程基础。

<details>
<summary><strong>面向开发者：环境、CLI 与运行产物</strong></summary>

### 手动安装

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\setup_runtime.ps1
conda run -n good-badminton bdp verify --profile production --strict --bst-repository .\third_party\BST-Badminton-Stroke-type-Transformer --bst-weights .\weights\bst\bst_AP_JnB_bone_train_partial_0p25_merged_2.pt
```

安装脚本只更新已有的 `good-badminton` 环境。完整直接依赖锁定在 `badmintondataprocess/requirements-runtime.txt`；RTMLib 会在保留 `onnxruntime-gpu` 的前提下单独安装，避免 CPU/GPU ONNX Runtime 混装。

### 一条命令分析完整转播

```powershell
bdp analyze F:\material\match.mp4 --run-id match_full_analysis
```

### 主要命令

```text
bdp analyze <video>             # 一键完整分析
bdp pipeline run / batch        # 分阶段 / 批量运行
bdp calibrate                   # 球场标定
bdp track players / shuttle     # 球员 / 羽毛球追踪
bdp render demo                 # 重渲染分析视频
bdp webui                       # 启动浏览器界面
bdp verify --profile production --strict  # 完整生产环境自检
```

每次运行的核心产物位于 `badmintondataprocess/runs/<run-id>/`，包括 `manifest.json`、`analysis_summary.json`、CSV 轨迹、图表和 `outputs/demo/badminton_full_analysis.mp4`。

当前自动化测试套件包含 **176 项测试**。

</details>

## License

本项目沿用上游的 [Apache License 2.0](LICENSE)。使用、修改或再发布前，请同时保留许可证与上游归属说明。
