# promptplan.md

## 项目初步计划：基于职业羽毛球比赛录播的运动员与羽毛球轨迹识别及战术分析

### 1. 项目目标

本项目目标是从过去职业羽毛球大赛的官方录播视频中，自动或半自动提取：

- 运动员运动轨迹
- 羽毛球飞行轨迹
- 击球点与落点
- 每回合 rally 切分结果
- 球员站位热力图
- 移动距离、速度、覆盖范围等战术指标

最终形成一个可用于论文级数据分析的视觉分析框架。

推荐论文题目：

> 基于职业羽毛球比赛转播视频的运动员与羽毛球轨迹识别及战术分析研究

英文题目：

> A Vision-based Player and Shuttle Trajectory Analysis Framework for Broadcast Badminton Videos

---

## 2. 数据来源规划

### 2.1 推荐数据来源

优先使用官方公开渠道，避免版权风险：

1. BWF TV 官方 YouTube 频道
2. Olympics 官方回放 / Olympics.com
3. 赛事官方频道或授权平台
4. 电视台或平台授权回放

不建议把非官方搬运频道作为核心实验数据源。

### 2.2 推荐赛事

优先选择画质高、主视角稳定、转播切镜头较少的大赛：

- BWF World Championships
- All England Open
- Indonesia Open
- China Open
- Malaysia Open
- Japan Open
- Denmark Open
- BWF World Tour Finals
- Thomas Cup
- Uber Cup
- Sudirman Cup
- Olympic Games

### 2.3 数据记录字段

建立 `matches.csv` 或 `metadata.json`，记录每场比赛信息：

| 字段 | 示例 |
|---|---|
| match_id | MS_2024_AllEngland_Final_001 |
| source | BWF TV |
| url | official replay url |
| tournament | All England Open |
| year | 2024 |
| discipline | MS / WS / MD / WD / XD |
| round | Final |
| player_1 | Player A |
| player_2 | Player B |
| resolution | 1080p / 4K |
| fps | 30 / 60 |
| camera_type | broadcast |
| usable_rallies | number |
| notes | replay cuts / scoreboard occlusion / camera switch |

---

## 3. 数据处理总流程

```text
完整比赛录播
  ↓
去除片头、暂停、采访、慢动作、观众镜头
  ↓
保留主视角 rally 片段
  ↓
按 rally 切分
  ↓
球场关键点标定 / court calibration
  ↓
运动员检测与跟踪
  ↓
人体姿态估计
  ↓
羽毛球轨迹检测
  ↓
轨迹补全和平滑
  ↓
击球点、落点、移动距离、热力图、战术指标分析
```

---

## 4. 推荐技术路线

### 4.1 运动员轨迹识别

推荐方法：

```text
YOLO / RT-DETR 检测运动员
  ↓
ByteTrack / BoT-SORT 跟踪运动员 ID
  ↓
RTMPose / ViTPose 提取人体关键点
  ↓
使用脚踝或脚底点作为球员位置
  ↓
Homography 映射到标准羽毛球场坐标
```

注意：论文级分析不建议直接使用 bbox center 表示球员位置。更推荐使用：

```text
position = left_ankle 和 right_ankle 的中点
```

或者使用支撑脚点作为球员真实站位。

### 4.2 羽毛球轨迹识别

不建议只使用 YOLO 检测羽毛球。原因：

- 羽毛球目标极小
- 速度快
- 容易拖影
- 容易被球员、球拍、灯光、白线干扰
- 转播视频压缩后检测难度更高

推荐方法：

```text
TrackNetV3 / TrackNetV4
  ↓
多帧输入检测羽毛球热力图
  ↓
soft-argmax / argmax 得到球心坐标
  ↓
Kalman Filter / spline / physical constraint 补全和平滑轨迹
```

### 4.3 球场坐标映射

运动员位置可以通过 Homography 从图像坐标映射到标准羽毛球场平面坐标。

流程：

```text
标注球场角点 / 边线交点
  ↓
建立图像坐标和标准球场坐标对应关系
  ↓
计算 Homography 矩阵 H
  ↓
将运动员脚底点映射到真实场地坐标
```

注意：羽毛球在空中，不在地面平面上，因此不能简单使用 Homography 得到真实 3D 轨迹。单机位只能得到 2D 图像轨迹或地面投影估计。若要真实 3D 羽毛球轨迹，建议使用双机位或多机位同步拍摄。

---

## 5. 论文级系统架构

```text
                    ┌────────────────────┐
                    │  Badminton Video    │
                    └─────────┬──────────┘
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
          ▼                   ▼                   ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ Court Detection │  │ Player Tracking │  │ Shuttle Tracking│
│ / Calibration   │  │ YOLO + BoT-SORT │  │ TrackNetV4      │
└────────┬────────┘  └────────┬────────┘  └────────┬────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ Homography H    │  │ Pose Estimation │  │ Trajectory      │
│ Image → Court   │  │ RTMPose/ViTPose │  │ Rectification   │
└────────┬────────┘  └────────┬────────┘  └────────┬────────┘
         │                    │                    │
         └────────────────────┼────────────────────┘
                              ▼
                  ┌──────────────────────┐
                  │ Spatio-temporal Data │
                  │ Fusion               │
                  └──────────┬───────────┘
                             ▼
                  ┌──────────────────────┐
                  │ Tactical / Movement  │
                  │ Analysis             │
                  └──────────────────────┘
```

---

## 6. 建议数据集结构

```text
badminton_dataset/
├── raw_videos/
│   ├── match_001.mp4
│   └── match_002.mp4
├── metadata/
│   ├── matches.csv
│   └── video_sources.json
├── rallies/
│   ├── match_001_rally_001.mp4
│   ├── match_001_rally_002.mp4
│   └── match_002_rally_001.mp4
├── annotations/
│   ├── court_keypoints.json
│   ├── player_tracks.csv
│   ├── shuttle_tracks.csv
│   ├── hit_events.csv
│   └── landing_points.csv
├── outputs/
│   ├── trajectory_videos/
│   ├── heatmaps/
│   └── tactical_statistics.csv
└── scripts/
    ├── video_preprocess.py
    ├── rally_segmentation.py
    ├── court_calibration.py
    ├── player_tracking.py
    ├── shuttle_tracking.py
    ├── trajectory_smoothing.py
    └── visualize_results.py
```

---

## 7. 标注设计

为了满足论文级评估，需要构建人工标注测试集。

建议至少标注：

| 标注对象 | 标注内容 | 用途 |
|---|---|---|
| 球场 | 角点、边线交点、网柱点 | Homography / calibration |
| 运动员 | bbox、ID、脚踝/脚底点 | 运动员检测与跟踪评估 |
| 羽毛球 | 每帧球心坐标 `(x, y)` | 羽毛球轨迹评估 |
| 击球事件 | hit frame | 击球点分析、rally 分割 |
| 落点 | landing point / court zone | 战术统计 |
| 动作类型 | clear / drop / smash / lift / drive / net | 击球类型识别 |

推荐初期标注规模：

```text
10–20 场完整比赛
100–300 个 rally
人工精标 20–50 个 rally 作为测试集
其余用于弱标注和统计分析
```

---

## 8. 评价指标

### 8.1 运动员检测指标

- mAP@0.5
- mAP@0.5:0.95
- Precision
- Recall

### 8.2 运动员跟踪指标

- MOTA
- IDF1
- HOTA
- ID Switches
- Track Fragmentation

### 8.3 球场坐标误差

```text
Mean Position Error, MPE
单位：cm 或 m
```

公式：

```text
MPE = mean(|| predicted_position - ground_truth_position ||_2)
```

### 8.4 羽毛球检测指标

- Pixel Error
- Accuracy@5px
- Accuracy@10px
- Precision
- Recall
- F1-score
- Missing Rate

定义：

```text
Acc@5px  = 预测点距离人工标注点 ≤ 5 像素的比例
Acc@10px = 预测点距离人工标注点 ≤ 10 像素的比例
```

### 8.5 击球点和落点指标

- Hit Frame Error
- Hit Point Error
- Landing Point Error
- Zone Classification Accuracy

定义：

```text
Hit Frame Error = |predicted_hit_frame - ground_truth_hit_frame|
```

---

## 9. 实验设计

### Experiment 1：运动员检测与跟踪对比

| Method | Detector | Tracker | MOTA | IDF1 | HOTA | IDSW |
|---|---|---|---:|---:|---:|---:|
| YOLO + DeepSORT | YOLO | DeepSORT | - | - | - | - |
| YOLO + ByteTrack | YOLO | ByteTrack | - | - | - | - |
| YOLO + BoT-SORT | YOLO | BoT-SORT | - | - | - | - |
| RT-DETR + BoT-SORT | RT-DETR | BoT-SORT | - | - | - | - |

### Experiment 2：羽毛球轨迹检测对比

| Method | Temporal Input | Acc@5px | Acc@10px | Missing Rate | FPS |
|---|---:|---:|---:|---:|---:|
| YOLO | 单帧 | - | - | - | - |
| YOLO + Kalman | 单帧 + 滤波 | - | - | - | - |
| TrackNetV3 | 多帧 | - | - | - | - |
| TrackNetV4 | 多帧 + motion attention | - | - | - | - |
| Proposed | TrackNetV4 + 补全 + 物理约束 | - | - | - | - |

### Experiment 3：球场映射误差

| Method | Court Keypoints | Mean Error/cm | Max Error/cm |
|---|---:|---:|---:|
| Manual Homography | 手动 | - | - |
| Auto Court Detection | 自动 | - | - |
| Auto + Manual Correction | 半自动 | - | - |

### Experiment 4：战术特征分析

分析内容包括：

- 双方平均站位深度
- 前后场覆盖比例
- 左右移动距离
- 每回合跑动距离
- 击球点热力图
- 落点热力图
- 杀球后防守移动模式
- 多拍回合中的位置恢复速度

---

## 10. 阶段性开发计划

### Phase 1：数据收集与整理

目标：建立小规模官方录播数据集。

任务：

- 收集 10–20 场官方职业比赛录播链接
- 建立 `matches.csv`
- 下载或本地缓存研究用视频
- 人工检查视频质量、视角、帧率、遮挡情况

输出：

```text
metadata/matches.csv
raw_videos/
```

### Phase 2：视频预处理与 rally 切分

目标：从完整录播中提取可分析的主视角 rally 片段。

任务：

- 去除片头、暂停、慢动作、回放、观众镜头
- 根据比分牌、画面变化或人工标注切分 rally
- 保存每个 rally 视频片段

输出：

```text
rallies/match_xxx_rally_xxx.mp4
```

### Phase 3：球场标定

目标：建立图像坐标到标准羽毛球场坐标的映射。

任务：

- 标注球场边线交点
- 计算 Homography 矩阵
- 验证场地坐标误差

输出：

```text
annotations/court_keypoints.json
outputs/court_projection_check.mp4
```

### Phase 4：运动员轨迹识别

目标：获得每个 rally 中双方球员轨迹。

任务：

- 使用 YOLO / RT-DETR 检测球员
- 使用 BoT-SORT / ByteTrack 跟踪 ID
- 使用 RTMPose / ViTPose 提取脚踝点
- 映射到标准球场坐标

输出：

```text
annotations/player_tracks.csv
outputs/player_trajectory.mp4
outputs/player_heatmap.png
```

### Phase 5：羽毛球轨迹识别

目标：获得羽毛球时序轨迹。

任务：

- 使用 TrackNetV3 / TrackNetV4 检测羽毛球
- 过滤低置信度预测
- 用 Kalman Filter / spline 补全缺失轨迹
- 判断异常轨迹点

输出：

```text
annotations/shuttle_tracks.csv
outputs/shuttle_trajectory.mp4
```

### Phase 6：击球点、落点与战术分析

目标：把轨迹转化为论文可用的战术统计。

任务：

- 检测击球帧
- 估计击球点
- 估计落点或落点区域
- 统计球员跑动距离、速度、站位、击球区域、落点分布

输出：

```text
annotations/hit_events.csv
annotations/landing_points.csv
outputs/tactical_statistics.csv
outputs/heatmaps/
```

---

## 11. 给 Claude / Copilot 的开发 Prompt

### Prompt 1：项目总架构设计

```text
你是一个计算机视觉和体育视频分析专家。请帮我设计一个论文级项目：基于职业羽毛球比赛转播视频的运动员与羽毛球轨迹识别及战术分析系统。

要求：
1. 输入为 BWF/Olympics 等官方职业比赛录播视频。
2. 系统需要完成 rally 切分、球场标定、运动员检测与跟踪、人体姿态估计、羽毛球轨迹检测、轨迹补全、击球点与落点分析。
3. 运动员轨迹使用 YOLO/RT-DETR + BoT-SORT/ByteTrack + RTMPose/ViTPose + Homography。
4. 羽毛球轨迹优先使用 TrackNetV3/TrackNetV4，而不是单纯 YOLO。
5. 输出需要包括 player_tracks.csv、shuttle_tracks.csv、hit_events.csv、landing_points.csv、trajectory visualization video、heatmap 和 tactical_statistics.csv。
6. 请给出完整的代码目录结构、每个模块输入输出、核心算法流程和开发顺序。
7. 方案必须适合写入毕业论文或科研论文。
```

### Prompt 2：视频预处理与 rally 切分

```text
请为羽毛球职业比赛转播视频设计一个 rally segmentation 模块。

输入：完整比赛录播 mp4。
输出：每个 rally 的短视频片段和对应 metadata。

要求：
1. 去除片头、暂停、慢动作、回放、观众镜头、近景镜头。
2. 尽可能保留主视角比赛画面。
3. 可以结合画面变化、比分牌区域、场地线可见性、运动员位置和人工校正。
4. 输出 rallies.csv，字段包括 match_id、rally_id、start_frame、end_frame、start_time、end_time、notes。
5. 给出 Python + OpenCV 的实现思路和伪代码。
```

### Prompt 3：球场标定模块

```text
请实现一个羽毛球场地标定模块。

目标：将视频中的图像坐标映射到标准羽毛球场平面坐标。

要求：
1. 支持手动点击球场角点和边线交点。
2. 根据标准羽毛球场尺寸建立 world/court coordinate system。
3. 使用 OpenCV 计算 Homography 矩阵。
4. 将运动员脚底点从 image coordinate 转换到 court coordinate。
5. 保存 court_keypoints.json 和 homography_matrix.npy。
6. 提供 Python 代码结构和关键函数。
```

### Prompt 4：运动员轨迹模块

```text
请实现一个羽毛球运动员轨迹识别模块。

输入：rally 视频片段。
输出：player_tracks.csv 和可视化视频。

要求：
1. 使用 YOLO 或 RT-DETR 检测运动员。
2. 使用 BoT-SORT 或 ByteTrack 进行多目标跟踪。
3. 使用 RTMPose 或 ViTPose 提取人体脚踝关键点。
4. 使用脚踝中点作为运动员位置，而不是 bbox center。
5. 使用 Homography 将位置映射到标准羽毛球场坐标。
6. 输出字段包括 frame_id、timestamp、player_id、bbox、ankle_left、ankle_right、image_x、image_y、court_x、court_y、confidence。
7. 给出模块代码结构和伪代码。
```

### Prompt 5：羽毛球轨迹模块

```text
请实现一个论文级羽毛球轨迹识别模块。

输入：rally 视频片段。
输出：shuttle_tracks.csv 和可视化视频。

要求：
1. 优先使用 TrackNetV3 或 TrackNetV4，而不是只使用 YOLO。
2. 输入连续多帧，输出每帧羽毛球热力图。
3. 将热力图转换为球心坐标。
4. 对低置信度和缺失帧使用 Kalman Filter、spline smoothing 或物理约束进行补全。
5. 输出字段包括 frame_id、timestamp、x、y、confidence、is_interpolated、visibility。
6. 需要给出异常点过滤策略，例如速度突变、轨迹不连续、位置跳变。
7. 给出 Python 代码结构、伪代码和评估指标。
```

### Prompt 6：论文实验设计

```text
请帮我设计这个羽毛球轨迹分析项目的论文实验部分。

要求：
1. 分别评估运动员检测、运动员跟踪、球场坐标映射、羽毛球轨迹检测、击球点检测和落点分析。
2. 指标包括 mAP、MOTA、IDF1、HOTA、MPE、Acc@5px、Acc@10px、Missing Rate、Hit Frame Error、Landing Point Error。
3. 设计 baseline：YOLO、YOLO+Kalman、TrackNetV3、TrackNetV4、Proposed Method。
4. 设计消融实验：无姿态关键点、无轨迹补全、无物理约束、无 Homography 校正。
5. 输出实验表格模板和论文文字描述。
```

---

## 12. 版权与学术规范

为降低版权风险，建议：

- 只使用官方公开或授权回放作为研究数据来源。
- 本地视频仅用于算法研究，不公开传播完整原视频。
- 论文中记录原始 URL、赛事、年份、轮次、球员等 metadata。
- 公开数据集时只发布标注、坐标、帧号、统计结果，不发布原始比赛视频。
- 展示结果时尽量使用轨迹图、热力图、统计图，而不是长片段原视频。
- 如需公开 demo，优先使用自己拍摄或已获得授权的视频。

---

## 13. 初步结论

本项目的核心路线是：

```text
数据来源：官方职业比赛录播
数据处理：完整录播 → 主视角 rally → 标定 → 轨迹识别 → 战术分析
运动员轨迹：YOLO/RT-DETR + BoT-SORT + Pose + Homography
羽毛球轨迹：TrackNetV3/V4 + 轨迹补全 + 平滑 + 物理约束
论文重点：准确性评估、消融实验、战术统计分析、可视化结果
```

最重要的技术判断：

```text
YOLO 可以用于运动员检测，但不应作为羽毛球轨迹识别的唯一核心方法。
羽毛球轨迹应优先使用 TrackNet 系列等多帧小目标轨迹模型。
如果只做 2D 战术分析，单机位官方转播视频可行。
如果要真实 3D 羽毛球轨迹，需要双机位或多机位同步视频。
```
