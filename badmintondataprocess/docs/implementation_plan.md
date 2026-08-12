# 实施计划

## 1. 总体原则

本项目优先采用“先可运行、再可评估、后可论文化”的推进方式。

开发主线如下：

1. 数据闭环先行：先打通从原始视频到 rally 片段的流程
2. 标定先于分析：所有战术指标建立在稳定的球场映射之上
3. 运动员与羽毛球轨迹分开开发、后期融合
4. 先建立人工测试集，再进行模型比较与消融实验

## 2. 里程碑规划

### Milestone A：数据资产建立

目标：整理研究可用的官方比赛视频样本。

输入：官方比赛链接或本地视频。
输出：`metadata/matches.csv`、`raw_videos/`。
验收标准：

- 至少收集 10 场比赛元数据
- 每场视频可读取分辨率、帧率、时长
- notes 字段记录切镜头、遮挡、慢动作等信息

### Milestone B：rally 切分原型

目标：从完整录播中提取主视角 rally 视频。

输入：`raw_videos/*.mp4`
输出：`rallies/*.mp4`、后续可补 `metadata/rallies.csv`
验收标准：

- 能批量读取完整录播
- 能输出候选 rally 时间段
- 支持人工校正起止帧

### Milestone C：球场标定原型

目标：建立图像平面到标准羽毛球场平面的映射。

输入：rally 首帧或代表帧。
输出：`annotations/court_keypoints.json`、`outputs/court_projection_check.mp4`
验收标准：

- 支持手动标点
- 自动计算 Homography
- 可视化映射结果无遮挡严重偏移

### Milestone D：运动员轨迹模块

目标：输出球员图像轨迹与球场坐标轨迹。

输入：`rallies/*.mp4` 与 Homography。
输出：`annotations/player_tracks.csv`
验收标准：

- 每帧具有稳定 player_id
- 位置优先使用双脚踝中点
- 坐标可映射到球场平面

### Milestone E：羽毛球轨迹模块

目标：输出可分析的羽毛球时序轨迹。

输入：`rallies/*.mp4`
输出：`annotations/shuttle_tracks.csv`
验收标准：

- 支持多帧输入模型接口
- 记录置信度与插值标记
- 过滤明显跳点

### Milestone F：战术分析与论文实验

目标：形成统计结果、图表与实验表格。

输入：运动员轨迹、羽毛球轨迹、击球点、落点。
输出：`outputs/tactical_statistics.csv`、热力图与论文实验表。
验收标准：

- 输出站位、跑动距离、覆盖范围等指标
- 可生成热力图
- 可形成 baseline 与 ablation 表格

## 3. 推荐优先级

### 第一周

- 填写 `metadata/matches.csv`
- 建立 3 到 5 场先导样本
- 确认视频命名规范与 match_id 规则
- 完成视频读取与元数据检查脚本

### 第二周

- 实现 rally segmentation 原型
- 增加人工校正接口
- 产出一批 rally 短视频样本

### 第三周

- 实现球场关键点标注工具
- 输出 Homography 与投影验证图

### 第四周至第六周

- 完成运动员检测、跟踪、姿态点融合
- 完成羽毛球轨迹模型接入
- 加入轨迹平滑与异常过滤

### 第七周以后

- 建立测试集
- 设计实验与对比表
- 输出论文图表与可视化视频

## 4. 数据规范

### 4.1 match_id 规范

建议格式：`discipline_year_tournament_round_index`

示例：`MS_2024_AllEngland_Final_001`

### 4.2 原始视频命名

建议格式：`{match_id}.mp4`

### 4.3 rally 命名

建议格式：`{match_id}_rally_{rally_id:03d}.mp4`

### 4.4 CSV 输出规范

所有输出 CSV 建议至少包含：

- `match_id`
- `rally_id`
- `frame_id`
- `timestamp`
- `source_video`
- `confidence`
- `notes`

## 5. 技术决策

- 运动员检测优先尝试 YOLOv8/YOLO11 或 RT-DETR
- 多目标跟踪优先 ByteTrack 或 BoT-SORT
- 姿态估计优先 RTMPose
- 羽毛球轨迹优先 TrackNet 系列，多帧输入优于单帧检测
- 轨迹补全优先 Kalman Filter + spline 的组合方案
- 战术分析基于 2D 平面坐标，明确不声称恢复真实 3D 球轨迹

## 6. 风险与规避

- 转播切镜头频繁：先筛选主视角稳定赛事
- 记分牌遮挡：在标注测试集内单独记录遮挡情况
- 羽毛球目标过小：优先使用高分辨率样本和多帧模型
- 单机位限制：论文中明确说明 2D 分析边界
- 标注成本高：先建设小规模高质量测试集

## 7. 我建议的下一批实现

优先实现以下 3 个文件中的实际逻辑：

1. `scripts/prepare_matches.py`
2. `scripts/rally_segmentation.py`
3. `scripts/court_calibration.py`
