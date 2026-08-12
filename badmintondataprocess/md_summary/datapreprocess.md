# datapreprocess.md

# 羽毛球大赛录像主视角比赛片段提取数据预处理方案

## 1. 任务目标

本预处理模块的目标是从一段完整的羽毛球大赛录像中，自动提取出适合后续论文级数据分析的“主视角比赛片段”。

完整比赛录像中通常包含：

- 主视角比赛画面
- 球员近景
- 观众席镜头
- 喝水暂停
- 教练指导
- 广告画面
- 慢动作回放
- 裁判或鹰眼挑战
- 转播包装画面
- 入场、赛后采访、颁奖等非比赛内容

本模块希望保留：

```text
高机位、完整球场、双方运动员可见、正在比赛或准备发球的直播主视角画面
```

需要剔除：

```text
观众席、广告、回放、球员近景、教练席、喝水暂停、裁判特写、转播包装、赛后采访等非主视角内容
```

最终输出可以用于后续：

- 运动员轨迹检测
- 羽毛球轨迹检测
- 回合切分
- 技术动作识别
- 战术分析
- 论文级视频数据处理

---

## 2. 总体处理流程

整体流程建议采用：

```text
完整比赛录像
    ↓
镜头切分 Shot Detection
    ↓
片段关键帧采样 Keyframe Sampling
    ↓
片段级分类 Segment Classification
    ↓
时间线平滑 Timeline Smoothing
    ↓
导出主视角比赛视频 Export Clean Video
    ↓
生成帧索引 Frame Index
```

核心思想是：

```text
不要直接对完整视频逐帧做 YOLO 检测，而是先做片段级视角分类。
```

因为完整大赛录像中干扰内容较多，如果直接逐帧检测球、人、姿态，容易被回放、近景、观众席、广告等画面污染。更稳妥的方法是先把视频清洗成只包含主视角比赛画面的 clean video，再进行轨迹和姿态分析。

---

## 3. 推荐项目结构

```text
badminton_video_cleaner/
│
├── configs/
│   └── default.yaml
│
├── scripts/
│   ├── 01_extract_shots.py
│   ├── 02_sample_keyframes.py
│   ├── 03_classify_segments.py
│   ├── 04_smooth_timeline.py
│   ├── 05_export_main_view.py
│   └── run_pipeline.py
│
├── models/
│   ├── court_detector.pt
│   ├── view_classifier.pkl
│   └── replay_logo_detector.pt
│
├── data/
│   ├── raw/
│   │   └── full_match.mp4
│   ├── annotations/
│   └── processed/
│
├── outputs/
│   ├── shots.json
│   ├── keyframes/
│   ├── segment_predictions.json
│   ├── clean_timeline.json
│   ├── main_match_only.mp4
│   └── frame_index.csv
│
├── requirements.txt
└── datapreprocess.md
```

---

## 4. 输入数据

### 4.1 输入视频

输入是完整羽毛球大赛录像，例如：

```text
data/raw/full_match.mp4
```

要求：

- 格式：mp4、mkv、mov 均可
- 分辨率：建议 720p 或以上
- 帧率：25 fps、30 fps、50 fps、60 fps 均可
- 视频内容：可以包含完整直播转播内容

---

## 5. 输出数据

### 5.1 shots.json

镜头切分结果。

示例：

```json
[
  {
    "shot_id": 0,
    "start": 0.0,
    "end": 12.4,
    "duration": 12.4
  },
  {
    "shot_id": 1,
    "start": 12.4,
    "end": 18.8,
    "duration": 6.4
  }
]
```

---

### 5.2 keyframes/

每个镜头片段采样若干关键帧。

示例结构：

```text
outputs/keyframes/
├── segment_0000/
│   ├── frame_000.jpg
│   ├── frame_001.jpg
│   └── frame_002.jpg
├── segment_0001/
│   ├── frame_000.jpg
│   ├── frame_001.jpg
│   └── frame_002.jpg
```

---

### 5.3 segment_predictions.json

片段分类结果。

示例：

```json
[
  {
    "shot_id": 0,
    "start": 0.0,
    "end": 12.4,
    "label": "MAIN_LIVE_VIEW",
    "confidence": 0.91
  },
  {
    "shot_id": 1,
    "start": 12.4,
    "end": 18.8,
    "label": "PLAYER_CLOSEUP",
    "confidence": 0.86
  }
]
```

---

### 5.4 clean_timeline.json

经过时间状态机平滑后的主视角片段时间线。

示例：

```json
[
  {
    "segment_id": 0,
    "start": 21.5,
    "end": 88.2,
    "label": "MAIN_LIVE_VIEW",
    "confidence": 0.92
  },
  {
    "segment_id": 1,
    "start": 103.0,
    "end": 188.6,
    "label": "MAIN_LIVE_VIEW",
    "confidence": 0.89
  }
]
```

---

### 5.5 main_match_only.mp4

最终清洗后的视频，只保留主视角比赛画面。

```text
outputs/main_match_only.mp4
```

---

### 5.6 frame_index.csv

帧索引文件，用于将清洗后视频的每一帧映射回原始视频。

示例：

```csv
clean_frame_id,original_time,original_frame_id,segment_id
0,21.500,645,0
1,21.533,646,0
2,21.566,647,0
3,21.600,648,0
```

该文件对论文级数据分析非常重要，因为后续轨迹检测、姿态估计和回合分析都需要知道每一帧来自原始视频的哪个时间点。

---

## 6. 分类标签设计

建议第一版先使用二分类：

```text
MAIN_LIVE_VIEW
NON_MAIN_VIEW
```

当二分类效果稳定后，再扩展为多分类：

```python
LABELS = [
    "MAIN_LIVE_VIEW",
    "REPLAY",
    "PLAYER_CLOSEUP",
    "COACH_BREAK",
    "AUDIENCE",
    "AD",
    "UMPIRE",
    "OTHER"
]
```

各标签含义如下：

| 标签 | 含义 |
|---|---|
| MAIN_LIVE_VIEW | 主视角比赛画面 |
| REPLAY | 慢动作回放或赛事回放 |
| PLAYER_CLOSEUP | 球员近景、表情、动作特写 |
| COACH_BREAK | 教练指导、喝水、暂停 |
| AUDIENCE | 观众席镜头 |
| AD | 广告、转播包装、赞助商画面 |
| UMPIRE | 裁判、鹰眼挑战、判罚画面 |
| OTHER | 其他无法明确分类的画面 |

---

## 7. 主视角判断标准

主视角比赛画面通常具有以下特征：

```text
1. 能看到完整或接近完整的羽毛球场
2. 镜头为高机位
3. 画面比较稳定
4. 球场线条明显
5. 双方运动员同时出现在球场两侧
6. 人物框相对较小
7. 画面不是慢动作回放
8. 画面不是球员、观众、教练或裁判近景
```

可以融合以下信号：

---

### 7.1 Court Confidence

球场置信度，用于判断画面中是否存在完整羽毛球场。

可选方法：

```text
方法一：HSV 颜色分割 + Hough Line 检测球场线
方法二：YOLOv8-seg / Mask R-CNN 训练球场区域分割模型
方法三：使用关键点检测模型识别球场角点和边线
```

建议输出：

```python
court_confidence = 0.0 ~ 1.0
```

主视角常见阈值：

```python
court_confidence > 0.65
```

---

### 7.2 Main Camera Confidence

主机位置信度，用于判断当前画面是否为直播高机位主视角。

推荐方法：

```text
CLIP / ResNet / ViT 提取关键帧特征
再使用 Logistic Regression、SVM 或 MLP 做二分类
```

训练数据可以从比赛录像中人工标注：

```text
MAIN_LIVE_VIEW
NON_MAIN_VIEW
```

第一版人工标注 500 到 1000 张关键帧即可得到初步可用效果。

---

### 7.3 Person Layout Score

运动员布局分数。

主视角中通常可以检测到：

```text
单打：2 名运动员
双打：4 名运动员
```

判断依据：

```text
1. 人物框数量为 2 到 4
2. 人物框位于球场内部
3. 人物框尺寸相对较小
4. 人物分布在球网两侧
```

示例规则：

```python
if person_count in [2, 3, 4] and players_are_inside_court:
    person_layout_score = 0.8
else:
    person_layout_score = 0.2
```

---

### 7.4 Replay Detection

回放检测。

回放通常具有：

```text
1. 慢动作
2. 非主机位角度
3. 球员或羽毛球特写
4. 画面中可能出现 REPLAY、SLOW MOTION 等字样
5. 画面节奏和正常比赛不同
```

可用方法：

```text
OCR 检测 REPLAY 字样
检测转播回放 logo
光流估计慢动作特征
画面分类器判断 replay / non-replay
```

---

### 7.5 Ad / Graphics Detection

广告和转播包装检测。

常见特征：

```text
1. 大面积文字或 logo
2. 没有完整球场
3. 画面切到品牌宣传、比分板、赛事图形包装
4. 颜色和画面布局与比赛场地明显不同
```

---

## 8. 片段分类逻辑设计

建议使用多信号融合：

```python
def classify_segment(frames):
    court_score = estimate_court_confidence(frames)
    main_score = main_view_classifier.predict(frames)
    replay_score = detect_replay(frames)
    ad_score = detect_ad_or_graphics(frames)
    person_layout_score = estimate_person_layout(frames)

    if ad_score > 0.75:
        return "AD", ad_score

    if replay_score > 0.70:
        return "REPLAY", replay_score

    if court_score < 0.40:
        return "NON_MAIN_VIEW", 1.0 - court_score

    final_main_score = (
        0.40 * court_score +
        0.35 * main_score +
        0.25 * person_layout_score
    )

    if final_main_score > 0.65:
        return "MAIN_LIVE_VIEW", final_main_score

    return "OTHER", final_main_score
```

---

## 9. 时间线平滑策略

直播录像的分类结果可能会出现短暂误判，例如：

```text
主视角 30 秒
球员近景 1 秒
主视角 40 秒
```

如果中间非主视角时间非常短，可以选择合并。

推荐参数：

```python
MIN_MAIN_SEGMENT = 3.0
MAX_GAP_TO_BRIDGE = 2.0
```

规则：

```text
1. 删除短于 3 秒的 MAIN_LIVE_VIEW 片段
2. 合并间隔小于 2 秒的相邻 MAIN_LIVE_VIEW 片段
3. REPLAY 不允许被合并进主视角
4. AD 不允许被合并进主视角
5. COACH_BREAK 不允许被合并进主视角
```

示例代码：

```python
def smooth_segments(segments, min_main_duration=3.0, max_gap=2.0):
    main_segments = []

    for seg in segments:
        duration = seg["end"] - seg["start"]

        if seg["label"] == "MAIN_LIVE_VIEW" and duration >= min_main_duration:
            main_segments.append(seg)

    merged = []

    for seg in main_segments:
        if not merged:
            merged.append(seg)
            continue

        prev = merged[-1]
        gap = seg["start"] - prev["end"]

        if gap <= max_gap:
            prev["end"] = seg["end"]
            prev["confidence"] = max(prev["confidence"], seg["confidence"])
        else:
            merged.append(seg)

    return merged
```

---

## 10. 各脚本设计

---

### 10.1 scripts/01_extract_shots.py

功能：

```text
对完整视频进行镜头切分，输出 shots.json
```

推荐工具：

```text
PySceneDetect
OpenCV scene change detection
ffmpeg scene detection
```

输入：

```bash
python scripts/01_extract_shots.py \
  --video data/raw/full_match.mp4 \
  --out outputs/shots.json
```

输出：

```text
outputs/shots.json
```

---

### 10.2 scripts/02_sample_keyframes.py

功能：

```text
对每个 shot 采样 3 到 5 张关键帧
```

输入：

```bash
python scripts/02_sample_keyframes.py \
  --video data/raw/full_match.mp4 \
  --shots outputs/shots.json \
  --out outputs/keyframes
```

采样策略：

```text
每个片段采样 start / middle / end 三个位置
如果片段较长，可以额外采样 1/4 和 3/4 位置
```

---

### 10.3 scripts/03_classify_segments.py

功能：

```text
对每个 shot 进行片段级分类
```

输入：

```bash
python scripts/03_classify_segments.py \
  --shots outputs/shots.json \
  --frames outputs/keyframes \
  --out outputs/segment_predictions.json
```

输出：

```text
outputs/segment_predictions.json
```

分类模块建议包含：

```python
estimate_court_confidence(frames)
estimate_person_layout(frames)
detect_replay(frames)
detect_ad_or_graphics(frames)
classify_segment(frames)
```

第一版可以先实现占位规则，后续逐步接入：

```text
YOLOv8
CLIP
ResNet
ViT
OCR
Hough Line
Optical Flow
```

---

### 10.4 scripts/04_smooth_timeline.py

功能：

```text
对分类结果进行时间线平滑，只保留可靠主视角比赛片段
```

输入：

```bash
python scripts/04_smooth_timeline.py \
  --input outputs/segment_predictions.json \
  --out outputs/clean_timeline.json
```

输出：

```text
outputs/clean_timeline.json
```

---

### 10.5 scripts/05_export_main_view.py

功能：

```text
根据 clean_timeline.json 导出主视角比赛视频
同时生成 frame_index.csv
```

输入：

```bash
python scripts/05_export_main_view.py \
  --video data/raw/full_match.mp4 \
  --timeline outputs/clean_timeline.json \
  --out outputs/main_match_only.mp4 \
  --frame-index outputs/frame_index.csv
```

导出方式：

```text
使用 ffmpeg 根据时间段切片
再使用 concat 合并片段
```

注意：

```text
如果需要严格帧级对齐，不建议只使用 -c copy。
-c copy 速度快，但可能会因为关键帧导致切片时间不够精确。
论文级分析建议使用重新编码方式。
```

推荐命令逻辑：

```bash
ffmpeg -ss START -i input.mp4 -t DURATION -c:v libx264 -c:a aac clip_xxxx.mp4
```

---

### 10.6 scripts/run_pipeline.py

功能：

```text
一键运行完整预处理流程
```

示例：

```bash
python scripts/run_pipeline.py \
  --video data/raw/full_match.mp4 \
  --out-dir outputs
```

执行顺序：

```text
01_extract_shots.py
02_sample_keyframes.py
03_classify_segments.py
04_smooth_timeline.py
05_export_main_view.py
```

---

## 11. 一键运行脚本示例

```python
import subprocess
from pathlib import Path

VIDEO = "data/raw/full_match.mp4"
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

steps = [
    [
        "python", "scripts/01_extract_shots.py",
        "--video", VIDEO,
        "--out", "outputs/shots.json"
    ],
    [
        "python", "scripts/02_sample_keyframes.py",
        "--video", VIDEO,
        "--shots", "outputs/shots.json",
        "--out", "outputs/keyframes"
    ],
    [
        "python", "scripts/03_classify_segments.py",
        "--shots", "outputs/shots.json",
        "--frames", "outputs/keyframes",
        "--out", "outputs/segment_predictions.json"
    ],
    [
        "python", "scripts/04_smooth_timeline.py",
        "--input", "outputs/segment_predictions.json",
        "--out", "outputs/clean_timeline.json"
    ],
    [
        "python", "scripts/05_export_main_view.py",
        "--video", VIDEO,
        "--timeline", "outputs/clean_timeline.json",
        "--out", "outputs/main_match_only.mp4",
        "--frame-index", "outputs/frame_index.csv"
    ],
]

for cmd in steps:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
```

---

## 12. 人工标注建议

为了训练主视角分类器，建议从不同比赛中采样关键帧并人工标注。

第一版标注：

```text
MAIN_LIVE_VIEW
NON_MAIN_VIEW
```

推荐数量：

```text
500 到 1000 张关键帧
```

后续扩展多分类时，可以标注：

```text
MAIN_LIVE_VIEW
REPLAY
PLAYER_CLOSEUP
COACH_BREAK
AUDIENCE
AD
UMPIRE
OTHER
```

标注建议：

```text
1. 不同赛事、不同场馆、不同转播风格都要覆盖
2. 单打和双打都要覆盖
3. 主视角和回放视角要严格区分
4. 球员近景即使有羽毛球也应该标为 NON_MAIN_VIEW
5. 慢动作回放即使是球场画面，也应该标为 REPLAY
```

---

## 13. 第一版 MVP 实现建议

第一版不需要一次性做复杂模型。

推荐最小可行版本：

```text
1. 使用 PySceneDetect 做镜头切分
2. 每个 shot 抽 3 到 5 张关键帧
3. 人工标注 500 到 1000 张图像
4. 使用 CLIP 或 ResNet 提取图像特征
5. 训练二分类器判断 MAIN_LIVE_VIEW / NON_MAIN_VIEW
6. 使用状态机合并和过滤时间线
7. 用 ffmpeg 导出 clean video
```

第一版目标：

```text
能够从完整比赛录像中较稳定地提取主视角比赛片段
```

不是第一版目标：

```text
精确检测每一帧羽毛球位置
精确识别每一个技术动作
精确判断每个回合边界
```

这些任务应该放在 clean video 生成之后进行。

---

## 14. 后续扩展方向

### 14.1 加入球场检测

使用 YOLOv8-seg 或关键点检测模型检测球场区域。

可输出：

```text
court_bbox
court_mask
court_keypoints
court_confidence
```

用途：

```text
1. 提升主视角识别准确率
2. 对运动员位置进行场地坐标归一化
3. 支持后续战术空间分析
```

---

### 14.2 加入运动员检测

使用 YOLOv8 / RT-DETR / Grounding DINO 等模型检测运动员。

用途：

```text
1. 过滤非比赛画面
2. 检查单打/双打人数
3. 后续运动员轨迹跟踪
```

---

### 14.3 加入羽毛球检测

在 clean video 上使用专门的 shuttlecock detector。

注意：

```text
羽毛球体积很小，直接在完整直播视频上检测难度较大。
建议先完成主视角清洗，再做羽毛球轨迹检测。
```

---

### 14.4 加入回合切分

在主视角比赛视频上继续切分 rally：

```text
发球开始
回合进行
死球
比分更新
下一回合准备
```

可能使用信号：

```text
球员站位
球速变化
羽毛球轨迹中断
裁判声音
比分板变化
画面节奏
```

---

### 14.5 加入人工校正界面

可以开发一个简单网页或 Gradio 工具，用于快速检查和修正片段标签：

```text
左侧：视频片段
右侧：当前预测标签
按钮：MAIN / REPLAY / CLOSEUP / AD / OTHER
保存：更新 segment_predictions.json
```

这样可以快速积累高质量训练数据。

---

## 15. 论文级数据处理注意事项

如果目标是论文级分析，建议特别注意：

```text
1. 保存原始视频，不覆盖
2. 保存所有中间结果
3. clean video 必须能回溯到 original video
4. 每一帧都要有 original timestamp 映射
5. 所有阈值和模型版本要记录在 config 文件中
6. 对不同赛事的视频要做泛化测试
7. 分类结果最好保留 confidence
8. 需要人工抽样检查 precision 和 recall
```

建议额外保存：

```text
config.yaml
model_version.txt
processing_log.txt
manual_review.csv
```

---

## 16. 推荐配置文件 default.yaml

```yaml
input:
  video_path: data/raw/full_match.mp4

output:
  out_dir: outputs
  shots_json: outputs/shots.json
  keyframes_dir: outputs/keyframes
  predictions_json: outputs/segment_predictions.json
  clean_timeline_json: outputs/clean_timeline.json
  clean_video: outputs/main_match_only.mp4
  frame_index_csv: outputs/frame_index.csv

shot_detection:
  method: pyscenedetect
  threshold: 27.0
  min_scene_len: 15

keyframe_sampling:
  num_frames_per_shot: 5
  strategy: uniform

classification:
  labels:
    - MAIN_LIVE_VIEW
    - REPLAY
    - PLAYER_CLOSEUP
    - COACH_BREAK
    - AUDIENCE
    - AD
    - UMPIRE
    - OTHER

  main_view_threshold: 0.65
  court_threshold: 0.60
  replay_threshold: 0.70
  ad_threshold: 0.75

smoothing:
  min_main_duration: 3.0
  max_gap_to_bridge: 2.0
  forbidden_bridge_labels:
    - REPLAY
    - AD
    - COACH_BREAK

export:
  codec: libx264
  audio_codec: aac
  precise_cut: true
```

---

## 17. requirements.txt 建议

```text
opencv-python
numpy
pandas
tqdm
scenedetect
moviepy
PyYAML
scikit-learn
torch
torchvision
ultralytics
Pillow
```

如果使用 OCR：

```text
easyocr
```

如果使用 CLIP：

```text
open-clip-torch
```

---

## 18. 开发优先级

建议按以下顺序开发：

```text
Priority 1:
- 镜头切分
- 关键帧采样
- 手工标注
- 二分类主视角识别
- 导出 clean video

Priority 2:
- 时间线平滑
- frame_index.csv
- 多分类标签
- 回放检测

Priority 3:
- 球场检测
- 运动员检测
- 人工校正界面
- 回合切分

Priority 4:
- 羽毛球轨迹检测
- 姿态估计
- 战术分析
- 论文实验统计
```

---

## 19. Copilot / Claude 开发 Prompt

可以把下面内容直接交给 Copilot 或 Claude：

```text
请帮我开发一个 Python 项目，用于从完整羽毛球大赛录像中自动提取“直播主视角比赛片段”。

输入是一个完整 mp4 视频，里面可能包含主视角比赛、观众席、球员喝水、教练指导、广告、回放、球员近景、裁判、转播包装等内容。目标是只保留高机位、完整球场、正在比赛或准备发球的主视角片段。

请按以下模块实现：

1. scripts/01_extract_shots.py
   使用 PySceneDetect 或 OpenCV 对视频进行镜头切分，输出 outputs/shots.json，格式包含 start、end、duration。

2. scripts/02_sample_keyframes.py
   对每个 shot 采样 3 到 5 张关键帧，保存到 outputs/keyframes/segment_xxxx/。

3. scripts/03_classify_segments.py
   对每个 shot 的关键帧进行分类。第一版先实现规则接口和可插拔分类器接口。
   标签包括：
   - MAIN_LIVE_VIEW
   - REPLAY
   - PLAYER_CLOSEUP
   - COACH_BREAK
   - AUDIENCE
   - AD
   - UMPIRE
   - OTHER

   请设计以下函数：
   - estimate_court_confidence(frames)
   - estimate_person_layout(frames)
   - detect_replay(frames)
   - detect_ad_or_graphics(frames)
   - classify_segment(frames)

   第一版可以先用占位规则，后续方便接入 YOLOv8、CLIP、ResNet 或自训练分类器。

4. scripts/04_smooth_timeline.py
   使用时间状态机平滑分类结果：
   - 删除小于 3 秒的 MAIN_LIVE_VIEW 片段
   - 合并间隔小于 2 秒的相邻 MAIN_LIVE_VIEW 片段
   - REPLAY、AD、COACH_BREAK 不允许被合并进主视角
   输出 outputs/clean_timeline.json。

5. scripts/05_export_main_view.py
   根据 clean_timeline.json 用 ffmpeg 导出主视角片段，并拼接成 outputs/main_match_only.mp4。
   同时输出 frame_index.csv，记录 clean video frame 到 original video timestamp 的映射。

6. scripts/run_pipeline.py
   一键运行完整流程。

代码要求：
- Python 3.10+
- 使用 pathlib、argparse、json、subprocess
- 保持模块化，方便后续替换分类器
- 每一步都有清晰日志
- 出错时给出明确报错
- 输出文件结构固定在 outputs/ 下
```

---

## 20. 总结

本数据预处理方案的关键点是：

```text
先清洗直播视频，再做轨迹和动作分析。
```

推荐主线为：

```text
Shot Detection
→ Keyframe Sampling
→ Segment Classification
→ Timeline Smoothing
→ Clean Video Export
→ Frame Index Mapping
```

这样可以有效过滤观众席、回放、暂停、教练、广告和球员近景等干扰内容，为后续运动员轨迹、羽毛球轨迹、回合分析和战术分析提供更干净、更稳定的数据基础。
