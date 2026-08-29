# Good-Badminton 领域上下文

本文件记录新研究管线中需要保持稳定的领域词汇。它描述“数据代表什么”，不描述具体模型或脚本怎样实现。

## 核心流程

```text
Source Match
  -> Main View
  -> Usable Rally
  -> Validated Calibration
  -> Observation
  -> Trajectory Sample
  -> Analysis Eligibility
  -> Tactical Result / Diagnostic Demo
```

## 领域词汇

### Source Match（源比赛）

一次管线运行所读取的原始比赛视频及其不可变身份信息。视频路径、内容指纹、解码信息和所用配置共同决定本次运行是否可复现。

### Main View（主视角）

能够同时观察完整或近完整球场、镜头几何关系在短时间内稳定、适合后续球场标定与跟踪的转播画面。回放、特写、观众席、记分牌和明显移动机位不属于 Main View。跨阶段传递时使用规范标签 `MAIN_VIEW`；历史标签只允许在兼容 Adapter 处归一化，不能继续向下游传播。

### Usable Rally（可用回合）

位于 Main View 内、具有足够连续活动证据、满足最小时长和质量门槛、并具有明确原视频帧区间的候选回合。连续主视角片段不等于 Usable Rally；每个候选必须记录 `accepted/rejected` 与原因，无法证明是回合时必须拒绝或等待人工复核。

### Frame Interval（帧区间）

跨阶段使用原视频帧编号表达的半开区间 `[start_frame, end_frame)`：包含 `start_frame`，不包含 `end_frame`，帧数恒为 `end_frame - start_frame`。Main View、Usable Rally、视频导出和时间换算必须遵守同一约定。

### Validated Calibration（已验证标定）

通过几何、重投影、球场类型和时序稳定性检查的图像到标准球场平面的映射。只有 Validated Calibration 才能支持米制运动指标。

### Observation（观测）

检测器或跟踪器在某一原视频帧上直接给出的球员或羽毛球位置，同时包含置信度、来源和有效性。缺失观测必须显式保留，不能伪装成检测结果。

### Trajectory Sample（轨迹样本）

由 Observation 或受限的短缺口插值得到的时序点。每个样本必须区分 `observed`、`interpolated` 和 `missing`；平滑不得把 `missing` 改写为有效样本。

### Analysis Eligibility（分析资格）

某一回合或某项指标是否满足最低数据条件。资格由上游产物的有效性、覆盖率、连续性和标定质量共同决定；不满足时输出 `not_eligible` 与拒绝原因，而不是输出猜测结果。`not_eligible` 不等于数值 0，相关计数字段必须留空或使用显式空值。

### Near-only Analysis（仅近端分析）

只承诺近端球员定位与可由该轨迹直接计算的指标。它不承诺远端身份、击球者归属、完整回合战术、羽毛球落点或胜负推断。界面和报告必须显式标记该限制。

### Dual-side Observation（双端观测）

同时输出近端与远端场地角色的球员 Observation。`near/far` 只表示球网两侧的场地角色，不等于稳定的运动员身份；在远端小目标检测、跨帧身份和冻结标注评估完成前，Dual-side Observation 保持 `experimental`，不能自动升级为完整双人战术结论。

### Reversal Candidate（方向反转候选）

由连续图像帧中的羽毛球运动方向变化产生的诊断信号。它不是已确认的击球、落点或胜负事件；在独立事件标注与模型验证完成前，只能以 `experimental` 资格保留图像坐标，不能进入正式战术计数。

### Stage Result（阶段结果）

一个阶段的结构化执行结论，至少包括状态、输入身份、输出产物、质量摘要和失败/拒绝原因。进程未抛异常不代表阶段成功。

### Artifact（产物）

阶段实际写出的 CSV、JSON、图片、视频或目录，以及它必须满足的结构契约。Artifact 的 `missing`、`empty`、`invalid` 和 `valid` 是不同状态；结构有效只代表可以被下游读取，不代表模型结果已经达到研究质量。

### Run Manifest（运行清单）

一次运行的可复现记录，包含 Source Match 指纹、完整配置、代码版本、模型身份、环境信息、各 Stage Result 和产物身份。恢复运行必须核对这些身份。

### Run Layout（运行布局）

一次运行所拥有的唯一目录边界及其 Artifact 路径映射。运行根目录由项目根、配置后的 runs 目录和单段 `run_id` 共同确定；所有阶段和批处理必须通过同一布局 Interface 取得路径，不能各自拼接目录或把 Artifact 写出该边界。

## 可信性不变量

1. 缺失数据保持缺失，除非被明确标记为受限短缺口插值。
2. 失败、拒绝、无数据和成功是不同状态。
3. 只有 Usable Rally 才能进入标定和跟踪。
4. 只有 Validated Calibration 才能产生米制指标。
5. Tactical Result 只能消费满足 Analysis Eligibility 的数据。
6. Diagnostic Demo 只能显示截至当前帧可获得的信息，不能把全回合结论提前泄漏到首帧。
7. Near-only Analysis 不能被表述为完整双人战术分析。
8. 空中羽毛球的地面 Homography 投影不能作为正式落点或击球位置。
9. 所有跨阶段 Frame Interval 必须使用半开区间，产物实际帧数必须与区间长度一致。
