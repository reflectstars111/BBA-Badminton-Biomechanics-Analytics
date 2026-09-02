# BBA 骨骼动作细节分析开发计划

状态：执行中
建立日期：2026-09-01
上游调研：[biomechanics_open_source_evaluation.md](biomechanics_open_source_evaluation.md)

## 1. 目标与完成定义

本计划把报告中的“骨骼动作细节分析 · 开发中”升级为一条具有证据链、质量门槛和可恢复产物的正式管线。最终目标包括：

- 击球候选与击球者归属；
- 击球动作分类；
- 准备、加速、接触、随挥和恢复阶段；
- 肩、肘、髋、膝等二维投影角度；
- 姿态稳定性和左右侧差异；
- 启动、移动、跨步、制动和回位等步法描述；
- 逐事件、逐回合和全场报告；
- WebUI、CSV/JSON 下载和最终视频的同步呈现。

“功能已实现”不等于“模型已经输出一个数字”。每项能力只有同时满足以下条件才可标记完成：

1. 输入身份、帧号、时间戳和场地角色可追溯；
2. 缺失、低置信和不连续数据被显式拒绝；
3. 输出具有 `eligibility / reject_reason / confidence / evidence_source`；
4. 单元测试、产物契约测试和冻结样本验证通过；
5. WebUI 对二维投影、实验能力和不具备资格的结果有清晰提示；
6. 报告不包含医学诊断、伪三维角度或未经验证的绝对技术评分。

## 2. 顶层数据流

```text
Validated Calibration
        |
RTMPose Observation -----> Kinematics Frame Metrics --------+
        |                                                    |
TrackNet Observation ----> Stroke Event Spotter ------------+---> Action Event
        |                                                    |       |
Court Position ----------> Footwork / Stability ------------+       +--> Swing Phases
                                                                    +--> Stroke Classifier
                                                                             |
                                                     Per-event / Rally / Match Summary
                                                                             |
                                                        WebUI / CSV / JSON / Video Overlay
```

动作分析设置为独立的 `biomechanics_analysis` Stage，位于 `tactical_analysis` 之后、`demo_rendering` 之前。它只消费已有 Artifact，不重新运行 RTMPose、TrackNet 或球场标定。

## 3. 领域语义

### 3.1 Kinematics Sample

某一球员在某一视频帧上由有效骨骼直接计算的二维运动学样本。每个角度必须记录：

- 所需关键点是否全部存在；
- 最低和平均关键点置信度；
- 原始二维投影角度；
- 是否经过平滑；
- 无效原因。

二维投影角度不是三维关节角。低视角、人体旋转、遮挡和远端小目标会改变其含义。

### 3.2 Stroke Candidate

由多个时序证据支持的潜在击球事件。它不是已确认击球。证据可包含：

- 羽毛球轨迹方向显著变化；
- 羽毛球与球员/手腕接近；
- 手腕归一化速度峰值；
- 肘角或肩角快速变化；
- 事件前后轨迹均有真实观测；
- 候选与相邻事件间隔合理。

### 3.3 Action Eligibility

动作事件是否允许进入某项分析。资格按能力分别判定：

- `kinematics_eligibility`；
- `phase_eligibility`；
- `classification_eligibility`；
- `footwork_eligibility`；
- `stability_eligibility`。

某项不具备资格不能阻止其他可靠指标输出，也不能以数值 0 替代。

### 3.4 Stroke Classification

分类器的输出包含原始类别、面向用户的合并类别、Top-2、置信度、模型身份和输入覆盖率。`near/far` 是独立字段，不编码进动作类别名称。

### 3.5 Stability Descriptor

稳定性第一版是描述性二维指标集合，不是医学或教练绝对评分。包括：

- 身体中心相对支撑中心的水平偏移；
- 支撑宽度相对人物框高度；
- 身体中心在事件窗口内的抖动；
- 左右膝角差异；
- 击球后恢复到稳定区间的时间。

## 4. Artifact 契约

运行目录新增：

```text
outputs/biomechanics/
  kinematics_frames.csv
  action_events.csv
  swing_phases.csv
  biomechanics_rally_summary.csv
  biomechanics_match_summary.json
  action_event_timeline.png          # 后续可视化批次
```

### 4.1 `kinematics_frames.csv`

主键：`video_stem + rally_id + frame_id + player_id`

最低字段：

- 身份：`video_path, video_stem, rally_id, frame_id, timestamp, player_id`；
- 质量：`pose_valid, kinematics_eligibility, reject_reason, keypoint_coverage_ratio, mean_keypoint_confidence`；
- 角度：左右 `elbow, shoulder, hip, knee`，以及 `trunk_lean_deg`；
- 稳定性基础量：`support_width_ratio, body_support_offset_ratio`；
- 来源：`pose_model, metric_version`。

### 4.2 `action_events.csv`

主键：`video_stem + rally_id + event_id`

最低字段：

- 候选：`candidate_frame, candidate_timestamp, player_id`；
- 证据：`evidence_source, evidence_count, shuttle_turn_score, shuttle_proximity_score, wrist_motion_score`；
- 资格：各能力的 eligibility 和 reject reason；
- 分类：`stroke_class, stroke_class_zh, top2_json, classification_confidence, model_id`；
- 运动学快照：接触候选附近的肘、肩、膝、躯干和稳定性指标；
- 窗口：`window_start_frame, window_end_frame`，仍遵循半开区间。

### 4.3 `swing_phases.csv`

每个动作事件最多输出五行：`preparation, acceleration, contact_window, follow_through, recovery`。字段包含半开帧区间、持续时间、资格、置信度和边界证据。

### 4.4 汇总产物

逐回合和全场只聚合具备相应资格的事件，并同时输出候选数、可分析数、拒绝原因分布、关键点覆盖率和实验性声明。

## 5. 配置契约

新增 `biomechanics_analysis` 配置段：

- 总开关与 `heuristic / bst` 分类后端；
- 关键点阈值、最少必需关键点和最少连续帧；
- 事件前后窗口长度；
- 手腕速度、球路转角和接近度阈值；
- 事件最小间隔；
- 二维角度平滑窗口；
- BST 权重、设备、类别映射和最低置信度；
- 是否启用实验性远端动作分析。

配置必须在长任务开始前完成类型与范围验证。权重缺失时，`backend=bst` 预检失败；`backend=heuristic` 只输出粗粒度动作形态，不冒充 BST 类别。

## 6. 分批执行计划

### BA-00：文档、领域契约与第三方边界

状态：**已完成**

工作项：

- [x] 开源方案、许可证和适配风险调研；
- [x] 定义顶层数据流与能力边界；
- [x] 定义 Artifact、配置和验收原则；
- [x] 确定 BST + Sports2D 思路 + 可选 RacketVision 路线。

验收：文档可以独立指导实现，未把专有或无许可证代码列为可复制来源。

### BA-01：二维运动学核心

状态：**已完成**

工作项：

- [x] 新建 `analysis/biomechanics` 包；
- [x] 实现带置信度门槛的二维三点角；
- [x] 计算左右肩、肘、髋、膝和躯干倾角；
- [x] 计算支撑宽度和身体中心相对支撑中心偏移；
- [x] 输出逐帧 CSV，缺失保持缺失；
- [x] 增加平移缩放、退化共线、低置信和缺关键点测试。

验收：合成标准姿态的角度误差不超过 `1e-6`；缺失点不产生角度；相同姿态平移/等比缩放后归一化指标不变。

### BA-02：动作分析 Stage 与可恢复运行

状态：**已完成**

工作项：

- [x] 新增 `StageName.BIOMECHANICS_ANALYSIS`；
- [x] 扩展 `RunLayout`、Manifest、Full Analysis Summary；
- [x] 新增配置 dataclass 和验证；
- [x] 把 BA-01 产物接入主管线；
- [x] 确认旧 Manifest 需要 `--force` 或新 run id，不静默混用阶段顺序。

验收：阶段失败不会写成成功；空但结构正确的事件 CSV 与缺失 CSV 是不同状态；恢复运行不会重复执行已完成阶段。

### BA-03：多证据击球候选

状态：**已完成**

工作项：

- [x] 从真实观测球轨迹计算局部方向变化，排除大缺口插值；
- [x] 计算球与身体/左右手腕的尺度归一化距离；
- [x] 计算左右手腕相对躯干尺度的速度峰值；
- [x] 多证据融合、非极大值抑制和最小事件间隔；
- [x] 输出击球者、候选分数和逐项证据；
- [x] near-only 时允许输出该球员候选，但不推断对侧击球事件。

验收：仅球路跳点、仅手腕噪声或插值跨越画外不能单独生成高置信事件；合成完整事件可稳定定位到允许误差窗口。

### BA-04：挥拍阶段分解

状态：**已完成**

工作项：

- [x] 自动选择运动证据更强的手腕侧候选，不将其冒充已知持拍手；
- [x] 从平滑归一化腕速划分准备、加速、接触窗、随挥和恢复；
- [x] 所有阶段使用半开帧区间且互不重叠；
- [x] 阶段缺失或窗口截断时输出拒绝原因；
- [x] 输出阶段置信度而非伪精确接触单帧。

验收：阶段有序、位于事件窗口内、接触窗覆盖候选帧；短视频边界不会生成负时长。

### BA-05：稳定性与步法描述

状态：**已完成**

工作项：

- [x] 基于有效球场轨迹检测移动开始、制动和回到事件前位置的候选；
- [x] 基于双脚次序变化与支撑宽度检测交叉步和宽支撑候选；
- [x] 输出事件前后位移、恢复时间、身体摆动和左右膝差；
- [x] 只输出 `candidate / descriptor`，没有标签数据前不输出优劣等级；
- [x] 远端步法分析设独立开关与拒绝原因。

验收：没有 Validated Calibration 时不输出米制步法；只有单脚有效时不输出双脚稳定性指标。

### BA-06：BST 分类插件

状态：**运行链路已验证，待冻结人工标注集验收精度**

工作项：

- [x] 建立第三方声明与模型身份清单；
- [x] 实现 BBA Artifact 到 BST 输入张量的适配器；
- [x] 固定窗口长度、零填充掩码和官方归一化规则；
- [x] 支持 CUDA 推理、Top-2 和未知阈值；
- [x] 将 Top/Bottom 标签拆成 BBA 的 `player_id + stroke_class`；
- [x] 权重不安装时自动降级为无分类，不影响二维运动学；
- [x] 使用可公开下载的官方 25% 训练权重完成 CUDA 严格加载和 BBA 真实产物推理冒烟验证；
- [ ] 取得全量训练 checkpoint，并完成官方样本一致性和 BBA 冻结样本精度验收。

验收：官方样本与 BBA 适配器推理结果一致；缺少权重不会伪装成已分类；模型、权重哈希和类别映射写入 Manifest。

### BA-07：WebUI、报告与视频

状态：**已完成**

工作项：

- [x] 将“动作分析路线图”改为真实动作分析标签页；
- [x] 展示候选/可分析数量、拒绝原因和能力边界；
- [x] 增加逐事件与逐阶段表格；
- [x] 提供动作 CSV/JSON 下载；
- [x] 在最终视频中显示当前挥拍阶段和已通过门槛的动作类别；
- [x] 更新进度条为十阶段并重新校准 ETA 权重。

验收：无动作事件时界面显示“未获得合格动作事件”而不是空白或开发中；表格字体与白色主题保持高对比度。

### BA-08：冻结数据评估与发布门槛

状态：**进行中**

工作项：

- [ ] 从现有俯视角与低视角素材建立人工复核集；
- [x] 建立事件真值 CSV 契约以及 detection / timing / player / classification 评估器；
- [ ] 使用 ShuttleSet/BFMD 建立按比赛隔离的事件/类别评估；
- [ ] 分 near/far、俯视/低视角报告覆盖率和准确率；
- [ ] 记录失败样本，不只保存成功演示；
- [ ] 更新中英文 README、限制说明和第三方致谢。

发布门槛：

- 击球候选必须同时报告 precision、recall、时间误差和击球者准确率；
- 分类必须报告 macro-F1、Top-1、Top-2 和 unknown/reject 比例；
- 角度必须通过人工抽帧与合成几何测试；
- 未达到门槛的能力保持 `experimental`，不能进入宣传性核心指标。

## 7. 测试策略

### 单元测试

- 角度、距离归一化、速度、局部峰值和阶段区间；
- 缺失、NaN、低置信、重复帧和非单调时间戳；
- 左右镜像和 near/far 角色不改变几何定义；
- 事件合并、最小间隔和边界窗口。

### 契约测试

- CSV 字段、空产物、Artifact 状态和 Manifest 阶段顺序；
- 配置类型、范围、后端和权重预检；
- Web Report 对旧 Run 的兼容降级。

### 集成测试

- 合成小视频/CSV 的完整 `biomechanics_analysis` Stage；
- 完整一键流程产生动作 Artifact；
- WebUI 读取、展示和下载；
- `--force` 与恢复运行。

### 真实样本验证

- 标准俯视角近端；
- 标准俯视角远端；
- 低视角近端；
- 低视角远端；
- 遮挡、出画、回放和球飞出画面等负例。

先从管线产物生成三帧复核图和待审核 CSV：

```cmd
bdp biomechanics review outputs\biomechanics\action_events.csv evaluation\review --player-tracks annotations\player_tracks_smoothed.csv --max-events 200
```

输出目录包括：

- `images/*.jpg`：候选前帧、候选帧、候选后帧三联图，叠加球员框与骨骼；
- `biomechanics_ground_truth_draft.csv`：初始帧号、球员、模型预测和图片相对路径；
- `review_manifest.json`：导出数量、缺图数量与标注范围。

审核时把 `review_status` 从 `pending` 改成 `accepted` 或 `rejected`；可修改 `reference_frame`、`player_id` 和 `stroke_class`。该草稿的 `annotation_scope=prediction_seeded`，只能测候选 precision、时序误差、击球者和分类指标，**不能据此宣称 recall**。要测 recall，必须对冻结回合做独立、完整的人工事件标注，并把 `annotation_scope` 改为 `exhaustive`。

完成标注后执行：

```cmd
bdp biomechanics evaluate outputs\biomechanics\action_events.csv evaluation\biomechanics_ground_truth.csv evaluation\biomechanics_metrics.json --tolerance-frames 3
```

只有已经审核的 `accepted/rejected` 行进入评估；`pending` 行完全排除，不会被误计为误检。未确认或有争议的行不应伪装成真值。

## 8. 风险与回退

- 每个新增能力都有独立开关和独立 Artifact，可关闭而不影响清洗、标定、跟踪和现有报告；
- BST 作为可选插件，依赖或权重问题不会破坏基础管线；
- 不修改现有姿态 CSV 的语义，新指标写入新目录；
- 旧 Run 缺少动作产物时 WebUI 继续显示兼容说明；
- 如果事件检测未达门槛，保留二维运动学并将分类/阶段标为 `not_eligible`。

## 9. 执行记录

### 2026-09-01：BA-00 完成

- 完成开源项目与许可证调研；
- 确定 BST、Sports2D 思路和 RacketVision 的分层组合；
- 建立动作分析领域语义、Artifact 契约、配置边界、测试策略与 BA-01 至 BA-08 执行顺序。

### 2026-09-01：BA-01 与 BA-02 完成

- 新增基于现有 COCO-17 RTMPose 结果的二维关节角、躯干倾角和支撑稳定性基础量；
- 低置信或缺失关键点保持空值，并记录逐帧资格与拒绝原因；
- 新增 `biomechanics_analysis` 正式阶段、类型化配置和 `outputs/biomechanics` 产物目录；
- 完整分析摘要新增二维运动学覆盖统计，WebUI 进度更新为十阶段；
- 旧九阶段 Manifest 如果已经完成视频渲染，会要求新 run id 或 `--force`，避免阶段倒序混用；
- 47 项相关测试通过。

### 2026-09-01：BA-03 至 BA-05 完成

- 建立球路转向、球员接近度和手腕运动三证据候选器，并加入插值排除与时序非极大值抑制；
- 建立准备、加速、接触窗口、随挥、恢复的半开区间分解及截断拒绝规则；
- 增加支撑宽度、身体偏移 RMS、躯干摆动、膝角差、事件位移、速度、制动和回位候选；
- 新增逐回合描述汇总与全场能力声明，保持二维描述性边界；
- 50 项相关测试通过。

### 2026-09-01：BA-06 适配层与 BA-07 完成

- 按 BST 官方契约建立双球员 Joint/Joint+Bone 输入、归一化、定长填充、25/35 类标签和 Top/Bottom 角色拆分；
- 增加 CUDA、Top-2、置信度门槛、模型权重哈希和缺依赖降级；
- WebUI 新增动作事件与挥拍阶段表，下载项覆盖动作 CSV/JSON；
- 最终视频支持显示当前动作候选、通过门槛的动作类别和挥拍阶段；
- 进度条扩展为十阶段。

### 2026-09-01：BA-08 冒烟验证

- 全量单元与契约测试通过 167 项；
- 俯视样本：978 条运动学记录、22 个击球候选；样本有效时长约 16.4 秒；
- 低视角样本：22,492 条运动学记录中的 20,970 条具备资格，46 个回合、约 458.6 秒内得到 785 个候选；
- 根据低视角首轮 1,537 个过密候选，将事件门槛收紧为多帧球路转向 + 最低球员接近度，手腕运动只作为加分证据；
- 上述统计仅是无人工真值的冒烟检查，precision/recall、击球者准确率和分类指标仍待冻结标注集。

### 2026-09-01：BA-08 人工复核链完善

- 根据官方结果表确认 25 类、30 帧、2D、`JnB_bone` 的推荐权重档位，默认建议 `BST_AP serial_3`；
- checkpoint 严格加载失败时返回包含模型、模态、序列长度和类别数的 `bst_checkpoint_incompatible`；
- 新增动作候选三帧复核图、骨骼叠加、跨回合均衡抽样和待审核真值草稿导出；
- 区分 `prediction_seeded` 与 `exhaustive` 标注，修复 pending 行被误计为误检以及预测驱动标注虚构 recall 的风险。
- WebUI 动作细节页新增一键生成并下载复核包；全量回归测试通过 171 项。

### 2026-09-02：正式运行套件与 BST CUDA 链路验证

- 不创建新的 Conda 环境，明确已有 `good-badminton` 为唯一正式运行环境；
- 建立完整依赖锁定清单、统一安装/修复脚本和 `production --strict` 环境自检，覆盖 WebUI、CUDA PyTorch、ONNX GPU/RTMPose、TrackNet/Ultralytics、BST、统计评估与报告依赖；
- 处理 RTMLib 只声明 CPU `onnxruntime` 的元数据冲突：正式套件保留 `onnxruntime-gpu`，RTMLib 以 `--no-deps` 安装，并实际验证 `CUDAExecutionProvider`；
- 从上游公开文件夹取得 `bst_AP_JnB_bone_train_partial_0p25_merged_2.pt`，在 RTX 5070 Laptop GPU 上严格加载，模型 ID 为 `BST/BST_AP/JnB_bone/seq30/25/015f7010526b`；
- 俯视样本 22 个动作事件中 14 个满足 BST 输入门槛，低视角样本 785 个事件中 32 个满足门槛；该结果验证运行链路，不替代人工标注精度验收；
- 上游结果表中的全量训练 `BST_AP serial_3` 指标更高，但本次公开文件夹枚举未发现对应 checkpoint，后续取得权重后再进入最终模型验收。

### 2026-09-02：冻结评估集第一批复核包

- 将复核抽样从单纯按回合轮询升级为“球员角色 × 分类资格/拒绝原因 × 回合”的均衡抽样，避免低视角的单一失败类型淹没其他诊断样本；
- `review_manifest.json` 升级至 `bba_biomechanics_review_v2`，新增总体分类资格率、各拒绝原因、选中样本构成以及 near/far 分项覆盖率；
- 俯视样本导出全部 22 个候选，22 张三帧骨骼复核图全部成功；分类资格率为 63.64%，far 为 54.55%，near 为 72.73%；
- 低视角从 785 个候选中均衡抽取 200 个，200 张复核图全部成功；原始分类资格率为 4.08%，主要阻塞为对侧骨骼覆盖不足；
- 两套草稿保持 `annotation_scope=prediction_seeded` 和 `review_status=pending`，尚未人工确认，因此只能在审核后评估候选 precision、时序、球员归属与分类，不能据此报告 recall。
