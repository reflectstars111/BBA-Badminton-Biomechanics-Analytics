# 新研究管线全流程修改意见与实施路线

状态：执行中  
最后更新：2026-08-29  
适用范围：`badmintondataprocess/` 新研究管线  
旧管线策略：冻结，仅保留迁移期兼容入口，不再新增算法能力

## 1. 文档目的

当前系统已经能够从视频运行到演示视频，但“流程跑完”与“结果可信”尚未等价。实际效果差并非单个模型参数不足，而是多个阶段同时存在语义过宽、失败状态丢失、产物契约松散和后处理制造连续性的情况。

本计划的目标不是一次性重写，而是按可验证的小批次逐步完成三件事：

1. 先停止生成或展示虚假的有效结果。
2. 再建立从 Main View 到 Analysis Eligibility 的质量门控。
3. 最后才升级标定、球员跟踪、羽毛球跟踪和战术算法。

任何阶段都不能靠演示观感代替定量验收。完整演示只是 Diagnostic Demo，不是模型正确性的证据。

## 2. 总体结论

研究方向合理：从转播视频中提取主视角回合，完成球场标定、球员/羽毛球轨迹和二维战术统计，适合作为研究原型。

当前实现不适合直接声称“完整效果”：

- 原始转播可以绕过 Main View 直接进入 rally 切分。
- rally 判断实际更接近“连续球场画面”，不是可靠的回合识别。
- 自动标定把绿色区域轮廓极值当作球场角点，且忽略单双打宽度语义。
- 球员角色由位置启发式决定，不能保证人物身份连续。
- 羽毛球 baseline 主要跟踪亮且运动的小目标，TrackNet 路径仍有输入和内存问题。
- 平滑会把长缺失延续成有效轨迹。
- 战术阶段把启发式转向点解释为击球/落点，并可能覆盖击球者归属。
- 演示会连接缺失轨迹、钳制越界坐标并提前展示全回合统计。
- 阶段成功主要由“是否抛异常”决定，无法表达空结果、拒绝和质量不合格。

因此，近期产品定位必须明确为：

> 受质量门控的研究数据处理管线；Near-only Analysis 是当前可优先验证的范围，Diagnostic Demo 仅用于排错与展示已验证产物。

## 3. 目标架构

```text
统一入口
  |
  v
Run Specification Module
  |  固定输入、配置、模型和产物布局
  v
Main View Module
  |  Main View segments + frame mapping
  v
Usable Rally Module
  |  accepted/rejected rallies + reasons
  v
Validated Calibration Module
  |  homography + geometry/temporal quality
  v
Tracking Modules
  |  observed / interpolated / missing samples
  v
Trajectory Module
  |  preserve validity + short-gap smoothing
  v
Analysis Eligibility Module
  |  metric-specific eligibility + reasons
  +----------------------+
  |                      |
  v                      v
Tactical Result       Diagnostic Demo
```

横切整个流程的是 Run State Module：每个阶段返回 Stage Result，Run Manifest 记录输入身份、配置、代码、模型、环境、质量摘要和产物指纹。

### 3.1 Module 深化原则

这里的 Module 同时包含 Interface 和 Implementation。改造重点不是增加更多小文件，而是让调用者只需理解少量稳定 Interface，把复杂检查集中在 Implementation 内，提高 Depth。

- `Run Specification Module`：调用者只描述一次运行，不再知道每个输出路径怎样拼接。
- `Usable Rally Module`：调用者获得接受/拒绝结果，不再组合主视角、回合启发式和人工复核文件。
- `Validated Calibration Module`：调用者只能获得已验证映射或结构化拒绝，不能误用任意矩阵。
- `Trajectory Module`：调用者获得保留数据血缘的轨迹，不再自行猜测空字符串、置信度和平滑列。
- `Analysis Eligibility Module`：所有统计共用资格判定，避免每个脚本各自默许缺失数据。

这些 Module 的 Leverage 是一处规则服务所有阶段和测试；Locality 是有效性、失败语义和质量标准不再散落在脚本调用处。

### 3.2 Seam 与 Adapter 原则

只在确实存在两种实现时保留 Seam：

- 球员检测：YOLO Adapter 与测试用确定性 Adapter。
- 羽毛球跟踪：baseline Adapter 与 TrackNet Adapter。
- 标定输入：人工关键点 Adapter 与自动候选 Adapter。

只有一个实现的地方不提前抽象。一个 Adapter 只是“可能变化”的假设，两个 Adapter 才形成真实 Seam。现有只做参数转发的 legacy 包装属于浅 Module；通过删除测试后，应逐步把核心 Implementation 迁入包内，让 `scripts/*.py` 只保留兼容命令入口。

## 4. 必须保持的领域语义

稳定词汇见仓库根目录 `CONTEXT.md`。实施中尤其要遵守以下不变量：

1. `missing` 不能因滚动中位数或 EMA 变成 `valid`。
2. `interpolated` 仅适用于实际帧差不超过上限、且两端均有有效 Observation 的内部缺口。
3. 进程返回非零、必需产物缺失、产物为空或质量门槛失败，都不能记录为 `success`。
4. Main View 不等于 Usable Rally。
5. 只有 Validated Calibration 才能输出米、米/秒和平方米。
6. 空数据、未执行、被拒绝和运行失败必须分别表达。
7. Near-only Analysis 只输出近端球员可由轨迹直接支持的指标。

## 5. 分阶段实施计划

### P0-A：停止制造有效数据（第一批，已完成）

涉及文件：

- `scripts/trajectory_smoothing.py`
- `src/badminton_data_process/smoothing/trajectory.py`
- 新增轨迹平滑回归测试

问题：当前滚动中位数会从邻近点填充空值，EMA 会无限延续上一个值，`is_smoothed_valid` 又只检查最终坐标非空。因此长时间漏检会被报告为连续有效轨迹。分组数据也没有按 `frame_id` 排序，插值上限按 CSV 行数而不是真实帧差计算。

修改：

1. 每个轨迹组按真实 `frame_id` 排序。
2. 帧号不连续时切断平滑状态，禁止跨段中位数和 EMA。
3. 中位数只平滑已有或已批准插值的值，不能填补 `None`。
4. EMA 遇到缺失必须输出缺失并重置状态。
5. `is_smoothed_valid = source_valid OR short_gap_filled`，不再由输出是否非空反推。
6. 摘要增加可用于审计的覆盖率时，分母和状态定义必须固定。

验收：

- 100 帧缺失不会输出 100 帧静止坐标。
- 允许上限内的内部缺口能够插值，并标记 `is_gap_filled=1`。
- 超过上限的缺口保持空白且 `is_smoothed_valid=0`。
- CSV 行乱序不会改变结果。
- 球员数据中缺少整行的帧差会切断平滑状态。

### P0-B：阶段结果与失败传播（已完成）

涉及文件：

- `src/badminton_data_process/core/run.py`
- `src/badminton_data_process/core/schemas.py`
- `src/badminton_data_process/pipeline/run.py`
- 所有阶段入口及管线集成测试

问题：`stage_report` 只根据异常判断成功，而多个阶段用整数返回码表达失败。调用者还可能在阶段被记为成功之后才检查返回值；部分返回值完全被忽略。

目标 Module：`Stage Execution Module`。

建议 Interface 行为：

- 阶段 Implementation 返回结构化 Stage Result。
- 状态至少包含 `success`、`rejected`、`empty`、`failed`、`skipped`。
- Stage Result 包含产物存在性、记录数、关键质量指标和原因码。
- Run State Module 只依据 Stage Result 写清单，不猜测脚本返回值。

实施步骤：

1. 先为现有整数返回值建立兼容转换，非零一律不能写成功。
2. 给每个阶段增加必需产物校验与最小行数检查。
3. 将标定阶段的“部分拒绝”表达为成功阶段中的 rejected items，而不是先成功后抛错。
4. 为 `stop_after`、`skip_visualize`、`skip_demo` 和 resume 建立组合测试。

验收：任一必需产物缺失、为空或返回非零时，manifest 中不可能出现该阶段 `success`。

### P0-C：修正战术与演示的越权推断（C1 已完成，C2 待实施）

涉及文件：

- `scripts/tactical_analysis.py`
- `src/badminton_data_process/visualization/demo.py`
- `scripts/visualize_tracking.py`

问题：战术阶段把稀疏轨迹相邻点当连续时序、把最后一个转向点当落点、用地面 Homography 投影空中羽毛球，并用球场半区覆盖先前的最近球员判断。演示连接长缺失、钳制错误投影、从首帧显示全回合统计，且混合分辨率时可能使用错误坐标。

修改：

1. Near-only 模式先禁用“击球者、落点、完整战术”结论，输出 `not_eligible` 原因。
2. 所有速度和距离计算检查真实帧差与时间差；超阈值断开序列。
3. 取消“最后一个方向反转等于落点”的规则。
4. 羽毛球地面投影只作为诊断可视化，不进入正式战术指标。
5. 演示轨迹遇到 invalid 或长缺失立即断线。
6. 越界坐标显示为异常/拒绝，不钳制到球场边缘。
7. 统计只显示截至当前帧的累计值，或明确标记为“全回合摘要”并在回合结束后出现。
8. 修复 1–2 个 rally 时严格 `zip` 导致的可视化崩溃。

验收：用故意缺失、越界和稀疏的 fixture 验证输出不会产生击球/落点结论，也不会在画面上连接假轨迹。

### P0-D：让 Main View 真正成为入口门槛

涉及文件：

- `src/badminton_data_process/main_view/`
- `src/badminton_data_process/preprocess/timeline.py`
- `scripts/rally_segmentation.py`
- `src/badminton_data_process/pipeline/run.py`

问题：当前一键管线从原视频直接开始 rally segmentation；独立 Main View 分析、timeline、质量复核没有接入正式执行链。标签名称也存在 `MAIN_LIVE_VIEW` 与 `MAIN_BIRDSEYE_LIVE` 不一致。

目标 Module：`Usable Rally Module`，把 Main View、回合切分、质量门槛和拒绝原因隐藏在一个较深 Interface 后。

修改：

1. 管线首阶段改为 Main View 分析，输出原视频帧映射。
2. rally segmentation 只能消费已接受的 Main View 区间。
3. 统一标签枚举，禁止自由字符串跨阶段传播。
4. 明确使用或删除 `max_gap_seconds`、前后上下文和当前只记录不决策的运动/线条参数。
5. 零个 Usable Rally 返回 `empty/rejected`，不能作为成功继续。
6. 修正片段末帧包含关系，统一 `[start_frame, end_frame)` 或闭区间约定。

验收：包含回放、特写和主视角的合成时间线只输出主视角内的回合；全程非主视角时管线在此停止并给出原因。

### P1-A：Run Specification、产物契约与可恢复运行

涉及文件：

- `core/config_schema.py`
- `core/io.py`
- `core/run.py`
- `pipeline/run.py`
- `scripts/batch_pipeline.py`

目标 Module：`Run Specification Module` 与 `Artifact Module`。

修改：

1. 配置解析拒绝未知键、负数、越界比例和不合法角色组合。
2. 删除无 Implementation 消费的伪配置，或在实现完成前标记为 unsupported 并拒绝启用。
3. 统一 `RunLayout`，输出根目录、annotations、outputs 和批处理 `runs_dir` 不再由调用处硬编码。
4. CSV 读取区分 missing file、empty artifact 和 valid empty set。
5. 每种产物定义 schema version、必需字段、行唯一键和状态字段。
6. resume 核对输入内容指纹、解析后配置、模型指纹、代码版本和上游产物指纹。
7. 修正跳过可视化后恢复运行的阶段排序问题。

验收：输入视频、配置或模型任一变化时不能静默复用旧产物；未知配置立即失败并指出字段路径。

### P1-B：Validated Calibration

涉及文件：

- `scripts/court_calibration.py`
- `src/badminton_data_process/calibration/`
- 配置 schema 与标定测试集

目标 Module：`Validated Calibration Module`。

修改：

1. 自动候选从“绿色最大轮廓极值”升级为球场线交点/结构模板候选。
2. 人工关键点与自动候选作为两个真实 Adapter，共用验证 Implementation。
3. `court_type` 必须决定标准球场宽度；单打 5.18m，双打 6.10m。
4. 检查矩阵条件数、角点顺序、凸性、重投影误差、线支持、投影越界率。
5. 从多个代表帧估计稳定性；检测到机位漂移时拒绝整个区间或切段重标定。
6. 标定产物记录使用的帧、关键点来源、质量指标和版本。
7. 所有图片/视频写入必须检查返回状态。

验收：人工标注基准集上报告角点误差、重投影误差和通过率；单打/双打米制距离分别正确。

### P1-C：球员 Observation 与身份

涉及文件：

- `tracking/player/tracking.py`
- 球员检测与跟踪 fixture / 标注集

近期范围仍以 Near-only Analysis 为主：

1. 输出每帧显式 `observed/missing`，不省略漏检帧。
2. 坐标阈值按画面尺寸、球场尺寸或时间归一化，取消不可迁移的固定像素/帧阈值。
3. 位置优先使用脚踝中点；姿态不可用时回退到 bbox 底边中点并记录来源。
4. 近端候选必须位于 Validated Calibration 的近半场并满足时序连续性。
5. 角色是 `near/far`，人物身份是 `athlete_id`；两者不得混用。
6. 双端模式在具备 ReID/多目标关联和标注评估前保持 experimental。

验收指标：近端检测 precision/recall、轨迹覆盖率、位置误差、ID switch；不能只看输出行数。

### P1-D：羽毛球 Observation

涉及文件：

- `tracking/shuttle/tracking.py`
- `tracking/shuttle/tracknet.py`
- 模型权重与评估脚本

修改：

1. 固化 TrackNet 的真实输入契约：帧数、通道、归一化、背景模式和输出坐标定义一致。
2. search mask 必须进入候选限制或明确删除。
3. 视频流式解码，只保留模型所需的滑动窗口，禁止整段帧驻留内存。
4. 权重采用安全加载方式，并记录校验和、模型结构和推理设备。
5. baseline 不再把最低置信度强制抬到 0.2；置信度必须可校准。
6. 两个 Adapter 在同一人工标注测试集上比较，不用“覆盖率越高越好”代替精度。

验收指标：可见性 precision/recall、像素距离阈值内准确率、轨迹连续性、跳点率、显存/内存峰值和处理速度。

### P2-A：Analysis Eligibility 与正式战术指标

在 P0/P1 的 Observation、Trajectory Sample 与 Validated Calibration 稳定后实施。

1. 每项指标声明依赖，例如跑动距离需要已验证标定、连续近端轨迹和最大缺失阈值。
2. 距离只累计连续样本；速度按真实时间差计算。
3. 覆盖面积报告采样数和置信区间/稳定性，不对少量点给出强结论。
4. 击球与落点建立独立标注任务和模型，不从“方向反转”直接命名为真实事件。
5. Near-only 只发布站位、移动距离、区域占比等可直接支持的指标。
6. 每个结果携带 eligibility、coverage、quality 和 reason codes。

验收：在冻结测试集上，指标与人工轨迹计算结果一致；不具资格的回合只给拒绝原因。

### P2-B：Diagnostic Demo 与研究报告

1. 演示层只消费结构化产物，不自行修补或重新解释数据。
2. 用颜色/线型区分 observed、interpolated、missing 和 rejected。
3. 明确显示当前分析模式（near-only / experimental two-player）。
4. 显示每项结果的资格和覆盖率；失败时显示原因，而不是空白或假数据。
5. 研究报告从冻结评估结果生成，不能从演示视频肉眼判断模型优劣。

## 6. 测试与评估体系

### 6.1 测试金字塔

- 纯函数测试：几何、帧区间、短缺口插值、状态转换。
- Module Interface 测试：每个深 Module 的输入、输出、不变量和错误模式。
- Adapter 契约测试：YOLO/测试 Adapter、baseline/TrackNet Adapter 在相同 fixture 上满足同一产物契约。
- 管线集成测试：合成短视频覆盖成功、空结果、部分拒绝、阶段失败、skip 和 resume。
- 冻结数据集评估：人工标注的 Main View、rally、球场角点、近端球员和羽毛球真值。

Interface 是主要测试面。若测试必须越过 Interface 才能验证关键语义，说明 Module 仍然过浅。

### 6.2 最低冻结数据集

建议先建立小而高质量的集合，而不是继续调单个演示视频：

- 3 场不同转播/分辨率比赛。
- 每场至少 5 个主视角回合、2 个回放/特写负样本。
- 每个回合标注主视角区间与 rally 起止。
- 每个回合至少 3 帧球场关键点。
- Near-only 第一阶段每 5 帧标注近端球员脚点。
- 羽毛球选择 500–1000 帧密集标注可见性与中心点。

当前 `evaluation/test_set.csv` 使用本机绝对路径且没有真值，必须替换为仓库相对 ID 与单独标注产物。

### 6.3 发布门槛

在满足下列条件前，不把“一键完整效果”作为正式能力：

- 全流程集成测试覆盖成功、拒绝、失败和恢复。
- 所有正式统计都由 Analysis Eligibility 门控。
- 标定和近端球员在冻结测试集上达到预先确定的误差阈值。
- 羽毛球模型至少优于 baseline，且比较包含精度而非只有覆盖率。
- manifest 可重现实验输入、代码、配置和模型。

## 7. 运行与迁移策略

### 7.1 入口

1. 根目录入口和 README 改为默认引导 `bdp`。
2. 旧 `main.py` 显示弃用提示和迁移命令；冻结一段明确版本周期后删除。
3. 新旧管线产物目录不得混用。

### 7.2 兼容脚本

当前 `src/...` 多处通过动态加载 `scripts/*.py` 提供浅 Adapter。迁移采用逐阶段方式：

1. 先把测试钉在新包的 Interface 上。
2. 将核心 Implementation 迁入包内。
3. `scripts/*.py` 改为调用包内 Interface 的薄命令入口。
4. 应用删除测试：删除兼容脚本时，复杂度不应散回多个调用者。

### 7.3 小批次规则

每批只处理一个可验证不变量：

- 改动前记录错误 fixture。
- 改动后跑针对性测试和全量测试。
- 更新本文件的状态与结果。
- 不在同一批同时更换模型、配置含义和产物 schema。
- 真实视频回归使用固定 run ID 的新目录，不能覆盖历史结果。

## 8. 建议提交/问题拆分

每项应可被独立领取和验收：

1. `P0-A1` 平滑保留缺失语义与真实帧间隔。
2. `P0-B1` Stage Result 兼容层与失败传播。
3. `P0-B2` 必需产物验证和 manifest 状态扩展。
4. `P0-C1` Near-only 禁用无依据的击球/落点推断。
5. `P0-C2` Diagnostic Demo 不连接/钳制无效轨迹。
6. `P0-D1` Main View 接入正式管线并统一标签。
7. `P0-D2` Usable Rally 空结果和时间区间语义。
8. `P1-A1` 严格配置与 RunLayout。
9. `P1-A2` 产物 schema/version 与安全 resume。
10. `P1-B1` 标定验证器和单双打坐标。
11. `P1-B2` 多帧标定稳定性与人工 Adapter。
12. `P1-C1` 近端逐帧 Observation 与脚点来源。
13. `P1-D1` TrackNet 输入修复与流式内存。
14. `P1-D2` 羽毛球冻结评估集和模型比较。
15. `P2-A1` Analysis Eligibility 与近端正式指标。
16. `P2-B1` 可信 Diagnostic Demo 与研究报告。

如需发布到 GitHub Issues，应按上述纵向切片进一步补充 fixture、命令和验收值，避免把整个架构改造放进一个大 issue。

## 9. 风险与回退

- 修复缺失语义后，覆盖率会明显下降。这是暴露真实质量，不是回归；演示会更“断”，但数据更可信。
- 接入 Main View 和质量门槛后，可能没有任何 Usable Rally。此时正确行为是停止并报告原因。
- 严格配置会让过去被忽略的字段报错，需要提供迁移提示。
- 标定验证加强后，自动通过率可能下降，应优先增加人工关键点 Adapter，而不是放宽质量阈值。
- 双端人物跟踪与羽毛球事件识别可能需要新标注和模型训练，不能用后处理替代。

每批改动都通过新 run 目录验证，历史 run 保持只读；若新规则错误，只回退该批代码和 schema 版本，不修改历史产物。

## 10. 当前执行记录

### 2026-08-28：架构审计完成

- 已从入口、编排、配置、Main View/rally、标定、球员、羽毛球、平滑、战术、演示、评估与测试核对实现。
- 已确认首要问题是“未知/缺失/失败被包装成连续成功”，不是简单的 YOLO 参数问题。
- 现有测试 22 项通过，但未覆盖全管线、resume、质量门槛和真实模型精度。

### 2026-08-28：P0-A1 开始

- 目标：平滑不得制造有效轨迹。
- 状态：已完成。
- 行为变化：轨迹组按 `frame_id` 排序；插值按真实帧差限制；不连续帧切断平滑状态；滚动中位数不填空值；EMA 遇到缺失输出缺失并重置；`is_smoothed_valid` 只来自原始有效观测或获准的短缺口插值。
- 新增回归：短缺口插值、长缺口保持缺失、省略整帧时断开状态、乱序输入排序。
- 验证结果：针对性测试 4 项通过；项目全量测试 25 项通过；`compileall` 通过。
- 剩余限制：球员跟踪器仍会省略漏检帧，本批只保证不跨这些帧平滑，不负责合成显式 `missing` 行；历史 run 产物不会自动更新，需要从平滑阶段重新生成。

### 2026-08-28：P0-B1 完成

- 目标：旧阶段返回非零时，manifest 不得记录成功，后续阶段不得继续。
- 新增 `StageResult`、`StageExecution` 和 `StageExecutionError`，形成整数返回码到结构化结果的兼容 Interface。
- rally、标定、球员、羽毛球、图表和战术阶段均在 `stage_report` 内消费返回码；`None/0` 为兼容成功，非零或非预期类型形成失败并立即抛出。
- `PipelineStageReport` 和 manifest 新增向后兼容的 `exit_code`；旧 manifest 没有该字段时仍可读取。
- `stage_report` 现在也会把 `SystemExit` 记录为失败，避免命令式 Adapter 提前退出却留下成功状态。
- 标定 summary 的空结果、返回码/summary 矛盾、全部失败和部分失败处理都移入 stage 内；部分失败时保留既有策略，只剔除明确失败的 rally，并在成功报告 message 中记录接受/拒绝数量。
- 新增回归覆盖：成功/失败返回码、`SystemExit`、失败后 resume、rally/player/shuttle/visualization/tactics 失败传播、标定部分成功与全部失败。
- 验证结果：本批针对性测试 11 项通过；项目全量测试 36 项通过；`compileall` 与 `git diff --check` 通过。
- 剩余限制：零 rally、空 tracking CSV、缺失图表/视频等“返回 0 但产物不可用”的情况尚未统一拦截，归入下一批 P0-B2 Artifact 验证。

### 2026-08-29：P0-B2 完成

- 目标：返回 0 但必需 Artifact 缺失、为空或结构错误时，stage 不得记录成功。
- 新增 Artifact Module，统一输出 `missing`、`empty`、`invalid`、`valid` 四种检查结论；Interface 支持普通文件、CSV、目录、文件集合和可解码视频。
- `StageStatus` 扩展为 `success/rejected/empty/failed/skipped`；当前零 rally 和必需 CSV 零行会记录为 `empty`，缺失或结构错误记录为 `failed`。
- Stage Result 与 manifest 现在保存每项 Artifact 的类型、状态、路径、消息和细节；CSV 包含行数与字段，目录包含匹配文件数，视频包含首帧尺寸、FPS 与声明帧数。
- rally metadata 必须至少一行且每个 rally 视频至少可解码一帧；标定 summary 必须至少一行且成功项必须具有非空 JSON 文件。
- 球员/羽毛球原始轨迹与 summary 必须存在、字段完整且至少一行；全不可见羽毛球仍允许，因为轨迹 CSV 的逐帧结构本身可以有效。
- 平滑后的两类轨迹和两份 summary 均为必需 Artifact；图表目录至少产生一张 PNG。
- 战术 summary 必须至少一行；战术 events 允许零行但必须具有正确表头，从而区分“没有事件”和“没有产物”。
- Diagnostic Demo 必须存在、非空且至少能解码一帧。
- 新增回归覆盖 Artifact 四态、CSV 字段与合法空集合、不可解码视频、manifest 序列化、零 rally 和返回成功但缺少球员轨迹。
- 验证结果：本批相关测试 20 项通过；项目全量测试 45 项通过；`compileall` 与 `git diff --check` 通过。
- 剩余限制：本批验证结构与最低可读性，不判断轨迹精度、覆盖率、标定几何质量或战术可信度；同一 run 上 `--force` 时旧 Artifact 的清理/指纹核对归入 P1-A2。

### 2026-08-29：P0-C1 完成

- 目标：Near-only Analysis 不得输出无依据的击球者、击球次数、落点或完整战术事件。
- 新增事件 Analysis Eligibility Interface，统一返回解析后的 `analysis_mode`、`event_eligibility` 和 `event_reject_reason`。
- 管线根据球员跟踪角色传播模式：仅配置 `near` 时传入 `near_only`；其他角色组合明确标记为 `experimental_two_player`。
- Near-only 在调用事件 Implementation 前即停止事件分析，不再执行羽毛球方向反转、空中地面投影、击球者或最后反转落点推断。
- `tactics_summary.csv` 继续保留近端站位、距离、速度、覆盖面积和区域占比，但事件资格写为 `not_eligible`，`hit_count/landing_count` 留空而不是伪装为 0。
- `tactics_events.csv` 在 Near-only 下保留合法表头但不写猜测事件；Artifact Module 将其识别为合法空集合。
- Diagnostic Demo 对不具资格的事件显示 `events N/A`，不再把空计数渲染成 `hits 0`。
- 双端模式仍保留旧事件启发式以便对照，但明确标记为 `experimental`；它不是正式研究结论。
- 新增回归覆盖 Near-only/missing-role/双端资格、模式传播、事件 Implementation 不被调用、空事件产物、空计数和 demo 文案。
- 验证结果：本批针对性测试 22 项通过；项目全量测试 49 项通过；`compileall` 与 `git diff --check` 通过。
- 剩余限制：移动距离仍可能跨稀疏帧累计，Diagnostic Demo 仍可能连接或钳制无效坐标；这些归入下一批 P0-C2。

### 2026-08-29：P0-C2 完成

- 目标：Diagnostic Demo 与战术摘要不得通过连接缺失、钳制越界或提前显示未来信息来制造“完整且稳定”的假象。
- 战术指标 Module 只在 `frame_id` 连续、时间严格递增且瞬时速度不超过 12 m/s 的相邻样本间累计距离；跨帧缺口、时间异常和不可信跳变均切断序列，并通过 `distance_steps`、`discontinuity_count`、`movement_duration_seconds` 暴露计算依据。不存在合法运动步时写 `movement_eligibility=not_eligible` 并留空距离/速度，不再把未知伪装成 0。
- 近端/远端球员的正式米制指标只消费场内且位于对应半场的坐标；被拒绝的位置数量写入 `rejected_position_rows`，不再进入距离、速度、覆盖面积和区域占比。
- 羽毛球方向反转检测只检查连续三帧，不跨缺失 Observation 形成候选。
- 双端实验模式中的旧事件 Implementation 已降级为 `reversal_candidate`：只保留图像坐标和 `experimental` 资格，不再写 `hit`、`landing` 或空中羽毛球的地面投影坐标；`hit_count/landing_count` 对所有模式均留空，实验候选另写 `reversal_candidate_count`。
- Diagnostic Demo 遇到缺失羽毛球样本立即清空轨迹；不同源分辨率的视频按各自尺寸缩放 bbox、图像点、球场角点和 Homography，消除“画面缩放但标注未缩放”的错位。
- 俯视图 Interface 对场外坐标返回拒绝，不再钳到边线，并显示 `projection rejected`；空中羽毛球不再投影到俯视球场。
- 全回合摘要明确标记为 `Full-rally summary`，且仅在回合最后一帧出现；实验事件显示为 `reversal candidates`，不再冒充击球次数。
- 修复羽毛球样例图在只有 1–2 个 rally 时，固定六宫格与严格 `zip` 长度不一致导致的崩溃，并去除首尾抽样产生的重复回合。
- 新增回归覆盖稀疏帧距离、半场越界、反转跨缺失、实验候选语义、无效轨迹断线、越界拒绝、摘要时机、混合分辨率 Homography 和少量 rally 图表。
- 验证结果：针对性测试 18 项通过；项目全量测试 60 项通过；`compileall` 与 `git diff --check` 通过。
- 剩余限制：12 m/s 目前是保守的全局质量阈值，尚未进入严格配置与数据集校准；末帧摘要依赖视频容器报告可靠的帧数；覆盖面积仍是点集凸包，下一阶段应增加最少采样数、覆盖率与稳定性资格。

### 2026-08-29：P0-D1 完成

- 目标：Main View 必须成为正式管线首个质量门禁，原始转播不得再绕过它直接进入 rally segmentation。
- `main_view` YAML 配置已进入类型化 `PipelineConfig`，过去存在但被解析器忽略的 `posthoc_min_quality_score` 已删除；管线参数直接来自同一配置对象。
- `StageName` 新增并前置 `main_view`。Stage Result 记录原视频输入、采样参数以及帧评分、接受区间、质量表和 timeline 四项必需 Artifact。
- 接受区间为零时，`accepted main-view segments` Artifact 形成 `empty`，管线立即停止，rally Implementation 不会被调用。
- rally Stage 改用现有 timeline-constrained Implementation，输入同时声明 Source Match 与 `main_view_timeline.json`；采样帧按半开区间筛选，导出端点被钳制在接受区间内部。
- Main View 标签 Interface 统一为 `MainViewLabel.MAIN_VIEW` / `MAIN_VIEW`。历史 `MAIN_LIVE_VIEW`、`MAIN_BIRDSEYE_LIVE` 仅由兼容 Adapter 归一化，未知自由字符串直接拒绝。
- scoreboard 选择仍保留，但只能作用于已经通过 Main View 门禁的候选区间。
- 为防旧 run 静默复用未经过 Main View 的 rally，缺少 Main View 成功记录但已有下游成功 Stage 的 manifest 会拒绝 resume，要求新 run ID 或显式 `--force`。
- 新增回归覆盖规范标签、历史标签归一化、timeline 帧约束、全非主视角停止、Main View 非零返回码以及 timeline 向 rally 的传播。
- 验证结果：项目全量测试 64 项通过；`compileall` 与 `git diff --check` 通过。
- 剩余限制：当前 Main View 是基于采样帧的启发式评分，尚未在冻结标注集上校准；rally 的空结果、端点语义和未实际参与决策的部分参数继续归入 P0-D2。

### 2026-08-29：P0-D2 完成

- 目标：Main View 连续片段不能自动冒充 Usable Rally；每个候选必须具有可审计的接受/拒绝结论，所有帧区间必须遵守同一端点语义。
- 新增 Usable Rally 分类 Implementation：只使用 Main View 内同时满足球场结构、线条质量和运动阈值的 `is_candidate` 样本作为活动证据，按 `max_gap_seconds` 对真实帧差分组，再施加前后上下文、最少活动样本和最小/最大时长门槛。2026-08-29 的真实素材回归进一步把运动证据从全画面均值改为归一化球场主体区域局部帧差，两个分数均写入审计 CSV；正式实现已迁入 `rally/activity.py`。
- 新增 `rally_decisions.csv` Artifact。每行记录候选 ID、来源 Main View、半开区间、证据样本数、`accepted/rejected` 和稳定 reason code；`rallies.csv` 只包含 accepted Usable Rally。
- 没有 accepted 行时，Stage Result 使用 `rejected`，并在 message 汇总 `no_active_play_evidence`、`insufficient_active_samples`、时长不合格或 scoreboard 未确认等原因；它不再被表达为成功、空文件或运行失败。
- scoreboard Adapter 仍可把活动证据候选进一步收紧为 `scoreboard_confirmed`；没有对应更新的候选改写为 `no_scoreboard_confirmation`。
- 正式 rally Interface 删除未参与决策的 `pad_before_seconds`、`pad_after_seconds`、`max_pre_context_seconds`、`max_post_context_seconds`、`allowed_context_drop_samples` 和假手工复核开关；改为实际生效的 `pre_context_seconds`、`post_context_seconds`、`min_active_samples`。旧脚本参数只保留在冻结兼容 Adapter。
- Frame Interval 统一为 `[start_frame, end_frame)`；分析筛选、Main View 尾段裁剪、rally 元数据、时间换算和视频写入使用同一语义。最小/最大时长使用取整后的帧数比较，不使用浮点秒直接比较；视频导出严格检查实际帧数等于 `end_frame - start_frame`，不足时失败并移除本次不完整输出。
- `RALLY_FIELDS` 新增区间约定、来源 Main View、资格、原因和证据样本数字段；实验配置已迁移到新的类型化参数。
- 新增回归覆盖活动证据接受、无证据拒绝、Main View 外帧排除、半开区间视频帧数、Stage `rejected` 状态和全部实验配置解析。
- 验证结果：项目全量测试 70 项通过；`compileall` 与 `git diff --check` 通过。
- 剩余限制：活动/线条证据仍是启发式而非人工标注真值，可能把热身或持续主视角空档误判为候选；下一阶段应引入发球/对打/死球状态机及冻结 Main View/rally 标注集，校准 precision、recall 和 reason 分布。独立无 timeline 的旧 `rally segment` 命令仍是兼容入口，不代表新研究管线语义。

### 2026-08-29：P1-A1 完成

- 目标：错误配置必须在运行前精确失败；一次 run 的全部路径必须来自统一布局 Interface。
- `PipelineConfig` 新增类型化 `DataConfig`，配置解析同时拒绝未知顶层键和分区内部未知键；类型错误与语义错误会聚合成一份报告，每条均包含完整字段路径。
- 增加数值范围和关联约束：采样/时长/尺寸必须为正，缺失窗口不得为负，比例必须位于 `[0,1]`，EMA 位于 `(0,1]`，ROI 必须完整位于归一化画面内，最小/最大阈值保持有序。
- 球员角色只允许正式近端 `[near]` 或实验双端 `[near, far]`；检测器只允许 `heuristic/yolo`，羽毛球模型只允许 `motion_bright_baseline/tracknet`，TrackNet 启用时必须提供权重路径。
- 删除无 Implementation 消费的伪配置：阶段输出路径、`court_type/save_homography`、`tracker/pose_model/use_ankle_midpoint`、`future_model_interface`、顶层 `video/outputs` 及项目展示元数据。旧配置再次提供这些键会收到迁移所需的精确 unknown-key 错误。
- 新增 `RunLayout` Module，集中拥有 run 根、manifest/report、Main View、rally、annotations、outputs、标定、跟踪、平滑、战术、图表和 demo 的路径规则，并拒绝包含目录跳转的 `run_id`。
- 单次主管线、Main View 独立入口和 batch Adapter 均消费同一 RunLayout Interface；`data.runs_dir` 和 `--runs-dir` 对成功路径、失败报告与 batch summary 使用同一解析规则。
- 新增回归覆盖聚合错误、伪配置拒绝、非法角色、运行目录逃逸、自定义 `runs_dir` 与关键 Artifact 映射。
- 验证结果：P1-A1 针对性测试通过；项目全量测试 78 项通过；`compileall` 与 `git diff --check` 通过。
- 剩余限制：配置/输入/模型/代码和上游 Artifact 指纹尚未写入 resume 决策；产物 schema version、唯一键和状态字段也尚未统一，继续归入 P1-A2。

### 2026-08-29：实验双端 Observation 与标准俯视球场

- 目标：在不破坏当前近端基线的前提下恢复远端人物识别，并把 Diagnostic Demo 的简化四分框升级为完整羽毛球场标线。
- 默认 `player_tracking.roles` 从 `[near]` 改为 `[near, far]`，主管线和独立追踪入口都会生成双端 Observation；`[near]` 继续保留为稳定回退模式，heuristic synthetic smoke 配置显式固定为近端，避免实验用途被默认值改变。
- 角色关联 Implementation 继续以 Validated Calibration 的 Homography 作为强门禁：near 只消费近半场候选并优先分配，far 只消费远半场未使用候选，双方维护独立 bbox、速度和缺失计数；因此恢复 far 不会把同一候选再次写到 near。
- 双端模式仍标记为 `experimental_two_player` / experimental Dual-side Observation；`near/far` 只是场地角色，不等于稳定运动员身份，也不恢复无依据的击球、落点或胜负结论。
- Diagnostic Demo 新增标准场地绘制 Implementation：6.10m × 13.40m 双打外框、5.18m 单打边线、距网 1.98m 前发球线、距底线 0.76m 双打后发球线、两侧分区中线和球网；右上角面板按画面高度响应式放大，并显示 FAR/NEAR 与球员位置标签。
- 俯视图仍只消费标准 court coordinates；越界点继续拒绝而不钳制，保持 Diagnostic Demo 不制造虚假位置的不变量。
- 新增回归覆盖完整九条内部标线的名称、相对位置和中线不跨越球网；配置、角色关联、战术模式与 demo 集成测试继续通过。
- 真实 All England rally 验证输出 966 条 Observation：far 492 帧、near 474 帧；far 平均 `court_y=2.527`、near 平均 `court_y=10.796`，474 个双人同现帧中脚点小于 20 像素的重叠为 0。新 Diagnostic Demo 输出 492 帧并完成人工抽帧复核，远端框和俯视点均落在远端运动员上。
- 验证结果：项目全量测试 79 项通过；`compileall` 与 `git diff --check` 通过。
- 剩余限制：本批恢复的是远端场地角色检测，不是 ReID；真实视频上的远端 precision/recall、轨迹覆盖率和 ID switch 仍需在冻结标注集上评估。
