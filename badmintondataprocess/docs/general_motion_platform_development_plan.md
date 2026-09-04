# BBA 通用人体运动分析平台与个体化建模开发计划

状态：待执行<br>
建立日期：2026-09-05<br>
适用范围：BBA 现有羽毛球分析管线，以及后续健身、跑步和其他运动垂类<br>
相关文档：[骨骼动作细节分析开发计划](biomechanics_development_plan.md)、[现有开源方案评估](biomechanics_open_source_evaluation.md)、[重构后架构](refactored_architecture.md)

## 1. 决策摘要

BBA 后续不应继续把“骨骼分析”写成羽毛球管线中的一组专用公式，也不应自行实现姿态模型、三维重建、逆运动学或通用动作识别网络。推荐路线是：

1. 在当前仓库内抽出一个与运动项目无关的 `motion` 核心，保留 BBA 作为第一个垂类插件。
2. 继续使用 RTMPose 产生二维人体观测；不重复推理，也不改变现有球员身份跟踪流程。
3. 使用 OpenSim 负责个体化骨骼模型、关节约束、缩放和逆运动学，不自行编写生物力学求解器。
4. 参考 Pose2Sim/OpenCap 的数据流和质量控制；多机位验证优先复用 Pose2Sim。
5. 单目三维重建采用可替换后端。OpenCap Monocular/WHAM 先用于研究基线，不能在许可证未解决时成为商业运行时的硬依赖。
6. 360° 视频首先服务于“个体骨架与体段比例标定”，而不是追求写实人体模型。外观网格是可选资产，不是关节角计算的前置条件。
7. 第一个非羽毛球垂类选择受控机位下的深蹲，再扩展弓步和俯卧撑；先证明可重复测量，再增加自动动作评价。
8. 所有结果必须区分 `observed`、`estimated`、`derived` 和 `not_eligible`。低置信度时拒绝输出，不能用平滑或模板强行补成“正确结果”。

目标架构不是“一个模型识别所有运动”，而是：

```text
通用观测与人体建模核心
    + 运动项目知识包
    + 经验证的动作标准
    + 可追溯的质量与不确定性
    = 可扩展的运动分析产品
```

## 2. 产品边界

### 2.1 第一阶段承诺

- 用户完成一次引导式身体标定，建立版本化的 `SubjectProfile`。
- 后续上传受支持的单目视频，系统复用该档案约束人体尺度和体段比例。
- 输出二维观测、三维估计、动作阶段、关节角度、节奏、对称性、稳定性和可解释的改进提示。
- 对每项指标显示数据来源、置信度、适用条件和拒绝原因。
- BBA 羽毛球分析继续工作，不因通用核心重构而回退。

### 2.2 第一阶段不承诺

- 不把单目估计描述为实验室级动作捕捉。
- 不从普通视频直接宣称精确关节力矩、肌肉力量、地面反作用力或受伤概率。
- 不把 360° 外观扫描等同于精确骨骼、关节中心或实时三维动作。
- 不使用单个职业运动员模板给所有用户打绝对“标准/不标准”分数。
- 不提供医疗诊断、康复处方或治疗建议。
- 不要求写实人脸、皮肤纹理或可识别外观；这些数据会增加隐私风险，却不提高核心生物力学结果。

### 2.3 建议的产品模式

| 模式 | 首次采集 | 后续视频 | 输出级别 | 目标用户 |
|---|---|---|---|---|
| Quick | 身高、体重、正面与侧面静态片段 | 单目、固定机位 | 二维投影指标 + 有限三维估计 | 普通个人用户 |
| Pro | 引导式 360° A-pose + 功能标定动作 + 尺度参照 | 单目或双目、固定机位 | 个体化三维运动学 + 更严格质量报告 | 教练、工作室、研究试点 |
| Validation | Pro 标定 + 2 台以上同步相机/参考设备 | 多视角 | 评测与校准，不作为普通入口 | 内部测试、合作机构 |

## 3. 现有 BBA 能力与可复用资产

当前代码已经具备通用平台最困难的一部分工程基础：

- `tracking/player/pose.py`：RTMPose/YOLO Pose 推理、COCO-17 关键点、置信度与球员候选绑定。
- `analysis/biomechanics/kinematics.py`：带资格门槛的二维关节角和稳定性基础量。
- `analysis/biomechanics/events.py`：多证据事件候选、非极大值抑制和事件资格。
- `analysis/biomechanics/phases.py`：时序平滑和动作阶段分解。
- `analysis/biomechanics/descriptors.py`：逐事件、逐回合与全场聚合。
- `analysis/biomechanics/evaluation.py`：预测和人工真值的冻结评估入口。
- `core/schemas.py`、`core/artifacts.py`、`core/run.py`：结构化产物、阶段状态和可恢复运行。
- WebUI：视频上传、运行进度、人工场地标定和结果交付。

必须避免的做法是直接把这些文件复制一份并改成“fitness”。应先抽离通用契约，再让羽毛球和健身使用同一实现。

## 4. 开源技术选型

### 4.1 采用原则

每个候选依赖必须分别核对：

- 源代码许可证；
- 模型结构许可证；
- 预训练权重许可证；
- 数据集许可证；
- 商业使用、再分发和云端服务限制；
- CUDA、操作系统、Python 与现有 `good-badminton` 环境的兼容性。

“GitHub 仓库是 MIT/Apache”不代表下载的模型权重和训练数据可商用。所有依赖在固定 commit/版本后必须生成 `THIRD_PARTY_NOTICES` 和机器可读许可证清单。

### 4.2 推荐矩阵

| 项目 | 能力 | 许可证/约束（2026-09-05 核对） | BBA 决策 |
|---|---|---|---|
| [MMPose / RTMPose](https://github.com/open-mmlab/mmpose) | 二维多人姿态估计 | 代码 Apache-2.0；权重仍需逐项核对 | **生产主干保留**，沿用现有 RTMLib 适配器 |
| [OpenSim](https://github.com/opensim-org/opensim-core) | 肌骨模型、模型缩放、IK/ID、分析工具 | Apache-2.0；第三方组件另行核对 | **生产核心候选**，不自研 IK/肌骨求解器 |
| [Pose2Sim](https://github.com/perfanalytics/pose2sim) | 多相机标定、同步、关联、三角化、过滤、OpenSim IK | BSD-3-Clause | **验证与多视角后端候选**；优先调用/适配，不复制整条管线 |
| [Sports2D](https://github.com/davidpagnon/Sports2D) | 单目二维关节角、过滤和报告 | BSD-3-Clause | **二维基线与定义参考**；只复用纯计算思路，避免重复跑姿态模型 |
| [MMAction2](https://github.com/open-mmlab/mmaction2) | RGB/骨骼动作识别、时序定位、PoseC3D/ST-GCN 系列 | Apache-2.0；具体权重/数据另查 | **训练与实验基础设施**；不直接用通用类别权重给健身纠错 |
| [WHAM](https://github.com/yohanshin/WHAM) | 世界坐标下的单目时序人体重建 | 代码 MIT，但运行依赖 SMPL 资产 | **研究后端**；许可证链未解决前不进入正式商业运行时 |
| [OpenCap Core](https://github.com/opencap-org/opencap-core) | 多手机三维运动学/动力学完整参考流程 | Apache-2.0 | **架构与验证参考**；选择性适配，不整仓嵌入 |
| [OpenCap Monocular](https://github.com/utahmobl/opencap-monocular) | 单手机三维运动学、相机/姿态优化、OpenSim | PolyForm Noncommercial 1.0.0；SMPL 资产另受限 | **只做非商业研究基线**，商业使用前必须取得授权或替换实现 |
| [MMHuman3D](https://github.com/open-mmlab/mmhuman3d) | 参数化人体模型工具箱与数据约定 | 框架 Apache-2.0；所支持算法/人体模型可能有额外许可 | **仅作适配与格式参考**，不默认引入全部依赖 |
| [SMPL-X](https://github.com/vchoutas/smplx) | 参数化人体网格 | 默认限非商业科学研究；商业使用需单独授权 | **禁止成为未授权生产硬依赖** |
| [COLMAP](https://github.com/colmap/colmap) | SfM/MVS、相机重建 | 库代码 New BSD；构建时第三方依赖可能改变分发义务 | **离线实验工具**；不把普通 COLMAP 点云当作骨骼模型 |
| [AliceVision/Meshroom](https://github.com/alicevision/meshroom) | 摄影测量工作流 | MPL-2.0，第三方组件另查 | **备选离线扫描基线**，不进入首版用户流程 |
| [RepNet](https://github.com/google-research/google-research/tree/master/repnet) | 类别无关重复动作计数 | Google Research 源代码 Apache-2.0；权重另查 | **重复计数对照基线**，不是动作质量评价器 |

### 4.3 最终组合

正式主干优先采用：

```text
OpenCV / FFmpeg
    -> RTMPose（2D Observation）
    -> BBA 身份与时序跟踪
    -> 可替换的 3D Reconstruction Backend
    -> OpenSim（个体化模型 + IK）
    -> BBA Motion Metrics
    -> MMAction2 或自有轻量时序模型（动作类别/阶段）
    -> 规则与统计评价层
```

对照与验证采用：

```text
Sports2D（平面动作二维基线）
Pose2Sim（多视角三维基线）
OpenCap / OpenCap Monocular（论文与研究基线）
```

### 4.4 明确不自行实现的组件

- 通用二维人体姿态网络；
- 摄像机内参标定、PnP、基础 Homography 和多视角三角化算法；
- 通用 SfM/MVS 摄影测量引擎；
- 肌骨模型缩放和通用逆运动学求解器；
- 通用视频编解码器；
- 通用动作识别训练框架；
- 三维场景查看器的底层渲染引擎。

BBA 自研价值应集中在：数据契约、质量控制、个体档案、单目约束融合、运动项目知识包、可解释评价、产品交互和真实世界验证。

## 5. 为什么 360° 扫描不能直接解决后续单目三维问题

360° 静态视频可以提高以下先验的准确度：

- 身高尺度和体段长度；
- 肩宽、髋宽和躯干比例；
- 静态关节中心的初始位置；
- 可选的人体外形参数。

但后续单目视频仍然缺少：

- 每一帧的绝对深度；
- 被遮挡肢体的真实姿态；
- 出平面旋转方向；
- 相机内外参和地面方向；
- 脚底接触、滑动和受力信息。

因此正确模型是：

```text
二维观测
  + 个体骨架先验
  + 时序连续性
  + 相机与地面约束
  + 关节活动范围
  + 接触/动作先验
  -> 带不确定性的个体化三维估计
```

而不是“把二维骨骼直接贴到 360° 网格上”。网格只决定身体形状，不会自动消除投影歧义。

## 6. 个体建模方案

### 6.1 首版：个体骨架档案，不做写实网格

首版 `SubjectProfile` 应通过下列采集获得：

1. 用户输入身高；体重可选，但启用动力学估计时必须提供。
2. 相机固定、全身入镜，画面中放置已知尺寸的尺度参照。
3. 正面 A-pose 静止 2–3 秒。
4. 左/右侧面各静止 2–3 秒。
5. 缓慢深蹲 3 次、抬臂 3 次、原地踏步 5–10 秒，作为功能标定。
6. Pro 模式再录制一圈低速 360° 视频；系统抽取合格视角，而不是使用每一帧。

输出重点是体段长度、关节中心、左右侧差异、模型缩放参数和标定不确定性。优先通过 OpenSim `ScaleTool` 生成用户特定 `.osim`，并保存重投影与尺度残差。

### 6.2 后续：可选外观网格

如确有虚拟人展示需求，再增加 `BodyShapeBackend`：

- 输入为经质量筛选的多视角静态帧、轮廓、二维关键点和尺度参照；
- 输出为去纹理或匿名纹理网格、参数化形状和骨架注册结果；
- 外观网格与生物力学骨架分开存储、单独版本化；
- 未取得商业授权的 SMPL/SMPL-X 后端只能在研究配置启用；
- COLMAP/Meshroom 只作为相机和表面重建实验基线，不能单独生成关节定义。

### 6.3 标定质量门槛

以下任一条件不满足时，不创建可用于正式分析的档案：

- 全身（头顶到脚底）连续可见；
- 尺度参照可识别且尺寸已知；
- 关键点覆盖率、左右一致性和重投影误差达标；
- 运动期间无明显宽松衣物遮挡和多人误绑定；
- 360° 采集的角度覆盖、清晰度和人体静止程度达标；
- 功能标定动作没有越界、截断或身份切换。

允许用户重录失败的子步骤，不要求整套重来。

## 7. 顶层架构

```text
                         Subject Onboarding
                 height / scale / A-pose / 360° / calibration motions
                                      |
                                      v
                              Subject Profile Store
                                      |
Video Upload -> Decode -> Person Track -> 2D Pose Observation -> Session Calibration
                                      |                         |
                                      +------------+------------+
                                                   v
                                      3D Reconstruction Adapter
                                    / research / production / none
                                                   |
                                                   v
                                   OpenSim Personalised IK Adapter
                                                   |
                                                   v
                                     Canonical Motion Sequence
                                                   |
                         +-------------------------+-------------------------+
                         |                         |                         |
                 Badminton Pack              Fitness Pack               Running Pack
              stroke/footwork/court       squat/lunge/push-up        gait/cadence/asymmetry
                         |                         |                         |
                         +-------------------------+-------------------------+
                                                   v
                                  Metrics + Eligibility + Uncertainty
                                                   |
                                      Report / WebUI / CSV / JSON / Video
```

### 7.1 模块建议

第一阶段仍保留单仓库，新增以下内部边界：

```text
src/badminton_data_process/
  motion/
    contracts.py
    quality.py
    subjects/
      profile.py
      capture.py
      scale.py
    pose/
      conventions.py
      adapters/rtmpose.py
    reconstruction/
      interfaces.py
      none_backend.py
      research_wham.py
    biomechanics/
      opensim_adapter.py
      kinematics_2d.py
      kinematics_3d.py
    actions/
      interfaces.py
      segmentation.py
      repetition.py
    evaluation/
      metrics.py
      uncertainty.py
      benchmark.py
  sports/
    badminton/
      events.py
      phases.py
      bst.py
      footwork.py
    fitness/
      squat.py
      lunge.py
      pushup.py
      standards/
```

不要立即移动全部文件。先新增契约和兼容导入，逐模块迁移；现有公共 import 至少保留一个发行周期。

### 7.2 现有文件迁移映射

| 现有位置 | 处理方式 | 目标 |
|---|---|---|
| `tracking/player/pose.py` 的数据类 | 抽出，旧位置 re-export | `motion/pose/conventions.py` |
| `tracking/player/pose.py` 的 RTMPose 实现 | 保留行为，改为 backend adapter | `motion/pose/adapters/rtmpose.py` |
| `analysis/biomechanics/kinematics.py` | 通用纯函数迁移 | `motion/biomechanics/kinematics_2d.py` |
| `analysis/biomechanics/events.py` | 羽毛球专属 | `sports/badminton/events.py` |
| `analysis/biomechanics/phases.py` | 先保留羽毛球定义，共用分段原语下沉 | `sports/badminton/phases.py` + `motion/actions/segmentation.py` |
| `analysis/biomechanics/descriptors.py` | 拆分通用稳定性与羽毛球步法 | `motion/biomechanics/*` + `sports/badminton/footwork.py` |
| `analysis/biomechanics/bst.py` | 保持羽毛球插件 | `sports/badminton/bst.py` |
| `analysis/biomechanics/evaluation.py` | 抽出通用评测器，保留标签适配器 | `motion/evaluation/benchmark.py` |

## 8. 核心数据契约

### 8.1 `SubjectProfile`

```json
{
  "schema_version": "bba_subject_profile_v1",
  "subject_id": "opaque-id",
  "created_at": "ISO-8601",
  "consent_version": "...",
  "anthropometry": {
    "height_m": 1.75,
    "mass_kg": null,
    "measurement_source": "user_and_scale_reference"
  },
  "skeleton": {
    "backend": "opensim_scale_v1",
    "model_ref": ".../subject.osim",
    "segment_lengths_m": {},
    "joint_centers_ref": "...",
    "fit_residuals": {}
  },
  "body_shape": {
    "backend": "none",
    "asset_ref": null,
    "license_class": null
  },
  "calibration_quality": {
    "status": "valid",
    "score": 0.0,
    "reject_reasons": []
  },
  "provenance": {
    "code_commit": "...",
    "model_ids": [],
    "source_asset_hashes": []
  }
}
```

原始 360° 视频不是档案本身。用户应可以选择“生成档案后删除原视频”，档案删除也必须级联删除派生网格与缓存。

### 8.2 `MotionObservation2D`

必须包含：视频/帧/时间戳、track id、关键点命名约定、像素坐标、置信度、可见性、bbox、模型 id、是否直接观测。不能只保存无置信度的坐标数组。

### 8.3 `MotionEstimate3D`

必须包含：

- 相机坐标与世界/地面坐标的明确坐标系；
- 每关节位置、旋转和置信度；
- `source=estimated`，不得伪装成观测值；
- 使用的 `subject_profile_id/version`；
- 2D 重投影误差、骨长残差、接触残差和时序残差；
- 后端名称、模型/权重 id、代码 commit；
- 帧级与序列级 eligibility。

### 8.4 `MotionMetric`

```text
metric_id
value / unit
coordinate_system
observation_basis        # observed_2d / estimated_3d / opensim_derived
eligibility
reject_reason
confidence
uncertainty_interval
algorithm_version
evidence_frame_interval  # 半开区间
```

### 8.5 `EvaluationFinding`

评价不得直接由大语言模型自由生成。结构化结果至少包括：

```text
finding_id
action_id + phase_id
metric_refs[]
rule/model version
expected_range + source
observed_value
severity
confidence
eligibility
user-facing template key
```

语言模型只能在这些已验证事实之上做措辞整理，不能补写未测量的“膝盖压力”“受伤风险”等结论。

## 9. 动作知识包

每个动作必须是版本化配置与代码的组合，而不是散落在页面里的阈值：

```yaml
action_id: fitness.squat.bodyweight
version: 1
supported_views: [front, side, oblique]
required_profile_level: quick
phases: [setup, descent, bottom, ascent, finish]
required_joints: [hip, knee, ankle, shoulder]
metrics:
  - knee_flexion
  - hip_flexion
  - trunk_inclination
  - left_right_timing_delta
rules:
  - id: depth_relative_to_profile
    basis: estimated_3d
    applicability: controlled_camera_only
references: []
```

动作标准的建立流程：

1. 明确动作变体、器械、机位和目标人群。
2. 由至少两名领域专家给出阶段、指标含义和禁用场景。
3. 标注同一批样本并统计标注一致性；分歧未解决的规则不进入产品。
4. 优先使用连续指标和个人基线，不强行二元判定。
5. 规则阈值必须记录来源、版本和适用范围。
6. 每次规则或模型升级都在冻结测试集上回归。

## 10. 第一条垂直切片：自重深蹲

选择深蹲的原因是重复明显、动作阶段相对稳定、用户易按引导固定机位，并且 OpenCap 等方案已有可比较结果。首版只支持：

- 单人、全身入镜；
- 固定相机；
- 正面或近似侧面机位；
- 自重深蹲，不含杠铃、深蹲架遮挡和他人进入画面；
- 连续 3–10 次。

首版输出：

- 重复次数与每次起止时间；
- 下蹲、底部、起身和结束阶段；
- 每次动作时长、离心/向心时长比；
- 二维投影角度；三维后端通过验证后再增加三维角度；
- 左右时间差、最低点一致性、躯干轨迹稳定性；
- 与用户本人历史基线的变化；
- 每条结论的适用性和置信度。

首版不输出“膝内扣导致受伤”等医学化结论。正面观察到的膝相对足部偏移只能作为画面投影描述；通过个体化三维和验证后才能升级表述。

## 11. 分批实施计划

工期是单人全职的粗略工程量，不是发布日期。每批都必须独立可回滚、可测试，不允许连续几个月后一次性交付。

### GM-00：许可证与架构闸门（3–5 天）

任务：

- [ ] 新增 ADR：通用 motion 核心留在单仓库，BBA 为垂类插件。
- [ ] 建立 `third_party_components.yaml`，分别记录代码、权重、模型和数据许可。
- [ ] 对当前 RTMPose、TrackNet、BST 权重补齐来源与哈希。
- [ ] 为 OpenCap Monocular、SMPL/SMPL-X 标记 `research_only`。
- [ ] 决定生产人体模型路线：购买商业许可，或采用不依赖受限网格的 OpenSim 骨架方案。

退出条件：正式配置不能加载 `research_only` 组件；CI 能检测缺失许可证元数据。

### GM-01：通用契约与兼容层（1–2 周）

任务：

- [ ] 新建 `motion/contracts.py`、`motion/quality.py`。
- [ ] 定义关键点 convention 映射，不把 COCO-17 写死在全部下游模块。
- [ ] 抽出 `MotionObservation2D`、`MotionMetric`、`EligibilityResult`。
- [ ] 旧的 `PoseKeypoint/PoseObservation` 保持兼容导入。
- [ ] 将二维角度计算迁移为无 I/O 的纯函数，CSV 读写放在 adapter。
- [ ] 为 schema version、坐标系和单位增加契约测试。

退出条件：现有 BBA 全量测试通过，同一视频关键产物逐字段等价；没有新跑一次 RTMPose。

### GM-02：SubjectProfile Quick（1–2 周）

任务：

- [ ] WebUI 增加用户档案与 Quick 标定向导。
- [ ] 支持身高、可选体重、正面/侧面静态视频和尺度参照。
- [ ] 增加全身完整性、模糊、遮挡、多人和尺度参照检查。
- [ ] 估计稳健体段比例，保存中位数和离散度。
- [ ] 增加档案版本、撤销、删除和运行绑定。

退出条件：重复标定的主要体段长度变异系数达到预定门槛；失败样本明确拒绝，不生成“有效档案”。

### GM-03：Pro 360° 采集（2–3 周）

任务：

- [ ] WebUI 提供相机距离、光照、服装、A-pose 和绕行速度引导。
- [ ] 从视频抽取 8–16 个合格视角，并显示覆盖圆环和需补拍区间。
- [ ] 检测人体移动、运动模糊、脚部截断、尺度漂移和背景他人。
- [ ] 建立 `BodyShapeBackend` 接口和 `none` 默认实现。
- [ ] 在隔离实验配置测试 COLMAP/Meshroom 与参数化人体拟合，不进入默认运行时。

退出条件：360° 采集能稳定形成完整视角包；即使网格后端关闭，也能为个体骨架标定提供增益。

### GM-04：OpenSim 个体骨架（2–4 周）

任务：

- [ ] 固定 OpenSim 版本并在 `good-badminton` 环境做导入/CLI 预检。
- [ ] 选定适合首批动作的通用肌骨模型，并记录模型许可和引用。
- [ ] 将 SubjectProfile 体段测量转换为 ScaleTool 输入。
- [ ] 生成版本化 `.osim`、缩放参数和残差报告。
- [ ] 建立 OpenSim IK adapter、超时、缓存和错误分类。
- [ ] 保留 OpenSim 原生 `.trc/.mot` 产物，同时输出 BBA JSON/CSV。

退出条件：同一档案重复运行结果确定；不合理缩放被拒绝；模型/输入/输出单位测试通过。

### GM-05：单目三维后端研究闸门（3–6 周）

任务：

- [ ] 定义 `ReconstructionBackend`，输入 2D 序列、相机信息和 SubjectProfile。
- [ ] 实现 `none` 后端，确保无三维模型时二维管线完整可用。
- [ ] 隔离接入 WHAM/OpenCap Monocular 作为研究基线。
- [ ] 评估固定相机内参、地面/重力方向、脚接触和个体骨长优化。
- [ ] 与 Pose2Sim 多视角结果比较 MPJPE、角度误差和重投影误差。
- [ ] 完成生产后端许可证审查；不通过则继续研究模式，不得绕过。

退出条件：在冻结的深蹲/弓步样本上达到预注册精度，或明确判定三维暂不上线。没有“为了完成计划而强行启用”。

### GM-06：深蹲端到端（2–4 周）

任务：

- [ ] 新建 `fitness.squat.bodyweight` 动作包。
- [ ] 用骨骼序列实现重复计数和阶段分割；RepNet 只作对照。
- [ ] 输出节奏、幅度一致性、左右差异和稳定性。
- [ ] 增加专家规则版本和适用条件。
- [ ] WebUI 增加“健身—自重深蹲”入口、录制指南和逐次报告。
- [ ] 生成带阶段、骨骼、关键角度和拒绝提示的视频。

退出条件：对新用户可从上传到报告一键完成；不支持机位能在长推理前被识别；每条评价可追溯到帧和指标。

### GM-07：验证与置信度校准（持续 3–6 周，和 GM-06 交叠）

任务：

- [ ] 建立按“人”隔离的 train/validation/test 划分，禁止同一用户跨集合泄漏。
- [ ] 同时覆盖体型、服装、光照、手机、机位和动作质量差异。
- [ ] 用多相机 Pose2Sim/OpenCap 或实验室设备获取参考数据。
- [ ] 双专家标注动作阶段和评价，统计一致性。
- [ ] 对置信度做校准，验证 abstention 是否真的集中在错误样本。
- [ ] 按相机视角、性别/体型区间和运动经验分组报告误差。

退出条件：冻结测试报告可复现；上线阈值来源清晰；不存在只报告总体均值、掩盖远端或特殊人群失败的情况。

### GM-08：扩展动作与商业加固（4–8 周）

任务：

- [ ] 按相同契约增加弓步和俯卧撑，不复制核心代码。
- [ ] 增加任务队列、GPU 并发限制、失败重试和产物生命周期。
- [ ] 支持本地处理/私有部署配置，减少身体视频离开用户设备的需求。
- [ ] 增加审计日志、授权撤销、数据删除和模型卡。
- [ ] 邀请教练/工作室试点，优先评估报告是否能指导复测，而非只看页面效果。

退出条件：第二、第三动作包只新增垂类规则和少量适配代码；核心 schema 和运行时无需分叉。

## 12. 验收指标

以下是开发启动门槛，须在 GM-00 中预注册并由实测修订：

| 层级 | 指标 | 首版建议门槛 |
|---|---|---|
| 上传/解码 | 支持样本成功进入预检 | 100%，失败必须有可操作原因 |
| 身份跟踪 | 受控单人视频身份切换 | 0 次 |
| 2D 姿态 | 必需关键点有效覆盖 | ≥ 95%（受控采集） |
| 个体标定 | 重复标定主要体段长度 CV | ≤ 2–3%（待试验确认） |
| 3D 重建 | 支持动作的主要关节角 MAE | 目标 ≤ 8°，并逐关节报告 |
| 阶段分割 | 关键边界误差 | 目标 ≤ 3 帧或 100 ms，取较宽者 |
| 重复计数 | 完全正确的视频比例 | ≥ 95%（支持场景） |
| 评价 | 高严重度提示 precision | ≥ 90%，宁可少报，不可乱报 |
| 可靠性 | 不支持场景正确拒绝率 | ≥ 95% |

这些门槛不能从论文数字直接复制。OpenCap 的受控实验结果可作为上限参考，而不是 BBA 已达到的结果。

## 13. 测试策略

### 13.1 单元与契约测试

- 坐标系、角度方向、单位和左右侧定义；
- 缺失关键点不生成数值；
- `observed` 不能被平滑改写为 `estimated`；
- schema 向前兼容和版本拒绝；
- OpenSim adapter 的输入输出、超时和错误分类；
- 许可证策略阻止生产配置加载研究组件。

### 13.2 集成测试

- Quick 标定 -> SubjectProfile -> 单目深蹲 -> 报告；
- Pro 360° 部分失败后的补拍与恢复；
- 同一视频在无 3D、研究 3D、生产 3D 后端下的降级行为；
- BBA 原有完整视频分析回归；
- WebUI 中断、重启、断点恢复和删除数据。

### 13.3 冻结样本

至少建立：

- 10 个受控合格样本；
- 10 个机位/遮挡/截断失败样本；
- 5 个重复标定样本；
- 5 个多视角参考样本；
- 当前俯视角和低视角羽毛球样本各一组回归集。

真实验证集扩展前，所有“准确率”只能标记为内部工程指标。

## 14. WebUI 改造

WebUI 信息架构建议改为：

```text
1. 选择/创建用户档案
2. 选择运动项目与动作
3. 上传或录制视频
4. 采集质量预检
5. 确认分析能力与预计用时
6. 运行进度
7. 结果、证据、不确定性与下载
```

新增页面：

- `身体档案`：Quick/Pro 标定、质量、版本、重录、删除；
- `录制指南`：根据动作和机位动态展示拍摄要求；
- `动作详情`：逐次时间轴、阶段、角度曲线和证据帧；
- `结果可信度`：说明哪些是观测、估计、推导和不可分析；
- `开发中`：未通过验证的动作只展示路线，不提供伪结果。

现有 Gradio 可继续承载验证阶段。只有当多用户账号、任务队列、权限和长期档案成为瓶颈时，再评估前后端拆分；不为“看起来像产品”提前重写整套 WebUI。

## 15. 数据、隐私和合规

全身 360° 视频、身体尺寸和运动特征具有较高隐私敏感性。实施前必须：

- 明示采集目的、派生数据类型、保存期限和删除方式；
- 默认不采集姓名，人用随机 `subject_id`；
- 原视频、匿名骨架、外观网格和报告分级存储；
- 支持生成档案后自动删除原始 360° 视频；
- 对象存储和传输加密，下载链接有时效；
- 未成年人、医疗/康复用途和数据跨境另设闸门；
- 产品文案使用“运动表现参考/动作观察”，避免未经验证的医疗诊断表述。

## 16. 风险登记

| 风险 | 影响 | 缓解措施 |
|---|---|---|
| 单目深度歧义 | 三维角度错误却看起来合理 | 个体先验 + 相机/接触约束 + 重投影残差 + 拒绝机制 |
| 360° 采集时人体移动 | 外形或尺度失真 | 引导、关键帧筛选、运动检测、允许局部补拍 |
| SMPL 系列商业许可 | 无法合法发布生产功能 | adapter 隔离；OpenSim 骨架先行；商业授权决策前禁用 |
| 研究仓库依赖脆弱 | Windows/CUDA 环境经常失败 | 研究依赖与正式套件隔离；锁版本；预检；不污染默认运行时 |
| 通用模型类别不匹配 | “识别成功”但没有业务含义 | 用动作包自有标签和冻结数据训练；通用权重只作初始化 |
| 专家标准不一致 | 评价规则自相矛盾 | 双人标注、共识会、规则版本和一致性统计 |
| 指标被误解为医学结论 | 用户伤害与合规风险 | 限定措辞、适用性说明、高风险结论不上线 |
| 为通用化过早拆仓 | 开发速度下降、回归困难 | 第二垂类通过前保持单仓库和兼容层 |

## 17. 近期三批实际工作

### 下一批 A：先锁边界

1. 完成 GM-00 的 ADR 和第三方许可证清单。
2. 明确 OpenSim 模型选择与许可。
3. 定义 `SubjectProfile`、`MotionObservation2D`、`MotionMetric` schema。
4. 新增生产配置禁止研究后端的自动测试。

### 下一批 B：无行为变化地抽核心

1. 抽离 pose convention 与数据类。
2. 抽离二维运动学纯函数。
3. 保持旧 import 和现有 CSV 完全兼容。
4. 重跑 BBA 全量测试和俯视/低视冻结样本。

### 下一批 C：做可见的用户价值

1. 实现 Quick 身体档案向导。
2. 建立深蹲录制预检。
3. 先用二维骨骼完成计数、阶段和节奏报告。
4. 三维后端以 feature flag 接入，不阻塞二维产品闭环。

## 18. 推进规则

- 一次只推进一个编号批次；每批合并前更新本文状态和实测结果。
- 任何新框架先做 adapter spike，不允许把第三方内部类型扩散到 BBA 领域层。
- 任何新指标先定义坐标系、单位、资格和真值，再写计算代码。
- 任何动作评价先证明事件/阶段正确，再讨论“好坏”。
- 任何论文指标只有在 BBA 冻结样本复现后才能出现在宣传材料中。
- 当二维结果已足够回答问题时，不强制使用三维结果。
- 当三维不确定性过高时，系统必须降级或拒绝，而不是输出更平滑的错误曲线。

## 19. 参考资料

- [OpenCap：双手机运动学与动力学验证论文（PLOS Computational Biology, 2023）](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011462)
- [OpenCap Monocular：单手机三维运动学与动力学预印本（2026）](https://arxiv.org/abs/2603.24733)
- [OpenCap Core](https://github.com/opencap-org/opencap-core)
- [OpenCap Monocular source](https://github.com/utahmobl/opencap-monocular)
- [Pose2Sim](https://github.com/perfanalytics/pose2sim)
- [Sports2D 论文](https://joss.theoj.org/papers/10.21105/joss.06849)
- [OpenSim Core](https://github.com/opensim-org/opensim-core)
- [OpenSim Scaling 文档](https://opensimconfluence.atlassian.net/wiki/spaces/OpenSim/pages/53090000/Scaling)
- [MMPose / RTMPose](https://github.com/open-mmlab/mmpose)
- [MMAction2](https://github.com/open-mmlab/mmaction2)
- [WHAM](https://github.com/yohanshin/WHAM)
- [SMPL-X](https://github.com/vchoutas/smplx)
- [OpenCapBench](https://openaccess.thecvf.com/content/WACV2025/html/Gozlan_OpenCapBench_A_Benchmark_to_Bridge_Pose_Estimation_and_Biomechanics_WACV_2025_paper.html)
