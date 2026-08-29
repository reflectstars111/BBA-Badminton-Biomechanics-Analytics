# 上游新增能力迁移计划

状态：代码迁移已实施；真实冻结样本验收待补充  
建立日期：2026-08-29  
适用范围：`badmintondataprocess/` 新研究管线  
上游来源：`yo-WASSUP/Good-Badminton`，同步提交 `24f2208`  
本地合并提交：`807f292`  

## 1. 文档目的

本文件记录从上游新增代码中筛选出的可迁移能力，并把迁移工作拆成可独立验证、可独立回滚的小批次。后续每完成一项，应同步更新本文件中的状态、实际改动文件、测试结果和遗留问题。

这里的“迁移”不等于复制文件。上游新增代码仍属于旧管线，其数据结构、错误语义、输出目录和运行编排均不能直接进入新研究管线。迁移时只复用经过验证的算法或交互行为，并接入新管线已有的 Module、Interface、Stage Result、Artifact、Run Manifest 和 Run Layout。

## 2. 架构边界与不变量

迁移不得改变以下约束：

1. 自动检测结果只是校准候选，不是 `Validated Calibration`。
2. 只有通过几何、重投影、球场类型和时序稳定性检查的校准才能支持米制指标。
3. `missing`、`rejected`、`failed`、`not_eligible` 和 `success` 必须保持不同语义。
4. WebUI、CLI 和批处理必须调用同一个研究管线 Interface，不能各自编排分析步骤。
5. 所有新产物必须写入 Run Layout，并登记为 Artifact；不得自行创建时间戳输出目录。
6. 历史运行目录默认只读，不得由界面或分析阶段自动递归删除。
7. Near-only Analysis 和实验性的 Dual-side Observation 必须继续显式标注能力边界。
8. 新增启发式不能只凭演示视频观感进入默认配置，必须有正例、负例和回归测试。

目标校准数据流：

```text
Manual Points Adapter --------+
Green Contour Adapter --------+--> Calibration Candidate
Hough Line Adapter -----------+             |
                                             v
                                  Calibration Validation Module
                                             |
                           +-----------------+-----------------+
                           |                                   |
                           v                                   v
                 Validated Calibration              structured rejection
```

## 3. 上游新增内容盘点

| 编号 | 上游内容 | 结论 | 目标位置 | 优先级 |
|---|---|---|---|---|
| MIG-01 | 标准羽毛球场参考模型和完整场地线 | 迁移并统一现有重复常量 | `calibration/`、`visualization/` | P1 |
| MIG-02 | 完整球场线支持评分 | 迁移算法，纳入共享校准验证 | `calibration/` | P1 |
| MIG-03 | Hough 线段、交点和四边形候选 | 作为新的自动候选 Adapter 迁移 | `calibration/` | P1 |
| MIG-04 | 多帧候选选择 | 结合现有代表帧逻辑重写 | `calibration/` | P1 |
| MIG-05 | H.264、AAC、音频复用和浏览器兼容导出 | 迁移行为，不迁移旧错误处理 | `media/`、Artifact 导出阶段 | P2 |
| MIG-06 | Gradio 上传、预览、手工校正和结果展示 | 保留交互设计，重写为薄 Adapter | `webui/` 或独立应用层 | P3 |
| MIG-07 | README、截图、字体和示例媒体 | 仅按文档需要选择性保留 | 文档和演示资源 | P4 |
| MIG-08 | 人物躯干中心与地面接触点双锚点 | 新增显式语义，禁止用空中点生成正式场地坐标 | `tracking/player/`、`visualization/`、schemas | P1 |
| MIG-09 | COCO-17 姿态检测与骨架绘制 | 迁移为 YOLO Pose/RTMPose 可配置 Adapter | `tracking/player/pose.py`、`visualization/` | P1 |
| MIG-10 | 未清洗完整转播的一键 GPU 工作流 | 复用九阶段主管线，新增预检、生产配置、摘要和 PowerShell 薄入口 | `pipeline/full.py`、`scripts/run_full_analysis.ps1` | P1 |

明确不整块迁移：

- `badminton_analysis/court/mapper.py`
- `webui/pipeline.py`
- 旧 `BadmintonAnalysisSystem` 编排
- 自动清理历史输出目录的实现
- 旧管线的 ROI、球员跟踪和战术分析调用链

## 4. 分项迁移清单

### MIG-01：统一标准球场几何 Module

状态：`implemented`  
前置条件：无  
上游来源：`badminton_analysis/court/reference.py`  
现有重复实现：`src/badminton_data_process/visualization/demo.py`

#### 要迁移的能力

- 双打场地尺寸：`6.10 m × 13.40 m`。
- 单打边线偏移、双打后发球线、前发球线、中线和球网位置。
- 标准场地线段集合。
- 标准场地线到图像四边形的投影能力。

#### 不直接迁移的内容

- 旧类名和无类型约束的字典返回值。
- 仅服务旧 `CourtMapper` 的坐标表达。
- 与 OpenCV 预览绘制绑定的逻辑。

#### 修改方向

1. 在新管线包内建立标准球场几何的唯一事实来源。
2. 将球场物理尺寸、标准点和标准线集中到该 Module。
3. 让校准验证和俯视演示通过稳定 Interface 使用同一份几何定义。
4. 删除或改写 `demo.py` 中重复的尺寸和线段常量。
5. 明确坐标约定、角点顺序和 `court_type`；禁止调用方自行猜测。

#### 验收标准

- 单打边线、双打边线、前后发球线、中线和球网位置均有单元测试。
- 标准坐标投影到矩形时，所有线段落在预期位置。
- `demo.py` 不再维护另一套球场尺寸。
- 现有完整场地图和球员落点显示不发生可见退化。
- Module 不读取视频、不创建窗口、不写文件。

#### 回滚边界

本项仅调整几何定义和调用位置。若出现回归，可以恢复旧 `demo.py` 常量，不影响检测器和流水线其他阶段。

### MIG-02：建立共享 Calibration Validation Module

状态：`implemented`  
前置条件：MIG-01  
上游来源：`badminton_analysis/court/reference.py` 中的完整线支持评分  
现有实现：`scripts/court_calibration.py::court_line_support`

#### 当前问题

现有评分只测量四条外围边线的亮度支持率。绿色区域边界、广告牌边缘或错误裁剪仍可能通过；手工角点和自动角点也没有完整共享一套质量门槛。

#### 要迁移和新增的能力

- 完整标准场地线的投影与加权支持评分。
- 每条标准线的覆盖率、支持率和汇总质量指标。
- 角点顺序、凸性、面积和越界检查。
- Homography 可逆性、矩阵条件和重投影检查。
- `court_type` 和坐标单位的显式记录。
- 结构化接受/拒绝结果及原因。

#### 修改方向

1. 所有候选 Adapter 输出统一的 Calibration Candidate。
2. 只有本 Module 可以把候选提升为 `Validated Calibration`。
3. 手工角点也必须经过同一验证，不能因为来自人工输入就绕过质量检查。
4. 原始分数、阈值、接受/拒绝原因写入校准 JSON 和 Stage Result。
5. 预览图应显示候选四边形、投影后的完整场地线以及失败原因。

#### 验收标准

- 正确矩形和透视四边形能够通过验证。
- 角点乱序、凹四边形、近奇异矩阵、场外四边形被拒绝。
- 正确内场线的分数显著高于平移、缩放或错误透视后的场地线。
- 缺少部分场地线时返回降级质量，不把缺失伪装为满分。
- 失败结果不产生可供米制指标消费的校准 Artifact。
- 手工和 Hough 白线候选使用同一组验收规则；绿色轮廓只可作为场地区域/主视角证据，不能在生产 `hybrid` 模式中晋升为正式场地四角。

#### 回滚边界

共享验证应先以旁路诊断模式输出分数，再切换为强制门控。若阈值导致真实样本全部被拒绝，只回滚阈值启用，不回滚结构化质量记录。

### MIG-03：迁移 Hough 自动球场候选 Adapter

状态：`implemented`（生产 `hybrid` 已启用；更多真实冻结样本验收待补充）  
前置条件：MIG-01、MIG-02  
上游来源：`badminton_analysis/court/detector.py`

#### 要迁移的能力

- 球场白线/高亮线掩膜。
- Hough 线段检测。
- 横向线和侧边线分类、去重。
- 直线交点和四边形候选生成。
- 候选的线段支持、形状和标准球场匹配分数。
- 用于诊断的线段、候选和评分信息。

#### 不直接迁移的内容

- “只要存在候选就接受最高分”的行为。
- `_promote_far_baseline` 作为默认规则。
- 绑定特定转播构图的固定中心位置、宽度比和底边位置阈值。
- OpenCV `imshow`、`waitKey` 和终端确认流程。
- `corners | None` 加松散 debug 字典的旧返回协议。

#### 修改方向

1. 生产 `hybrid` 只输出 Hough 白线候选；绿色轮廓 Adapter 仅保留给显式 `contour` 兼容模式和场地区域诊断。
2. 所有画面比例先转成归一化坐标，分辨率不得改变候选排序。
3. 将上游硬编码先完整登记为实验参数，并标明其样本来源，不立即放入公共默认配置。
4. Hough Adapter 只负责候选及证据，不负责宣布校准成功。
5. 候选必须进入 MIG-02 的共享验证。
6. 诊断 Artifact 应保留原始线段、候选四边形、各项分数及最终拒绝原因。

#### 验收数据

至少建立以下冻结样本：

- 标准正面转播 Main View。
- 远端边线较暗或被球员遮挡的 Main View。
- 分辨率和画幅不同但机位相近的 Main View。
- 回放、特写、观众席和计分牌负例。
- 绿色广告区域明显的负例。
- 单打线和双打线都清晰的场景。

#### 验收标准

- 正例能够生成接近人工标注的候选。
- 所有负例均不能产生 `Validated Calibration`。
- 同一帧等比缩放后，归一化角点误差保持在约定阈值内。
- 检测失败返回结构化拒绝，不抛出未处理异常。
- 不降低当前人工参考点校准的成功率。

#### 回滚边界

Hough Adapter 必须可由配置单独关闭；关闭后继续使用绿色轮廓或人工角点 Adapter，不影响下游 Artifact 契约。

#### 2026-08-29 白线/绿色边缘纠偏

真实的 2012 印度公开赛样本暴露出一个系统性错误：旧亮线掩膜使用全局灰度阈值，明亮的灰绿色场地会整体成为前景；Hough 因而把塑胶地面或绿色区域外缘当作长直线。与此同时，`hybrid` 还把绿色轮廓候选与白线候选一起送入稳定性选择，错误四边形可以凭面积和连续边缘胜出。这不是最终绘制偏移，而是 Homography 的四个源点选错。

修正规则如下：

1. 白线掩膜同时要求“处于当前帧亮度高位”和“低饱和度”，不再把绿色/灰绿色场地面当作线。
2. Hough 几何候选先保留宽候选集，再使用标准羽毛球场全部 13 条规则线的投影支持率排序；不能再仅凭四条长边的长度排名。
3. 生产 `hybrid` 禁止绿色轮廓成为正式 Homography 四角；绿色仍可用于 Main View、粗 ROI 和诊断。
4. 生产最小完整线支持阈值提高为 `0.45`。未达到白色规则线证据门槛时应拒绝标定，不能回退到绿色边缘并报告成功。
5. 回归测试必须分别锁定“明亮低饱和度场地面不进入白线掩膜”和“hybrid 不产生 green_contour 候选”。

### MIG-04：多帧校准候选与时序稳定性

状态：`implemented`  
前置条件：MIG-02、MIG-03  
上游来源：上游单帧检测能力与现有代表帧选择逻辑的组合

#### 当前问题

现有流程会从若干代表帧中选择线支持分数最高的一帧，但没有验证角点在相邻 Main View 帧中是否稳定。偶然广告边缘、运动模糊或遮挡可能成为最高分候选。

#### 修改方向

1. 只在 Main View 的稳定区间内采样校准帧。
2. 对多帧候选分别执行共享验证。
3. 在归一化图像坐标或标准球场坐标中计算角点和 Homography 稳定性。
4. 选择稳定候选簇中的代表结果，而不是全局单帧最高分。
5. 记录采样帧、有效候选数、稳定性指标和拒绝原因。
6. 镜头切换后的候选不得与切换前候选混合统计。

#### 验收标准

- 单帧异常高分不能覆盖多个稳定中等高分候选。
- 相机固定、球员移动时校准保持稳定。
- 镜头切换或缩放时明确拒绝或切分，不输出平均后的错误矩阵。
- 校准 JSON 能追溯到全部采样帧和最终选择依据。

#### 回滚边界

多帧聚合可以独立关闭并退回“最佳已验证单帧”，但不能退回“未验证最高分单帧”。

### MIG-05：视频 Artifact 导出 Module

状态：`implemented`  
前置条件：校准迁移批次稳定；Artifact 契约保持可用  
上游来源：`badminton_analysis/media/video_audio.py`

#### 要迁移的能力

- H.264 `libx264` 编码。
- `yuv420p` 像素格式。
- AAC 音频编码和可选音轨复用。
- MP4 `faststart`。
- 同输入输出路径时使用临时文件并在成功后原子替换。
- 输出文件存在且非空的最低检查。

#### 不直接迁移的内容

- 硬编码调用系统 `ffmpeg` 和 `ffprobe`。
- 音轨检测异常时返回 `True`。
- 捕获所有异常后静默降级。
- 以布尔值表达所有成功和失败。
- 在共享工具中按文件名猜测哪些临时文件不能删除。

#### 修改方向

1. 将编解码作为独立 Media Export Module，不放进演示绘制循环。
2. FFmpeg 可执行文件解析使用统一运行环境配置。
3. 输出必须登记为视频 Artifact，并记录编码器、参数、输入音轨和结果摘要。
4. 导出失败应产生失败的 Stage Result；若允许降级，必须显式记录实际编码格式和降级原因。
5. 保持原视频时间轴、帧率和音视频时长一致性。

#### 验收标准

- Chrome、Edge 和 VS Code 内置播放器可直接播放。
- 有音频源时保留音轨；无音频源时正确输出纯视频。
- `ffmpeg` 不可用、编码失败或输出为空时不会报告成功。
- 导出视频帧数、FPS、分辨率和时长符合 Artifact 契约。
- 临时文件只在最终 Artifact 验证成功后清理。

#### 回滚边界

保留现有 OpenCV 临时视频 Artifact；浏览器兼容导出作为后续派生产物，可单独失败或关闭，不破坏前序分析产物。

### MIG-06：WebUI 薄 Adapter

状态：`implemented`（第一阶段：提交、状态和 Artifact 浏览）  
前置条件：MIG-01 至 MIG-05 的核心 Interface 稳定  
上游来源：`webui/app.py`  
禁止复用：`webui/pipeline.py::run_analysis`

#### 可保留的交互

- 视频和参考帧上传。
- 自动球场候选预览。
- 手工四点校正。
- 参数选择和配置摘要。
- 运行进度、阶段状态和错误提示。
- 视频、JSON、CSV、诊断图片和日志展示。
- 中英文界面结构。

#### 必须重写的部分

- WebUI 不得直接创建 `BadmintonAnalysisSystem`。
- WebUI 不得自行拼装球场 ROI、跟踪器或输出目录。
- WebUI 不得调用旧 `CourtMapper` 完成核心校准。
- WebUI 不得自动删除历史运行目录。
- WebUI 不得根据界面选项形成另一套默认参数语义。

#### 修改方向

1. 第一阶段只实现运行配置提交、Run Manifest 状态读取和 Artifact 浏览。
2. 第二阶段接入 Calibration Candidate 预览与手工修正。
3. WebUI 输入先转换为与 CLI 相同的 Run Specification，再调用统一管线入口。
4. 所有输出从 Run Layout 和 ArtifactReport 读取。
5. 用户取消、失败、拒绝和无结果必须分别显示。
6. 对 Near-only Analysis、Dual-side Observation 和实验性指标显示明确能力标签。

#### 验收标准

- 相同 Run Specification 经 CLI 和 WebUI 产生一致的运行配置和目录布局。
- 页面刷新后可以从 Run Manifest 恢复状态。
- WebUI 进程终止不会损坏已经完成的 Artifact。
- 界面不能把失败或被拒绝的阶段显示为成功。
- 不存在 WebUI 专用的旧分析编排分支。

#### 回滚边界

WebUI 是独立 Adapter，可以整体关闭；CLI 和批处理能力必须保持完整。

### MIG-07：文档与演示资源整理

状态：`completed`（完成审计；未复制权利来源不明的大文件）  
前置条件：无

#### 可选择性保留

- WebUI 截图可用于未来界面需求参考。
- 中英文 README 的安装和使用说明可作为文档素材。
- 小型、来源明确、许可允许的样本可进入测试夹具。

#### 不进入核心仓库路径的内容

- 大型演示视频和 GIF。
- 仅用于旧可视化的字体文件。
- 没有来源、许可或测试用途说明的媒体文件。
- 将演示截图作为算法正确性的验收证据。

#### 验收标准

- 每个测试媒体都有来源、用途、预期结果和许可说明。
- 大文件不混入 Python Module。
- README 不再引导用户运行已冻结的旧分析管线。

### MIG-08：人物躯干中心与地面接触点双锚点

状态：`implemented`  
前置条件：现有 near/far 人物跟踪契约保持稳定；姿态 Module 可后续接入但不是本项的强制前置  
相关来源：上游姿态关键点能力与新管线现有人物框底边投影

#### 问题与语义

当前人物跟踪把检测框底边中心同时用于图像标记和球场投影。它接近双脚接地点，适合 Homography，但在画面中不够贴近人物主体。若直接改成躯干中心，标注视觉效果会改善，但躯干位于地面平面上方，把该像素通过地面 Homography 投影，会产生随机位和人物远近变化的系统偏差。

本项必须拆分两个语义不同的锚点：

- `body_center`：人物躯干中心，只表示图像中的人体位置，用于画面标记、姿态关联和图像域轨迹。
- `ground_contact`：人物与地面的接触位置，用于 Homography、俯视球场、距离、速度和覆盖面积等米制指标。

文档和界面不得把 `body_center` 或 `ground_contact` 称为羽毛球“落点”。统一使用“躯干中心”和“球员地面接触点”，避免与羽毛球落点混淆。

#### 锚点来源与降级顺序

`body_center`：

1. 有可信姿态关键点时，优先使用左右肩与左右髋的有效关键点中心。
2. 只有部分躯干关键点有效时，可以使用通过最低质量门槛的有效点中心，并记录实际来源。
3. 没有姿态结果时，使用人物检测框水平中心及约定的垂直比例作为近似值。
4. 检测框或姿态无效时保持 missing，不沿用上一帧位置伪装成 Observation。

`ground_contact`：

1. 左右脚踝关键点均可信时，优先使用脚踝中点，并允许按姿态模型定义增加受控的脚底补偿。
2. 只有一个脚踝可信时，允许使用单脚点，但必须降低质量并记录来源。
3. 没有可信脚踝时，回退到检测框底边中心。
4. 没有有效人物 Observation 时保持 missing；短缺口预测必须继续标记为 interpolated。

#### Schema 与 Artifact 修改方向

1. 不再让无来源说明的 `image_x/image_y` 同时承担人体中心和地面接触语义。
2. 新增或迁移为明确字段：
   - `body_image_x`、`body_image_y`
   - `ground_image_x`、`ground_image_y`
   - `court_x`、`court_y`，只能由有效 `ground_contact` 产生
   - `body_anchor_source`、`ground_anchor_source`
   - 两类锚点各自的 confidence/validity
3. 兼容旧 CSV 时，通过版本化 Adapter 把旧 `image_x/image_y` 解释为 `ground_contact`，不能静默改成 `body_center`。
4. 平滑后的躯干中心和地面接触点必须保留各自的来源、missing 和 interpolated 状态。
5. Diagnostic Demo 可以同时显示两点；Presentation Demo 默认把人物标签和图像轨迹放在 `body_center`，把俯视位置放在 `ground_contact`。

#### 明确禁止

- 不得把 `body_center` 直接通过地面 Homography 后写入正式 `court_x/court_y`。
- 不得因躯干中心存在就推断脚点有效。
- 不得用同一个 confidence 同时代表人物检测、姿态躯干和脚踝质量。
- 不得在更换锚点语义后继续复用旧 schema 版本号。
- 不得让纯展示配置改变战术统计所消费的球场坐标。

#### 验收标准

- 画面标记落在人物躯干附近，而非脚下；俯视球场位置保持基于地面接触点。
- 只切换 Presentation Demo 的锚点显示方式时，`court_x/court_y`、距离、速度和覆盖面积完全不变。
- 同一站立人物在近端与远端时，躯干中心投影不会进入正式球场坐标 Artifact。
- 姿态有效、单脚有效、仅检测框有效和完全漏检四种情况均有单元测试。
- 旧 CSV 经兼容 Adapter 后仍保持原来“框底中心”的地面语义。
- near/far 角色关联使用哪类锚点必须由配置和 Interface 明确决定，并有身份交换回归测试。

#### 回滚边界

第一批只增加双锚点字段并保持当前 `ground_contact` 算法不变；第二批再把演示标记切换到 `body_center`；第三批才允许姿态关键点替换对应锚点来源。任一后续批次回滚时，现有框底中心的球场投影和米制统计必须保持可用。

### MIG-09：姿态检测 Implementation

状态：`implemented`  
前置条件：MIG-08 双锚点契约

#### 已实现能力

- COCO-17 具名关键点、置信度、骨架边和 JSON Artifact 契约。
- `yolo_pose` 与 `rtmpose` 两个候选 Adapter，共用 near/far 角色关联和下游 Track schema。
- RTMPose 的 mode、backend、device、本地检测/姿态 ONNX 路径和输入尺寸均为类型化配置。
- RTMPose 未提供内部检测框时，只有存在可信脚踝的姿态才能生成正式地面接触点；不使用姿态外接矩形底边冒充地面点。
- Presentation Demo 和人物调试视频从 Track Artifact 绘制通过阈值的骨架。
- 姿态缺失或短缺口预测继续明确记录为 `pose_valid=0` 与 `is_interpolated=1`。

#### 回滚边界

`player_tracking.detector` 改回 `yolo` 即可禁用姿态推理，bbox 双锚点、场地投影和其他 Stage 不依赖 RTMPose 可选依赖。

## 5. 推荐执行批次

每个批次只处理一个主要风险，完成后运行完整测试并人工查看 Artifact。

### 批次 A：几何唯一事实来源

- 执行 MIG-01。
- 不改变校准成功/失败判定。
- 重点验证俯视场地图没有退化。

### 批次 B：共享校准验证

- 执行 MIG-02。
- 先旁路输出质量报告，再启用强制拒绝。
- 冻结当前已知正确和错误样本。

### 批次 C：Hough 候选 Adapter

- 执行 MIG-03。
- 默认保持关闭，仅在实验配置启用。
- 与现有绿色轮廓和人工角点进行对照。

### 批次 D：时序稳定性

- 执行 MIG-04。
- 校准只有通过多帧稳定性或明确的单帧降级规则才能输出。

### 批次 E：人物双锚点语义

- 执行 MIG-08。
- 先扩展 schema 并保持当前球场坐标不变，再调整 Presentation Demo。
- 姿态关键点作为后续来源接入，不能与字段语义调整放在同一批完成。

### 批次 F：浏览器兼容视频

- 执行 MIG-05。
- 不修改跟踪、轨迹和战术计算。

### 批次 G：统一 WebUI

- 执行 MIG-06。
- 删除或冻结旧 WebUI 编排入口之前，先验证 CLI/WebUI 结果一致。

### 批次 H：资源清理

- 执行 MIG-07。
- 与核心算法迁移分开提交。

## 6. 每批修改的固定流程

1. 在本文件中把目标项改为 `in_progress`。
2. 记录本批实际修改文件和预期不变量。
3. 先增加最小回归测试或冻结样本。
4. 迁移最小可用 Implementation，不顺手扩大到下一项。
5. 运行相关单元测试和完整测试集。
6. 对生成的 JSON、CSV、图片和视频 Artifact 做结构检查。
7. 对视觉变化保留前后对比，但不以视觉观感替代定量验收。
8. 更新本文件的测试结果、已知限制和下一批前置条件。
9. 确认可以通过配置或独立提交回滚本批修改。
10. 将状态改为 `completed`，或保持 `blocked` 并记录具体原因。

推荐验证命令：

```powershell
Set-Location badmintondataprocess
python -m pytest -q
python -m compileall src scripts
```

涉及校准时还应执行对应的冻结样本评估；涉及视频导出时还应使用 `ffprobe` 核对视频流、音频流、帧率、分辨率和时长。

## 7. 状态记录

| 编号 | 状态 | 实际提交 | 测试结果 | 备注 |
|---|---|---|---|---|
| MIG-01 | implemented | 工作区修改，未提交 | `tests/test_calibration.py` | `calibration/reference.py` 为唯一球场几何事实来源 |
| MIG-02 | implemented | 工作区修改，未提交 | `tests/test_calibration.py`、`tests/test_artifacts.py` | 所有候选统一经过 validation Seam |
| MIG-03 | implemented | 工作区修改，未提交 | 合成 Hough、白线掩膜、hybrid 来源约束与真实素材 Run | 生产 hybrid 只允许 Hough 白线四角；更多真实正负冻结样本待补 |
| MIG-04 | implemented | 工作区修改，未提交 | 稳定簇与离群候选测试 | 校准 JSON 记录全部采样帧和选择依据 |
| MIG-05 | implemented | 工作区修改，未提交 | `tests/test_media_export.py` | 使用 `imageio-ffmpeg` 解析可执行文件；最终仍由视频 Artifact 检查兜底 |
| MIG-06 | implemented-phase-1 | 工作区修改，未提交 | `tests/test_webui_adapter.py` | 已统一运行 Interface；候选点图形化拖拽属于第二阶段交互增强 |
| MIG-07 | completed | 工作区修改，未提交 | 文档审计 | 见 `docs/upstream_resource_inventory.md` |
| MIG-08 | implemented | 工作区修改，未提交 | `tests/test_player_anchors.py`、`tests/test_demo_rendering.py` | 默认 `yolo` 仍使用 bbox；启用姿态 Adapter 后优先消费肩、髋和脚踝 |
| MIG-09 | implemented | 工作区修改，未提交 | `tests/test_pose.py`、`tests/test_demo_rendering.py` | RTMPose balanced 完整 rally：near 99.4%、far 94.4% 姿态有效率 |
| MIG-10 | implemented | 工作区修改，未提交 | `tests/test_full_analysis.py`、真实 `material` 素材 Run | 自动清洗、标定、RTMPose、TrackNet、统计与最终视频已统一为 `bdp analyze` / PowerShell 一键入口 |

### 8.1 实际迁移结构

- `calibration/reference.py`：标准场地几何 Module。
- `calibration/validation.py`：候选晋升为 Validated Calibration 的唯一 Seam。
- `calibration/hough.py`：只生成候选与证据的实验 Adapter。
- `tracking/player/anchors.py`：姿态、单脚、检测框和旧 CSV 的双锚点 Adapter。
- `media/export.py`：浏览器兼容导出 Implementation。
- `webui/adapter.py`：CLI/WebUI 共用运行规范的薄 Adapter。
- `webui/app.py`：可选 Gradio 表现层，不包含分析编排。
- `tracking/player/pose.py`：YOLO Pose 与 RTMPose 共享姿态契约及两个 Implementation。

### 8.2 尚不能伪装为已完成的验收

- 仓库目前没有许可明确的真实冻结视频集，因此 Hough 在标准转播、遮挡、回放、观众席、绿色广告等正负样本上的指标尚未形成基线。
- 默认 `detector: yolo` 不运行姿态模型；需要骨架时必须显式选择 `rtmpose` 或 `yolo_pose`，以便 Run Manifest 准确记录性能和能力变化。
- 浏览器兼容导出已有命令契约和 Artifact 检查；Chrome、Edge、VS Code 播放器以及有声真实样本的人工验收仍需在具备授权媒体后执行。
- WebUI 第一阶段已覆盖运行提交、Manifest 状态和 Artifact 浏览；自动候选预览与四点拖拽校正尚未进入 UI，现阶段继续通过统一配置中的 `court_calibration.reference_points` 提交。

### 8.3 本批验证记录

- `python -m pytest -q`：`122 passed`。
- `python -m compileall -q src scripts`：通过。
- `git diff --check`：无空白错误；仅报告工作区既有的 Windows CRLF 转换提示。
- 使用本地 61 帧、852×480、30 FPS 样本执行真实 Media Export；FFmpeg 解码复核为 H.264 High、`yuv420p`、30 FPS，输出非空。
- 标准球场宽度、长度和球网位置经代码搜索只在 `calibration/reference.py` 定义。
- RTMPose balanced 在 492 帧全英样本上输出 978 条人物轨迹：near 489/492 帧姿态有效，far 459/486 行姿态有效；平均有效关键点分别为 16.88 和 16.74。
- RTMPose CPU/ONNXRuntime 人物 Stage 耗时 620.656 秒，说明 balanced 适合离线质量验证，不适合作为实时默认值；快速迭代应选择 lightweight 或保持普通 YOLO。
- RTMPose 样本 Run 的 9 个 Stage 全部成功，最终 H.264 演示视频已人工核对 near/far 骨架、双锚点和完整俯视场地。
- 从 `stop-after=tracking` 续跑时发现并修复 `player_cfg` 的分支局部变量问题，新增恢复回归测试。
- RTMPose 验证配置曾为缩短运行时间错误地把羽毛球模型降为 `motion_bright_baseline`，导致可见帧从 TrackNet 的 471/492 降至 322/492；现已恢复 TrackNet，并用配置组合测试锁定“姿态后端不得改变羽毛球后端”。
- GPU 未启用的根因不是硬件或驱动：原运行解释器安装的是 `torch 2.12.0+cpu` 和 CPU 版 `onnxruntime`，因此 TrackNet 与 RTMPose 的 `auto` 均选择 CPU。本机 RTX 5070、驱动 581.57 正常。
- Conda `base` 不是有效环境；复用专用 `good-badminton` 环境（Python 3.12.13、PyTorch 2.11.0+cu128），补齐 `onnxruntime-gpu 1.29.0` 与 RTMLib。PyTorch 实测识别计算能力 12.0，ONNX Runtime 实测暴露 CUDA/TensorRT provider。
- TrackNet CUDA 冒烟测试确认模型参数位于 `cuda:0`、峰值显存约 412 MB；492 帧 TrackNet Stage 耗时 24.544 秒并恢复到可见 471 帧、插值 10 帧，最终 H.264 演示产物成功。验证配置现对 RTMPose 与 TrackNet 显式要求 `cuda`，代码在 GPU provider 缺失时直接报错，不再静默回退 CPU。
- 全新 GPU Run `rtmpose_tracknet_gpu_allengland_r001` 从头完成 9 个 Stage：RTMPose 人物 Stage 从 CPU 的 603.112 秒降至 40.967 秒，TrackNet 为 24.643 秒；输出 978 条人物轨迹（948 条有效姿态）和 492 条羽毛球轨迹（471 条可见、10 条插值），最终 H.264 演示视频成功。
- 一键完整视频 Run `full_material_india2012_r001` 直接读取 `F:\Good-Badminton\material` 的 315 秒未清洗素材：识别 19 段 Main View，接受 4 个 Usable Rally、拒绝 27 个候选。白线纠偏后 4/4 标定均由 `hough_lines` 产生，完整线支持分数依次为 0.820、0.914、0.817、0.827；输出 700 条人物轨迹（698 条有效姿态）和 350 条羽毛球轨迹（343 条可见、1 条插值）。最终合并视频 350 帧/11.67 秒，经抽帧确认黄色外框贴合最外侧白色双打边线，骨骼、双端角色和完整俯视场地同步使用修正后的 Homography。旧标定下只有 538 条人物轨迹，说明错误绿色边缘也曾造成球员候选误裁剪。
- 回合召回修正 Run `full_material_india2012_rally_recall_r002` 使用相同 315 秒素材：活动评分从全画面均值迁移为归一化球场主体区域局部帧差，生产阈值校准为 0.008、最少活动样本为 2、上下文恢复为 2.2/1.4 秒；同时修复 30.000033 FPS 下恰好 60 帧被浮点误判为不足 2 秒的问题。结果保持 19 段 Main View，接受 20 个 Usable Rally、仅拒绝 4 个子候选，20/20 标定、RTMPose 和 TrackNet 成功；输出 9620 条人物轨迹（9343 条有效姿态）和 5011 条羽毛球轨迹（4412 条可见、125 条插值）。最终 H.264 视频为 5011 帧/167.03 秒，首、中、末抽帧均确认是比赛主视角且白线、骨骼、羽毛球轨迹和完整俯视图存在。独立新包复跑 `full_material_india2012_rally_recall_r003` 再次得到 20 接受/4 拒绝，确认正式管线已使用 `rally/activity.py`，不是依赖旧脚本偶然生效。
- 羽毛球越界跳线修正：生产配置关闭 TrackNet 缺失帧盲目外推；轨迹平滑新增 80 像素二维端点位移上限，演示和战术阶段不再回退消费被拒绝的原始插值点。对上述 5011 帧真实样本重做平滑后，补点由 155 行降至 80 行，保留补线最大端点位移 79.19 像素，超过阈值的补线为 0；修正版 20 回合演示视频和关键区间六帧检查图均已生成。
- 一键入口新增 `-ValidateOnly`：使用真实 `material` 视频完成视频解码、生产配置、RTMPose CUDA、TrackNet CUDA 与权重预检，结果成功且不创建 Run；`analysis_summary.json` 同时记录 TrackNet 原始插值行数、平滑有效行数和获准短缺口补点数。

## 8. 完成定义

只有同时满足以下条件，迁移计划才算完成：

- 上游可复用算法已经通过新管线 Interface 使用，不再依赖旧分析编排。
- 自动候选不能绕过共享校准验证。
- 标准球场几何只有一个事实来源。
- 躯干中心与球员地面接触点具有不同字段、来源和有效性，正式球场坐标只来自地面接触点。
- 校准和视频导出失败会准确进入 Stage Result 和 Run Manifest。
- 所有输出均位于 Run Layout 并登记为 Artifact。
- CLI、批处理和 WebUI 不存在语义不同的平行管线。
- 旧管线仍保持冻结，不因迁移重新获得新的算法职责。
- 全部测试通过，冻结样本没有未解释的质量退化。
