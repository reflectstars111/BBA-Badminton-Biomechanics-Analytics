# BBA 的 BST 击球分类后端

状态：适配器、正式运行依赖和 CUDA 严格加载已验证；分类精度仍需冻结人工标注集验收。

## 1. 上游来源与边界

BBA 的可选击球分类后端来自 [Va6lue/BST-Badminton-Stroke-type-Transformer](https://github.com/Va6lue/BST-Badminton-Stroke-type-Transformer)（MIT License）。作者把 ShuttleSet 权重发布在[官方 Google Drive 文件夹](https://drive.google.com/drive/folders/1D4172WZDJWPvpJdpaHDhy_cA-s8F-zR5?usp=sharing)。

BST 代码与权重不复制进 BBA 自有源码包：上游代码保留自己的许可证，模型文件保存在被 Git 忽略的 `weights/bst/`。没有配置权重时，二维关节角、动作候选、挥拍阶段、稳定性和步法仍照常运行，动作类别明确显示为“未分类”。

## 2. 正式运行环境

所有依赖安装到已有的 `good-badminton` Conda 环境，不创建第二个环境：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\setup_runtime.ps1
```

脚本会一次性同步 BBA 完整运行套件，并严格验证 CUDA PyTorch、ONNX Runtime CUDA Provider、RTMPose、WebUI、统计评估和 BST 所需依赖。RTMLib 的包元数据只声明 CPU 版 `onnxruntime`，因此脚本会显式保留 `onnxruntime-gpu`，并以 `--no-deps` 安装 RTMLib，避免 CPU/GPU Runtime 混装。

## 3. 上游代码与权重的自动准备

`setup_runtime.ps1` 默认完成以下工作：

- 克隆官方仓库并固定到提交 `fb9b310bf4c8a8e3d89c75e61bc06a7ac3de62df`；
- 下载已核验的官方集成权重；
- 校验权重 SHA-256：`015F7010526BCC231ECD9006366078943DBD53C0DA8E6D2424B25F0B7A70A502`；
- 实例化模型并在 CUDA 上执行严格加载检查。

已有第三方目录不会被覆盖。如果版本与固定提交不一致，安装会明确失败，要求人工处理，而不是静默切换上游代码。

需要手工修复源码时，在项目根目录执行：

在项目根目录执行：

```cmd
git clone --depth 1 https://github.com/Va6lue/BST-Badminton-Stroke-type-Transformer.git third_party\BST-Badminton-Stroke-type-Transformer
```

不要删除上游仓库内的 LICENSE。

## 4. 已核验权重与推荐策略

2026-09-02 实际检查官方公开文件夹时，能够直接列出并下载的 25% 训练权重包括：

```text
bst_AP_JnB_bone_train_partial_0p25_merged_2.pt
```

文件 ID：`1tcx78bwCO6ZBasw1PHgfyT6BYmzCIqSS`。官方结果表中该档位的 Top-1 为 0.714、macro-F1 为 0.653、Top-2 为 0.872。它已经通过 BBA 的 CUDA 严格加载和真实产物推理冒烟验证，适合验证集成是否正确，但不能作为最终生产精度已经达标的证据。

官方结果表还记录了全量训练 `BST_AP / JnB_bone / serial_3` 的更高指标（Top-1 0.830、macro-F1 0.814、Top-2 0.952），但本次公开文件夹枚举没有发现对应文件。因此在取得并核对该 checkpoint 之前，文档不再假定它可以直接下载。

当前已验证权重放置路径：

```text
weights/bst/bst_AP_JnB_bone_train_partial_0p25_merged_2.pt
```

## 5. 启用配置

```yaml
biomechanics_analysis:
  classification_backend: bst
  bst_repository: ../third_party/BST-Badminton-Stroke-type-Transformer
  bst_weights: ../weights/bst/bst_AP_JnB_bone_train_partial_0p25_merged_2.pt
  bst_device: cuda
  bst_model_name: BST_AP
  bst_pose_style: JnB_bone
  bst_seq_len: 30
  bst_num_classes: 25
  bst_min_confidence: 0.45
```

模型结构、姿态模态、序列长度或类别数与权重不匹配时，严格加载会返回 `bst_checkpoint_incompatible`，不会把错误 checkpoint 当成有效分类器继续运行。

## 6. 输入契约

- 两名球员按 `far -> Top`、`near -> Bottom` 放入独立轴，输出再拆回 `player_id + stroke_class`；
- COCO-17 关键点相对人物框中心并按人物框对角线归一化；
- `JnB_bone` 使用上游 COCO 骨连接；
- 脚底球场位置按 `6.10 m × 13.40 m` 归一化；
- 羽毛球坐标按视频宽高归一化，插值点不进入 BST 输入；
- 缺失数据使用零掩码，不伪造关键点、球位置或对侧球员。

## 7. 当前验证结果

- CUDA 设备：NVIDIA GeForce RTX 5070 Laptop GPU；
- 严格加载模型 ID：`BST/BST_AP/JnB_bone/seq30/25/015f7010526b`；
- 俯视样本：22 个动作事件，14 个满足分类输入门槛；
- 低视角样本：785 个动作事件，32 个满足分类输入门槛；
- 以上只验证“依赖、张量契约、权重和推理链路能正确运行”，不等同于分类准确率验收。

正式发布分类结果前，必须用独立冻结的人工作业集报告 macro-F1、Top-1、Top-2、unknown/reject 比例、击球者一致率，并分别评估俯视和低视角。
