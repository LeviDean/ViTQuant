# ViTQuant 实验设计

本文档说明 ViTQuant 项目的**实验目标**（要回答什么问题）和**实验设置**（怎样设计实验来回答）。
项目本身做了什么、每个模块的实现细节，见 [PROJECT_REPORT.md](PROJECT_REPORT.md)；如何运行，见
[README.md](README.md)。本文档聚焦"为什么这样设计实验"。

## 1. 实验目标

项目的实际部署目标是**边缘 NPU**，但目标 NPU 型号、工具链、kernel 尚未确定。在这个前提下，
"真实 int8 延迟""真实磁盘压缩"这类数字在 CPU/GPU 上测出来对最终 NPU 部署没有参考价值——它们
强依赖具体硬件的 runtime 和 kernel 实现，换一个 NPU 就完全作废。

因此实验目标被限定为只回答**与具体硬件无关、可以现在就做、且未来换任何 NPU 都能复用**的问题：

1. **精度成本**：给定一个量化方案（位宽、粒度、对称性），相对 fp32 基线精度损失多少？能不能接受？
2. **敏感度定位**：模型的哪些部分（哪个 block、attention 里的哪个矩阵乘）对量化最敏感？如果未来
   NPU 需要混合精度（部分层高精度、部分层低精度），应该优先保留哪些层？
3. **方案选择依据**：per-tensor vs per-channel、对称 vs 非对称、不同 Observer（MinMax / 滑动平均 /
   百分位）之间，精度差异有多大？哪种更稳健？
4. **位宽下探的可行性**：很多边缘 NPU 偏好 4-bit 权重（W4A8），从 8-bit 降到 4-bit 精度会掉多少？
5. **理论压缩率**：不涉及具体 runtime，纯算术上（位宽 → 字节数）能拿到多少压缩比，作为立项前的
   预估参考。
6. （SAM）**分割任务的量化敏感性**：分类任务之外，量化对分割模型（结构和 attention 模式都不同）
   的影响是否一致，还是分割任务对量化更敏感/更不敏感？

明确不追求的（超出当前实验范围，非疏漏）：真实 int8 kernel 延迟、真实内存占用、真实设备上的
吞吐量——这些需要拿到具体 NPU 之后才有意义，见 [PROJECT_REPORT.md §8 已知限制](PROJECT_REPORT.md)。

## 2. 自变量：量化方案的可控维度

所有实验共享同一个 `QConfig`（[vitquant/quant/qconfig.py](vitquant/quant/qconfig.py)）参数空间，
每次实验固定其余维度、只变动一个维度，这样才能把精度变化归因到具体的设计选择上：

| 维度 | 可选值 | 默认（W8A8 基线） |
|---|---|---|
| 权重位宽 | 任意整数（已验证 8 / 4） | 8 |
| 激活位宽 | 任意整数 | 8 |
| 权重粒度 | per-tensor / per-channel | per-channel |
| 激活粒度 | per-tensor / per-channel | per-tensor |
| 权重对称性 | 对称 / 非对称 | 对称 |
| 激活对称性 | 对称 / 非对称 | 非对称 |
| Observer | MinMax / MovingAvg（动量 0.1） / Percentile | 权重 MinMax，激活 MovingAvg |

权重默认对称 + per-channel、激活默认非对称 + per-tensor + 滑动平均，是 ViT 量化文献里常见的
稳健起点（权重分布对称、激活分布常有偏移和离群值），实验里把它当作**基线方案**，其余方案都相对
它做消融对比。

## 3. 实验对象（模型 / 任务）

| 模型 | 任务 | 用途 | 配置 |
|---|---|---|---|
| `deit_tiny_patch16_224` | 图像分类 | 开发机快速迭代（小模型，跑得快） | [configs/deit_tiny.yaml](configs/deit_tiny.yaml) |
| `vit_base_patch16_224` | 图像分类 | 服务器完整评估（生产规格模型） | [configs/vit_base.yaml](configs/vit_base.yaml) |
| `deit_tiny_patch16_224` (W4A8) | 图像分类 | 4-bit 权重精度代价专项实验 | [configs/deit_tiny_w4a8.yaml](configs/deit_tiny_w4a8.yaml) |
| `facebook/sam-vit-base` | 分割（仅 vision encoder） | 分类之外的第二个任务类型，验证结论是否可迁移 | [configs/sam_vit_b.yaml](configs/sam_vit_b.yaml) |

选两个不同规格的分类模型（tiny 用于开发迭代、base 用于正式结果）+ 一个结构不同的分割模型
（SAM 的 ViT backbone 有 windowed attention，和标准 ViT 不同），是为了让敏感度/消融结论不只
建立在单一模型规格上。

分类模型量化范围：所有 `nn.Linear`（含 attention 的 qkv/proj 和 MLP）、`nn.Conv2d`（patch
embedding）、attention 内部两次矩阵乘（q@k^T、attn@v）；分类头 `head` 默认跳过。SAM 只量化
`vision_encoder`（ViT backbone），`prompt_encoder`/`mask_decoder` 保持 fp32——因为这两部分
参数量小、且分割质量对它们的精度更敏感，量化收益低而风险高，属于实验的既定范围（scope），不是
待办。

## 4. 实验流程与每类实验的设置

一次 `run_all.py`（分类）或 `quantize_sam.py`（SAM）按顺序做 5 类实验，全部在同一份权重和同一批
校准/评估样本上进行，保证组间可比：

### 4.1 fp32 基线

不做任何量化改动的原始 PyTorch 模型直接评估，作为所有对比的分母。分类任务用 Imagenette 验证集
全量（`configs/*.yaml` 中 `eval.max_batches: null`），SAM 用固定种子（`seed=1`）采样的图像+点
提示集合。

### 4.2 模拟 INT8 精度对比（主实验）

用基线 `QConfig` 转换模型（`convert_vit` / `convert_sam_vision_encoder`），跑一遍 `calibrate()`：
先做**数据无关的权重标定**（权重 qparams 只取决于权重张量，直接算一次冻结），再喂校准集只收集
**激活**统计量，冻结后切换到量化模式；然后在与 fp32 完全相同的评估集上评估。两者唯一的区别变量
是"是否量化"，其余（数据、设备、随机种子）保持一致。

- 分类：校准集 256 张图（`calib_images`，来自训练集，`calib_batch_size=32`），与验证集不重叠。
- SAM：校准集和评估集分别用 `seed=0` / `seed=1` 采样，默认各 16 个样本（正式实验建议提高到
  64–128，`sam_vit_b.yaml` 里已注明）。

### 4.3 逐块敏感度分析（`block_sensitivity`）

**控制变量法**：每次只让模型的一个 block（`patch_embed`、`blocks.0` … `blocks.N`）量化，其余
全部保持 fp32，测这一个 block 单独量化造成的 top-1 下降。一共做 `len(blocks)+1` 次完整评估
（每个 block 一次 + patch_embed 一次），全部用同一份基线 `QConfig` 和同一个校准/评估集，只变动
"哪个 block 被量化"这一个维度。结果按精度下降幅度从高到低排序，直接回答"如果未来 NPU 要混合
精度，应该优先给哪些层保留高精度"。

`try/finally` 保证即使扫描中途抛异常，模型也会被强制恢复为全量化状态，不会因为异常残留半量化
的脏状态污染后续实验。

### 4.4 量化方案消融（Ablation Matrix）

固定模型和数据不变，把 §2 中的某一个维度从基线值改成备选值，其余维度不变，逐一评估：

| 消融变体 | 相对基线改动的维度 |
|---|---|
| `weight per-tensor` | 权重粒度：per-channel → per-tensor |
| `activation symmetric` | 激活对称性：非对称 → 对称 |
| `activation minmax obs` | 激活 Observer：滑动平均 → MinMax |
| `activation percentile obs` | 激活 Observer：滑动平均 → Percentile |

每个变体都从**同一份原始 fp32 checkpoint 重新加载**（而不是复用上一步转换过的模型），避免
`convert_vit` 原地修改权重导致的变体间相互污染；且都用与主实验相同的校准集重新校准。这样每一行
结果只反映"改了这一个维度"的边际影响，可以直接对比出哪个设计选择最敏感。

### 4.5 W4A8 专项实验（`deit_tiny_w4a8.yaml`）

与主实验（W8A8）完全相同的流程，唯一变量是权重位宽从 8 改成 4（激活仍为 8-bit）。因为自研
fake-quant 内核对位宽是泛化的（`bits` 是普通参数，不是硬编码），这个实验是纯配置改动，不需要
改代码——这也顺带验证了框架本身对任意位宽的支持是可行的。

### 4.6 定性可视化（`qualitative.py` / `qualitative_sam.py`）

在与主实验相同的 fp32/量化模型对上，额外抽取一批真实样本（分类默认 30 张、SAM 默认 8 张）做
**逐样本对比**，作为定量指标之外的抽样核验：

- 分类：fp32 与量化模型的 top-1 预测逐图对比，预测发生翻转（fp32 对、量化错，或反之）的样本
  排在网格最前面并标红标题，用于人工抽查"精度指标上看起来影响很小，但具体错在哪些图/哪类样本"。
- SAM：fp32（青绿色实线）与量化（品红色虚线）预测的 mask 轮廓叠加画在原图上（含点提示位置），
  按 IoU 从低到高排序、低于阈值（0.9）的标红，用于人工核验"自一致性 IoU 掉得最多的样本，掉的
  是哪种类型的分割错误"。

这一步不产生新的量化数字，是对 4.2 定量结果的可解释性补充——数字说"平均掉了多少"，图看"具体
错在哪"。

## 5. 数据与评估协议

- **分类数据集**：Imagenette（ImageNet 的 10 类子集，约 100MB），可自动下载或离线预置。
  `IMAGENETTE_TO_IMAGENET1K` 把预训练模型的 1000 类 logits 映射到这 10 个类别索引上再算
  top-1/top-5，因此不需要在下游数据集上微调，直接复用 ImageNet 预训练权重评估。
- **SAM 数据**：与分类共享同一套 Imagenette 图像源，为每张图生成一个点提示（point prompt），
  校准集/评估集用不同随机种子采样，保证两者不重叠。
- **评估指标**：
  - 分类：top-1 / top-5 命中率（`AccuracyMeter`，自动兼容类别数小于 5 的情况）。
  - SAM：自一致性 IoU（同一输入下 fp32 mask 与量化 mask 的交并比），**不是**对标 COCO 等标注
    集的真值 mIoU——SAM 评估的问题本身是"量化改变了多少预测"，不是"分割准不准"，这是刻意的
    范围选择（见 [PROJECT_REPORT.md §4](PROJECT_REPORT.md)）。
- **理论压缩率**：`32 / weight_bits` 的纯算术值（int8 = 4x，int4 = 8x），不依赖任何 runtime，
  与实测精度并列在同一份报告里，作为"这个方案理论上能省多少"的参考。

## 6. 输出产物

每次实验（`run_all.py` 或 `quantize_sam.py`）落盘到 `outputs/<model>/`：

- `results.json`：结构化数据（各变体的 top-1/top-5 或 IoU、位宽、设备信息），供后续脚本或
  报表二次处理。
- `report.md`：人类可读报告，含精度对比表、理论压缩表、敏感度排名表、消融矩阵表（SAM 为
  IoU 表 + 理论压缩表）。
- `qualitative_grid.png` / `qualitative_sam_grid.png`：§4.6 的可视化网格，附带
  `qualitative.json`/`.md`（每张图的预测详情，供程序化二次分析）。

一次命令产出报告 + 可视化，避免"数字看着没问题，但没人去看具体样本图"的情况。

## 7. 实验设置的边界（What this is not）

- 所有精度数字来自**模拟量化**（fake-quant：数值上量化-反量化后仍用 fp32 计算），因此精度评估
  可信，但不产生任何真实加速或真实内存/磁盘变化——这是当前 NPU 型号未定阶段的设计选择。
- 不测真实 int8 kernel 延迟、真实吞吐量、真实设备内存占用；这些需要具体 NPU 及其工具链，超出
  当前实验范围。
- 敏感度分析是**逐 block** 粒度，不是逐层（within-block）粒度；消融矩阵覆盖的是几个代表性
  变体，不是量化方案空间的穷举网格搜索。
- SAM 仅覆盖 vision encoder 一种量化范围（`prompt_encoder`/`mask_decoder` 恒为 fp32），且
  评估协议是自一致性而非真值 mIoU。

这些边界都是当前"NPU 预研"阶段的既定范围，一旦确定具体 NPU 型号，下一步应该是：把 `QConfig`
换成该 NPU 实际支持的方案（很可能是 per-tensor + 对称 + 特定位宽组合），重跑本文档描述的全部
实验流程，拿到贴合目标硬件的精度参考。
