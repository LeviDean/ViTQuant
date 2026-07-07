# ViTQuant 项目实施报告

本文档详细说明 ViTQuant 项目具体做了什么：架构设计、每个模块的实现内容，以及实测得到的结果。

## 1. 项目目标与当前定位

项目的实际部署目标是**边缘 NPU**，不是 CPU。因此项目当前阶段是**纯模拟量化研究**，
不追求任何真实部署产物——CPU 上的"真实 int8 加速/体积压缩"对边缘 NPU 场景没有意义，
真实延迟只能在目标 NPU 的真实工具链和 kernel 上测得，在这之前无法有意义地模拟。

框架现在是**单层架构**：

| 层 | 目录 | 作用 |
|---|---|---|
| 研究层 | `vitquant/quant/` | 自研 fake-quant 内核，模拟量化，只测精度不测速度 |

> 备注：项目早期曾有一个交付层（`vitquant/deploy/`：ONNX 导出 + ONNX Runtime 真实
> INT8 量化 + CPU 延迟基准），在转向 NPU 预研后已整体删除；相关实现和踩坑记录仍可在
> git 历史中找到（`vitquant/deploy` 目录删除前的版本），此处不再展开。

研究层保留并可迁移到任意未来目标设备的产出是：

- **精度保持率**：某个量化方案（位宽/粒度/对称性）相对 fp32 基线的精度损失
- **逐层敏感度**：哪些 block 对量化最敏感，为未来在真实硬件上决定"哪些层保留高精度"提供依据
- **方案消融**：不同 Observer、对称性、粒度的精度对比，与具体 runtime 无关
- **理论压缩比**：纯粹从位宽算出的算术比值（如 int8 权重是 fp32 的 1/4），与目标硬件无关

`QConfig` 保持可插拔正是为了这个目的：一旦确定了具体的 NPU 型号，可以直接把量化方案
（逐张量/逐通道、对称/非对称、位宽）改成该 NPU 要求的方案，在接触厂商工具链之前先拿到
一个精度参考。`deit_tiny_w4a8.yaml` 这个 W4A8 配置的意义也在这里——很多边缘 NPU 偏好
4-bit 权重，提前测出降到 4-bit 的精度代价是有价值的预研工作。

## 2. 研究层：自研量化内核

### 2.1 量化配置（`vitquant/quant/qconfig.py`）

`TensorQConfig`（frozen dataclass）描述单一张量（权重或激活）的量化方案：位宽（`bits`）、是否对称（`symmetric`）、是否逐通道（`per_channel`）、通道轴（`ch_axis`）、Observer 类型（`observer`）、百分位阈值（`percentile`）。

`QConfig` 组合权重和激活两套 `TensorQConfig`，默认方案：权重逐通道对称 INT8，激活逐张量非对称 INT8（moving-average observer）。`qconfig_from_dict()` 支持从 YAML 加载的 dict 直接构造，位宽等参数完全独立可配置——这也是能直接支持 W4A8 等任意位宽组合的基础，不需要改代码，改配置文件即可。

### 2.2 Observer（`vitquant/quant/observers.py`）

三种统计量收集器，都继承自 `ObserverBase`（`nn.Module`），支持逐张量和逐通道两种归约方式：

- `MinMaxObserver`：跟踪全局最小/最大值
- `MovingAvgMinMaxObserver`：指数滑动平均（动量 0.1）更新最小/最大值
- `PercentileObserver`：用分位数裁剪离群值，仅支持逐张量（超过 100 万元素会自动子采样）

`compute_qparams()` 根据统计量计算 scale 和 zero_point：对称量化用受限范围（如 8-bit 为 -127~127，保证 zero_point 恰好为 0）；非对称量化用完整范围（-128~127），并强制数值范围包含 0（保证真实的 0 能被精确表示）。未校准就调用会抛出 `CalibrationError`。

### 2.3 FakeQuantize（`vitquant/quant/fake_quant.py`）

核心的量化-反量化原语：

```
q = clamp(round(x / scale) + zero_point, qmin, qmax)
dq = (q - zero_point) * scale
```

反向传播用直通估计器（Straight-Through Estimator，STE）：前向输出 `dq`，反向梯度按恒等函数传递（`x + (dq - x).detach()`），这样量化点不可导的问题被绕过，模型仍可训练/校准。

`FakeQuantize` 模块有三种状态：`observing`（收集统计量，直通不改变数值）、`quantizing`（应用量化）、都不开（恒等映射）。`freeze()` 从 Observer 计算出 scale/zero_point 并切换到量化模式。这就是"模拟量化"的含义——数值上经过量化再反量化回 fp32，用 fp32 继续计算，所以能精确测精度损失，但不会有任何真实加速，也不产生任何真实的磁盘体积变化。

### 2.4 量化模块（`vitquant/quant/modules.py`）

把标准 PyTorch 模块替换成带 FakeQuantize 的版本，全部通过 `from_float()` 类方法构造，与原模块共享参数存储（不复制权重）：

- `QuantLinear`：权重和输入激活各接一个 FakeQuantize
- `QuantConv2d`：用于 ViT 的 patch embedding 卷积层，同样支持 `padding_mode` 等全部原始参数
- `QuantMatMul`：给 attention 里的两次矩阵乘（q@k^T 和 attn@v）分别接 FakeQuantize
- `QuantAttention`：把 timm 的融合 attention（`F.scaled_dot_product_attention`）拆成显式的 `q@k^T → softmax → @v` 三步，这样中间的矩阵乘才能挂上量化钩子；LayerNorm 和 Softmax 保持 fp32（ViT 量化中公认的敏感点，不量化）

关键约束：所有模块在 `freeze()`（校准完成）之前必须和原始 fp32 模块数值完全一致，已用 `atol=1e-4~1e-6` 验证过。

### 2.5 模型转换（`vitquant/quant/convert.py`）

`convert_vit()` 递归遍历 timm ViT 模型树，把 `Attention → QuantAttention`、`nn.Linear → QuantLinear`、`nn.Conv2d → QuantConv2d`，分类头（`head`）默认跳过不量化。转换后模块命名规则固定（如 `blocks.0.attn.qkv.input_fq`），供后续按名称分组做敏感度分析用。

### 2.6 校准（`vitquant/quant/calibrate.py`）

`calibrate()` 驱动完整的 观察 → 冻结 → 量化 生命周期，但把**权重标定**和**激活标定**解耦：

1. **权重标定（数据无关）**：`calibrate_weights()` 先对每个权重量化器直接从权重张量观测一次、算出 qparams 并冻结——权重是静态的，其 scale/zero_point 只取决于权重本身，不需要任何校准数据。
2. **激活标定（数据相关）**：再把校准集喂给模型，此时只有尚未冻结的激活量化器收集统计量（`set_observing` 会跳过已冻结的权重量化器）。前向仍以 fp32 权重计算，与标准 PTQ 一致，激活范围与"权重每 batch 重复观测"的旧写法逐位相同。
3. **切换量化**：冻结激活 qparams，最后 `set_quantizing()` 把所有 FakeQuantize（权重+激活）切到量化模式。

`freeze()` 只负责"从 observer 算出 qparams"，不再顺带打开量化开关，"是否应用量化"由 `set_quantizing()` 独立控制——这样 `block_sensitivity` 才能在校准完成后逐块开关量化，也为后续研究（逐权重张量的 MSE 最优 scale 搜索、权重/激活各自独立的标定策略等）留出干净的接口。

## 3. 评估体系（`vitquant/eval/`）

### 3.1 指标（`metrics.py`）

`topk_correct()` 计算 top-k 命中数（自动兼容类别数小于 k 的情况，比如 Imagenette 只有 10 类却要算 top-5）；`AccuracyMeter` 跨 batch 累计。

### 3.2 评估循环（`evaluate.py`）

- `evaluate_torch()`：PyTorch 模型评估，把 1000 类 logits 切到 Imagenette 对应的 10 个类别索引上再算 top-1/top-5
- `block_sensitivity()`：逐块敏感度分析——先测全 fp32 基线，然后每次只让一个 block（`patch_embed`、`blocks.0`……`blocks.11`）量化、其余保持 fp32，测量精度下降幅度，按下降程度从高到低排序。用 `try/finally` 保证扫描过程中即使抛异常，模型也会被恢复成完全量化状态，不会留下半量化的脏状态。

两个函数都支持可选的 `progress` 回调（每完成约 20% 的 batch 打一行状态），`block_sensitivity` 额外支持 `log` 回调汇报当前扫到第几个 block——这是服务器上跑长时间任务时避免"看起来像卡住"而加的。

## 4. SAM：vision-encoder-only 量化

`vitquant/models/sam_loader.py`、`vitquant/quant/sam_convert.py`、`vitquant/quant/sam_calibrate.py`、`vitquant/eval/sam_evaluate.py` 组成一套与分类 ViT 并行的 SAM 研究流水线：只把 `SamModel.vision_encoder`（ViT backbone）转换成 FakeQuantize 版本，`prompt_encoder`/`mask_decoder` 保持 fp32。评估指标是**自一致性 IoU**——同一张图 + 同一个点提示下，fp32 模型和模拟量化模型预测的 mask 有多相似，而不是对标 COCO 之类的真值 mIoU（当前阶段的既定范围决策，非缺陷）。

## 5. 配置与脚本

### 5.1 配置文件（`configs/`）

- `deit_tiny.yaml`：开发机用的小模型配置
- `vit_base.yaml`：服务器用的完整模型配置
- `sam_vit_b.yaml`：SAM vision-encoder 量化配置
- `deit_tiny_w4a8.yaml`：W4A8 量化方案示例，用于研究 4-bit 权重的精度代价（面向偏好 4-bit 权重的边缘 NPU）

配置结构覆盖模型（架构名 + 本地权重路径）、数据（Imagenette 路径、批大小、是否自动下载）、量化方案、评估（`max_batches` 控制冒烟测试还是全量评估）、设备、输出目录。不再有 `onnx`/`benchmark` 相关字段。

### 5.2 脚本（`scripts/`）

- `quantize.py`：单次模拟量化实验（转换 → 校准 → 评估），输出 `quantize_result.json`
- `evaluate.py`：仅评估 fp32 基线
- `qualitative.py`：fp32 vs 模拟量化的逐样本预测对比，标出被量化"翻转"的 top-1 预测，输出 `qualitative.json`/`.md` 和标注过的样本图 `qualitative_grid.png`
- `run_all.py`：一键跑完整研究流水线——
  1. fp32 torch 基线
  2. 研究层模拟 INT8（转换 + 校准 + 评估）
  3. 逐块敏感度扫描（可跳过，`--skip-sensitivity`）
  4. 消融实验矩阵：默认方案 vs 权重逐张量 vs 激活对称 vs 不同 Observer（可跳过，`--skip-ablation`）

  产出 `results.json`（结构化数据）和 `report.md`（人类可读的 Markdown 报告，含精度对比表、**理论**体积压缩表——纯粹由位宽算出的算术比值、敏感度排名表、消融矩阵表）。
- `quantize_sam.py`：SAM 研究流水线——模拟量化 vision encoder，评估 fp32 vs 量化 mask 的自一致性 IoU，产出 `results.json` 和 `report.md`（IoU 表 + 理论压缩表）
- `qualitative_sam.py`：SAM 逐样本 mask 轮廓可视化（fp32 vs 模拟量化），按 IoU 从低到高排序，产出 `qualitative_sam_grid.png` 和 `qualitative_sam.json`

## 6. 离线权重与数据

框架**从不主动下载模型权重**——`vitquant/models/loader.py::load_model()` 只用 `timm.create_model(pretrained=False)` 构建架构，再从本地 checkpoint 文件加载权重；权重文件不存在时抛出的 `FileNotFoundError` 里直接带着可复制粘贴的下载指令（在有网络的机器上用 `pretrained=True` 下载后 `torch.save`）。也兼容 `{"model": state_dict}` 这种带 wrapper 的 checkpoint 格式（如 facebookresearch/deit 官方发布格式）。SAM 走同样的策略，但权重是 `save_pretrained` 产出的目录而非单个 `.pth` 文件（`vitquant/models/sam_loader.py`）。

Imagenette 数据集（`vitquant/data/imagenette.py`）可以自动下载（约 100MB），也可以离线预先放好目录后跳过下载。`IMAGENETTE_TO_IMAGENET1K` 硬编码了 10 个 Imagenette 类别到 ImageNet-1k 类别索引的映射，让预训练模型无需微调即可直接评估。

## 7. 实测结果（真实预训练权重）

### 7.1 deit_tiny，Imagenette 全量验证集

| 项 | 数值 |
|---|---|
| FP32（PyTorch）top-1 / top-5 | 98.19% / 99.87% |
| W8A8 模拟量化（研究层）top-1 / top-5 | 98.29% / 99.90% |
| top-1 变化 | 约 0%（相对 fp32 几乎无损，属噪声范围内的正向波动） |
| 理论权重压缩比（W8，算术值） | 4.0x |

逐块敏感度分析显示各 block 单独量化的精度影响都在 ±0.08% 以内（deit_tiny 对量化整体不敏感）；消融实验里"激活用 percentile observer"这个配置精度明显下降（96.74% vs 默认方案 98.29%），是几个消融变体里最差的一个。

### 7.2 sam-vit-base，vision-encoder W8A8 模拟量化自一致性

| 项 | 数值 |
|---|---|
| 模拟自一致性 mean IoU | 0.9287 |
| 模拟自一致性 min IoU | 0.8803 |
| 理论权重压缩比（W8，算术值） | 4.0x |

这是**真实预训练权重**（`facebook/sam-vit-base`）上的实测结果，不是玩具模型——玩具/随机权重模型上这个指标会失真地接近 1.0，因此 0.9287/0.8803 才是有意义的参考值：说明 vision encoder 量化到 W8A8 后，绝大多数情况下预测的 mask 和 fp32 高度一致，但确实存在少数样本（min IoU 0.88）有可观察的差异，值得在后续换用真实 NPU 时重点复核。

## 8. 已知限制

- 所有精度/压缩数字都是**模拟/理论值**：模拟量化在数值上做量化-反量化后仍用 fp32 计算，因此精度评估是可信的，但不产生任何真实加速或真实磁盘体积变化——这是当前阶段的设计选择（NPU 预研阶段目标硬件未定，CPU 上的"真实"数字对最终 NPU 部署没有参考价值），不是尚待修复的缺陷
- 真实延迟、真实压缩比、真实 kernel 行为要求实际的目标 NPU 及其工具链，目前不在本项目范围内；一旦确定了具体 NPU，需要单独的交付层适配该硬件的工具链（不是这个框架的职责）
- SAM 的评估是自一致性 IoU，不是对标 COCO 之类标注集的真值 mIoU
- SAM 仅量化 vision encoder，`prompt_encoder`/`mask_decoder` 保持 fp32

## 9. 如何运行

参见 [README.md](README.md)：环境搭建、权重获取、Imagenette 数据准备、各脚本用法均已说明。
</content>
