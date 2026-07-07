# ViTQuant 项目实施报告

本文档详细说明 ViTQuant 项目具体做了什么：架构设计、每个模块的实现内容、实测得到的结果，以及在服务器部署过程中排查并解决的一系列硬件兼容性问题。

## 1. 项目目标

为 Vision Transformer 构建一个 INT8 量化框架，同时满足两个诉求：

- **精度研究**：量化策略（位宽、粒度、对称性、Observer 类型）完全可插拔，支持逐层敏感度分析和消融实验
- **工程交付**：产出真正能在生产环境跑起来、有真实体积压缩和真实延迟数字的部署产物，而不是纸面上的模拟数据

为此采用**两层架构**：

| 层 | 目录 | 作用 |
|---|---|---|
| 研究层 | `vitquant/quant/` | 自研 fake-quant 内核，模拟量化，只测精度不测速度 |
| 交付层 | `vitquant/deploy/` | 导出 ONNX，用 ONNX Runtime 做真实 INT8 静态量化，真实体积/延迟 |

## 2. 研究层：自研量化内核

### 2.1 量化配置（`vitquant/quant/qconfig.py`）

`TensorQConfig`（frozen dataclass）描述单一张量（权重或激活）的量化方案：位宽（`bits`）、是否对称（`symmetric`）、是否逐通道（`per_channel`）、通道轴（`ch_axis`）、Observer 类型（`observer`）、百分位阈值（`percentile`）。

`QConfig` 组合权重和激活两套 `TensorQConfig`，默认方案：权重逐通道对称 INT8，激活逐张量非对称 INT8（moving-average observer）。`qconfig_from_dict()` 支持从 YAML 加载的 dict 直接构造，位宽等参数完全独立可配置——这也是后续能直接支持 W4A8 等任意位宽组合的基础，不需要改代码，改配置文件即可。

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

`FakeQuantize` 模块有三种状态：`observing`（收集统计量，直通不改变数值）、`quantizing`（应用量化）、都不开（恒等映射）。`freeze()` 从 Observer 计算出 scale/zero_point 并切换到量化模式。这就是"模拟量化"的含义——数值上经过量化再反量化回 fp32，用 fp32 继续计算，所以能精确测精度损失，但不会有任何真实加速。

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

`calibrate()` 驱动完整的 观察 → 冻结 → 量化 生命周期：把校准集喂给模型收集统计量，然后冻结 qparams，最后把所有 FakeQuantize 切到量化模式。

## 3. 评估体系（`vitquant/eval/`）

### 3.1 指标（`metrics.py`）

`topk_correct()` 计算 top-k 命中数（自动兼容类别数小于 k 的情况，比如 Imagenette 只有 10 类却要算 top-5）；`AccuracyMeter` 跨 batch 累计。

### 3.2 评估循环（`evaluate.py`）

- `evaluate_torch()`：PyTorch 模型评估，把 1000 类 logits 切到 Imagenette 对应的 10 个类别索引上再算 top-1/top-5
- `evaluate_onnx()`：同样逻辑，但通过 ONNX Runtime 推理
- `block_sensitivity()`：逐块敏感度分析——先测全 fp32 基线，然后每次只让一个 block（`patch_embed`、`blocks.0`……`blocks.11`）量化、其余保持 fp32，测量精度下降幅度，按下降程度从高到低排序。用 `try/finally` 保证扫描过程中即使抛异常，模型也会被恢复成完全量化状态，不会留下半量化的脏状态。

两个函数都支持可选的 `progress` 回调（每完成约 20% 的 batch 打一行状态），`block_sensitivity` 额外支持 `log` 回调汇报当前扫到第几个 block——这是服务器上跑长时间任务时避免"看起来像卡住"而加的。

## 4. 交付层：真实部署（`vitquant/deploy/`）

### 4.1 ONNX 导出（`export_onnx.py`）

`export_fp32_onnx()` 导出**原始 fp32 模型**（不是量化后的模型）到 ONNX，动态 batch 维度，导出后立即：

1. 用 `onnx.checker.check_model` 校验图的合法性
2. 拿同一个输入分别过 PyTorch 和 ONNX Runtime，要求数值一致（`atol=1e-4`），不一致直接抛错

这是一个数值一致性硬闸，保证导出的 ONNX 图和研究层用的 PyTorch 模型是"同一个模型"，而不是巧合地长得像。

### 4.2 ORT 静态量化（`quantize_ort.py`）

`TorchCalibrationReader` 把 PyTorch DataLoader 的校准数据喂给 ORT 的量化 API。`quantize_onnx()` 做的事：

1. `quant_pre_process`：ONNX 图的形状推断与预处理优化
2. `quantize_static`：真正的静态量化，per-channel 权重（对齐研究层默认方案），激活用 `QUInt8`（x86-64 CPU EP 推荐类型）

支持的参数化维度（都是这次实测过程中逐步加上的）：

- **`weight_bits`**（8 或 4）：8 是默认的成熟 INT8 路径；4 走 ORT 的 int4 contrib 算子（`com.microsoft` 域），产出更小的体积（实测约 6.5x 压缩 vs INT8 的 3.6x），但没有真实 CPU 加速（CPU EP 没有原生 int4 矩阵乘 kernel）
- **`quant_format`**（`qdq` 或 `qoperator`）：`qdq` 插入 QuantizeLinear/DequantizeLinear 节点，依赖 ORT 运行时优化器把它们融合成快速 int8 kernel；`qoperator` 在量化时就直接把量化算子（如 `QLinearMatMul`）写死进图里，不依赖运行时融合
- **`graph_optimization_level`**（通过 `vitquant/utils/ort_session.py::create_cpu_session` 统一管理）：可选地限制 ORT 的图优化级别，默认用 ORT 自己的最优设置

### 4.3 基准测试（`benchmark.py`）

`model_size_mb()` 读取文件真实大小；`benchmark_onnx()` 用 ORT CPU EP 跑多次推理（预热后取中位数），得到真实的单图延迟。

## 5. 配置与脚本

### 5.1 配置文件（`configs/`）

- `deit_tiny.yaml`：开发机用的小模型配置
- `vit_base.yaml`：服务器用的完整模型配置
- `deit_tiny_w4a8.yaml`：W4A8 量化方案示例

配置结构覆盖模型（架构名 + 本地权重路径）、数据（Imagenette 路径、批大小、是否自动下载）、量化方案、评估（`max_batches` 控制冒烟测试还是全量评估）、基准测试参数、设备、输出目录，以及可选的 `onnx` 段（`graph_optimization_level`、`quant_format`，用于规避特定硬件的兼容性问题）。

### 5.2 脚本（`scripts/`）

- `quantize.py`：单次模拟量化实验（转换 → 校准 → 评估），输出 JSON
- `evaluate.py`：评估 fp32 基线或指定 ONNX 文件
- `run_all.py`：一键跑完整 6 阶段流水线——
  1. fp32 torch 基线
  2. 研究层模拟 INT8（转换 + 校准 + 评估）
  3. 逐块敏感度扫描（可跳过）
  4. 消融实验矩阵：默认方案 vs 权重逐张量 vs 激活对称 vs 不同 Observer（可跳过）
  5. 交付层：ONNX 导出 + ORT 真实量化 + 精度评估 + 真实体积
  6. 真实延迟基准

  产出 `results.json`（结构化数据）和 `report.md`（人类可读的 Markdown 报告，含精度对比表、体积压缩表、延迟加速表、敏感度排名表、消融矩阵表），并自动检测"模拟量化 vs ORT 真实量化"精度差距是否超过 1%，超过就在报告里标红警示。

## 6. 离线权重与数据

框架**从不主动下载模型权重**——`vitquant/models/loader.py::load_model()` 只用 `timm.create_model(pretrained=False)` 构建架构，再从本地 checkpoint 文件加载权重；权重文件不存在时抛出的 `FileNotFoundError` 里直接带着可复制粘贴的下载指令（在有网络的机器上用 `pretrained=True` 下载后 `torch.save`）。也兼容 `{"model": state_dict}` 这种带 wrapper 的 checkpoint 格式（如 facebookresearch/deit 官方发布格式）。

Imagenette 数据集（`vitquant/data/imagenette.py`）可以自动下载（约 100MB），也可以离线预先放好目录后跳过下载。`IMAGENETTE_TO_IMAGENET1K` 硬编码了 10 个 Imagenette 类别到 ImageNet-1k 类别索引的映射，让预训练模型无需微调即可直接评估。

## 7. 实测结果（deit_tiny，真实预训练权重，Imagenette 全量验证集）

| 项 | 数值 |
|---|---|
| FP32（PyTorch）top-1 / top-5 | 98.19% / 99.87% |
| FP32（ONNX 导出后）top-1 / top-5 | 98.19% / 99.87%（与 PyTorch 完全一致） |
| W8A8 模拟量化（研究层）top-1 / top-5 | 98.29% / 99.90% |
| W8A8 真实 ORT 量化 top-1 / top-5 | 98.11% / 99.87% |
| 模拟 vs 真实量化精度差距 | 0.18%（低于 1% 告警阈值，交叉验证通过） |
| ONNX 体积压缩 | 23.0MB → 6.3MB，压缩比 3.63x |

逐块敏感度分析显示各 block 单独量化的精度影响都在 ±0.08% 以内（deit_tiny 对量化整体不敏感）；消融实验里"激活用 percentile observer"这个配置精度明显下降（96.74% vs 默认方案 98.29%），是几个消融变体里最差的一个。

## 8. 服务器部署踩坑记录

在把框架部署到用户的 Linux 服务器（AMD EPYC 云虚拟机）过程中，遇到并解决了以下问题：

1. **`ModuleNotFoundError: No module named 'vitquant'`**：新环境需要先 `pip install -e .`，框架代码本身没问题
2. **macOS 上 Imagenette 下载 SSL 证书报错**：python.org 版 Python 需要显式指定 `SSL_CERT_FILE=$(python -m certifi)`
3. **看起来"卡住没输出"**：根因是 Python 在非真实 tty 环境下对 stdout 使用全缓冲，进度打印全部堆积在缓冲区里。修复：强制 `sys.stdout.reconfigure(line_buffering=True)`，并加了细粒度的进度回调（`evaluate_torch`/`evaluate_onnx`/`calibrate`/`block_sensitivity` 都支持）
4. **W4A8 请求下 `Illegal instruction (core dumped)` 崩溃**：ORT 的 int4 反量化 contrib 算子需要 AVX-512，该服务器 CPU 缺失。修复：加了 `_check_int4_cpu_support()` 前置检测，缺 AVX-512 时抛清晰的 `RuntimeError` 而不是让进程被内核裸杀
5. **默认 W8A8 也崩溃，且是在 DataLoader worker 子进程里**：`num_workers>0` fork 出的子进程里 CPU 特征探测和主进程不一致，触发 SIGILL。修复：改用 `num_workers=0`
6. **W8A8 单进程下依然崩溃，位置在 `ort.InferenceSession.run()`**：加了 `faulthandler.enable()` 抓到崩溃时的 Python 调用栈，配合一个隔离诊断脚本二分排查 ORT 的四种图优化级别，定位到 `ORT_ENABLE_EXTENDED`（及默认的 `ORT_ENABLE_ALL`）触发的**量化算子融合**产生了这台机器执行不了的 kernel；`ORT_ENABLE_BASIC` 不融合，侥幸没崩
7. **发现"basic 不崩"只是因为它压根没有真正跑 int8 kernel**：BASIC 级别下 QDQ 图退化成"反量化回 fp32 算 → 再量化"，比纯 fp32 还慢（延迟从 fp32 的 14.28ms 变成 75.67ms）。尝试用 `QuantFormat.QOperator`（量化时直接把 int8 算子烤进图里，不依赖运行时融合）—— **同样在任意优化级别崩溃**，换了两个 onnxruntime 版本（1.27.0、1.24.1）结果一致
8. **最终结论**：这台服务器的 CPU/虚拟化环境**在内核层面无法执行 ORT 的 int8 量化 kernel**，与图构造方式、优化级别、ORT 版本均无关，是硬件/虚拟化限制,非代码缺陷。已在 `report.md` 里加了自动检测：一旦发现使用了 `qdq + basic/disable` 这个规避组合，会在延迟表格下方自动追加警示，说明该延迟数字不代表真实硬件加速，只有精度和体积数字仍然有效

以上第 4-8 项的技术细节和结论都写进了 [README.md](README.md) 对应章节，方便未来在类似硬件上快速定位同类问题。

## 9. 已知限制

- 交付层的 ONNX 精度/延迟评估固定用 CPU EP（这是刻意设计——int8 CPU 加速本身就是最有代表性的部署场景）；可选的 CUDA EP fp32 GPU 基线（spec 中提到的"可选"项）未实现
- int4（W4A8）真实部署路径要求 x86-64 CPU 支持 AVX-512，且在虚拟化程度较高的云 CPU 上可能完全无法执行（见第 8 节）
- 部分虚拟化云 CPU（如本项目实测的 AMD EPYC 实例）可能连标准 INT8 kernel 都无法执行，需要退化到"仅精度可信、延迟不可信"模式

## 10. 如何运行

参见 [README.md](README.md)：环境搭建、权重获取、Imagenette 数据准备、`run_all.py` 使用方法均已说明。
