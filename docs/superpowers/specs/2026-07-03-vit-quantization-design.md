# ViT 量化工程设计文档

日期：2026-07-03
状态：已确认

## 1. 目标与定位

构建一个 **可复用的 ViT 量化工程框架**，同时满足两个诉求：

1. **工程交付**：产出真正能跑、能加速、能落盘的量化模型（ONNX INT8），附带真实的精度/延迟/体积评估数字。
2. **精度攻坚/研究**：量化策略（位宽、粒度、scheme、Observer）全部可插拔，提供逐层敏感度分析与消融实验能力，为未来 INT4、QAT 等研究打基础。

### 非目标（本期）

- QAT 训练（预留接口，不实现）
- INT4 及以下位宽（预留接口，不实现）
- CoreML / TensorRT 后端（架构预留扩展点）
- 在完整 ImageNet 上的权威评估

## 2. 核心架构：两层设计

```
研究层（自研量化内核）              交付层（ONNX Runtime）
──────────────────────           ──────────────────────
· Observer / FakeQuant            · PyTorch → ONNX 导出
· 模拟量化精度评估                 · ORT 静态 INT8 量化（QDQ）
· 逐层敏感度分析                   · 真实延迟 benchmark
· 消融实验（策略可插拔）           · 真实模型体积
```

**分工**：

- **研究层**用自研内核做**模拟量化**（fake-quant：int8 quant→dequant 后仍用 fp32 计算）。它不产生加速，但能精确测量量化对精度的影响，支持任意策略实验。
- **交付层**用 ONNX Runtime 做**真实 INT8 量化**：权重真实以 int8 落盘（~4× 压缩），用 int8 算子推理（真实加速）。

**衔接与交叉验证**：研究层确定最佳量化策略 → 交付层按该策略导出真实模型。模拟精度与 ORT 真实精度应当接近（差距 <0.5% top-1 视为正常），这也是框架正确性的校验点。

## 3. 目录结构

```
vitquant/
├── quant/                     # 研究层：自研量化内核
│   ├── observers.py           # MinMaxObserver / MovingAvgMinMaxObserver / PercentileObserver
│   │                          # 支持 per-tensor 与 per-channel 统计
│   ├── fake_quant.py          # FakeQuantize 模块（STE 直通估计器）、quant/dequant 原语
│   ├── qconfig.py             # QConfig：位宽 / 粒度 / 对称性，权重与激活独立配置
│   ├── modules.py             # QuantLinear / QuantConv2d(patch embed) / QuantMatMul(attention)
│   ├── convert.py             # 模块替换：fp32 timm ViT → 可量化模型
│   └── calibrate.py           # 跑校准集，驱动 Observer 收集激活统计
├── deploy/                    # 交付层：ONNX Runtime
│   ├── export_onnx.py         # PyTorch → ONNX（opset 17+，动态 batch）
│   ├── quantize_ort.py        # ORT 静态量化：QDQ 格式，权重 per-channel，激活校准
│   └── benchmark.py           # 延迟测量（warmup + 多次取中位数）、体积统计
├── models/
│   └── loader.py              # timm 模型加载（vit_base_patch16_224 / deit_tiny_patch16_224）
├── data/
│   └── imagenette.py          # Imagenette 下载、DataLoader、Imagenette→ImageNet-1k 类别索引映射
├── eval/
│   ├── metrics.py             # top-1 / top-5、模型大小、延迟聚合
│   └── evaluate.py            # 评估主循环、逐层敏感度分析、消融驱动
├── configs/                   # YAML 实验配置（模型、量化策略、校准参数）
├── scripts/
│   ├── quantize.py            # 单次量化实验入口
│   ├── evaluate.py            # 评估入口
│   └── run_all.py             # 一键跑完整评估矩阵，产出报告
└── docs/superpowers/specs/    # 设计文档
```

## 4. 量化方案

| 项 | 默认配置 |
|---|---|
| 方法 | PTQ 静态量化（训练后量化） |
| 权重 | per-channel 对称 INT8 |
| 激活 | per-tensor 非对称 INT8，静态校准 |
| 校准集 | Imagenette train 随机 ~256 张 |
| 量化范围 | Linear（QKV/proj/MLP）、patch embed Conv2d、attention 中的 MatMul |
| 不量化 | LayerNorm、Softmax、GELU（保持 fp32，ViT 量化的已知敏感点） |

**可插拔维度**（消融实验轴）：

- 粒度：per-tensor ↔ per-channel
- 对称性：对称 ↔ 非对称
- Observer：MinMax / MovingAvgMinMax / Percentile
- 位宽：INT8（默认），接口支持任意位宽（为 INT4 研究预留）

## 5. 模型与数据

- **模型**：
  - `vit_base_patch16_224`（~86M 参数）——主要评估对象
  - `deit_tiny_patch16_224`（~5M 参数）——开发迭代与快速验证
  - 均为 ImageNet-1k 预训练权重（timm）
- **数据**：Imagenette v2（ImageNet 的 10 类子集，160px 版 ~100MB）
  - 预训练 ViT 可直接评估：预测 1000 类 logits 后，映射到 Imagenette 的 10 个 ImageNet 类别索引取 argmax
  - 无需微调即得真实 top-1/top-5

## 6. 评估交付物

`scripts/run_all.py` 一键产出以下报告（markdown 表格 + 原始 JSON）：

1. **精度对比表**：FP32 baseline / INT8 模拟量化 / INT8 ORT 真实量化 的 top-1、top-5 及精度差
2. **真实体积**：fp32 `.onnx` vs int8 `.onnx` 落盘大小（MB）与压缩比
3. **真实延迟**：ORT CPU（M5）上 fp32 vs int8 的单图延迟（ms，中位数）与加速比
4. **逐层敏感度分析**：逐个 block/layer 单独量化，量化其余保持 fp32，报告各层引起的精度下降排名
5. **消融矩阵**：粒度 × 对称性 × Observer 的精度对比

## 7. 错误处理与验证

- 校准前若 Observer 未收集到统计数据，convert/量化步骤显式报错
- ONNX 导出后用 `onnx.checker` 校验 + 与 PyTorch 输出做数值一致性对比（fp32 导出 atol<1e-4）
- 模拟量化与 ORT 真实量化精度差距 >1% top-1 时在报告中标红提示
- 单元测试：quant/dequant 数值正确性、Observer 统计正确性、模块替换后 fp32 等价性（关闭 fake-quant 时输出与原模型一致）

## 8. 环境

| 项 | 值 |
|---|---|
| 硬件 | Apple M5 / 32GB RAM（CPU 推理） |
| Python | 3.14.5 |
| 依赖 | torch 2.12.1、timm、onnx、onnxruntime、pyyaml |
| 数据 | Imagenette v2 160px（~100MB 自动下载） |

## 9. 已确认的决策记录

| 决策点 | 选择 | 理由 |
|---|---|---|
| 项目目标 | 可复用框架 + 精度研究 + 工程交付 | 用户确认 |
| 评估数据 | Imagenette（小型代理集） | 预训练 ViT 可直接评估，下载小 |
| 量化内核 | 自研（Observer + FakeQuant + 模块替换） | 全透明，适合精度研究 |
| 部署后端 | ONNX Runtime | 跨平台成熟稳定，CPU int8 真实加速，Python 3.14 兼容 |
| 模拟 vs 真实 | 两层都做 | 模拟量化量精度，ORT 给真实交付数字 |
