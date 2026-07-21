# ViTQuant 使用手册

面向"拿到项目要上手用"的人：这个项目能做什么、怎么装、怎么跑、产出怎么读。
想了解**为什么这样设计实验**看 [EXPERIMENT_DESIGN.md](EXPERIMENT_DESIGN.md)；想了解**每个模块
具体怎么实现的**看 [PROJECT_REPORT.md](PROJECT_REPORT.md)。

---

## 1. 这个项目能做什么

ViTQuant 是一个**模拟量化研究框架**，回答一个问题：**给定一个量化方案，模型精度会损失多少？**
用于**边缘 NPU 上线前的预研**——在还没拿到具体 NPU、没有厂商工具链之前，先把"哪些量化方案精度
可接受、哪些层最敏感"摸清楚。

它是**纯模拟量化**：数值上做量化→反量化，再用 fp32 计算，所以精度评估精确，但**不产生真实加速、
不产生真实体积压缩**（那些依赖具体 NPU 的 kernel，现在测没有意义）。没有 ONNX 导出、没有 runtime、
没有延迟基准。全部在 GPU（或 CPU）上跑。

支持两类模型，能力对称：

| | 分类 ViT（timm） | SAM 分割（HuggingFace，仅 vision encoder） |
|---|---|---|
| 主指标 | top-1 / top-5 精度 | 自一致性 IoU（fp32 mask vs 量化 mask） |
| 精度对比 | fp32 vs 模拟量化 | ✅ |
| 逐块敏感度 | ✅（哪个 block 最敏感） | ✅ |
| 混合精度权衡 | ✅（保护 K 个敏感块的精度 vs 压缩曲线） | ✅ |
| 方案消融 | ✅（粒度/对称性/Observer） | ✗（暂分类专属） |
| 理论压缩率 | ✅ | ✅ |
| 定性可视化 | ✅（逐图预测对比） | ✅（mask 轮廓叠加） |

**具体能产出的东西**：

1. **精度成本**：某个方案（W8A8 / W4A8 / 任意位宽）相对 fp32 掉多少精度。
2. **逐块敏感度排名**：哪个 block 对量化最敏感——决定未来混合精度该优先保护谁。
3. **混合精度权衡曲线**：保护最敏感的 K 个 block（保留 fp32）、其余量化，实测每个方案的精度和压缩比，
   得到一张"精度 vs 压缩率"选型表。
4. **方案消融矩阵**（分类）：per-tensor vs per-channel、对称 vs 非对称、不同 Observer 的精度对比。
5. **理论压缩率**：纯由位宽算出的算术比值（int8 = 4x，int4 = 8x）。
6. **定性可视化**：把量化前后差异最大的样本挑出来画成网格图，人工核验。

---

## 2. 安装与准备

### 2.1 环境

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
```

依赖：torch、torchvision、timm、transformers、pyyaml、numpy、matplotlib（`pyproject.toml` 里）。

### 2.2 模型权重（手动、离线）

**框架从不自动下载权重**。在有网络的机器上先下好，再拷到 `weights/`。

分类模型（单个 `.pth`）：

```bash
python -c "import timm, torch; m = timm.create_model('deit_tiny_patch16_224', pretrained=True); torch.save(m.state_dict(), 'weights/deit_tiny_patch16_224.pth')"
python -c "import timm, torch; m = timm.create_model('vit_base_patch16_224', pretrained=True); torch.save(m.state_dict(), 'weights/vit_base_patch16_224.pth')"
```

SAM（一个**目录**，不是单文件）：

```bash
python -c "from transformers import SamModel, SamProcessor; \
SamModel.from_pretrained('facebook/sam-vit-base').save_pretrained('weights/sam-vit-base'); \
SamProcessor.from_pretrained('facebook/sam-vit-base').save_pretrained('weights/sam-vit-base')"
```

> 权重文件不存在时，脚本会抛出带**可直接复制粘贴的下载命令**的报错，照着做即可。

### 2.3 数据集

分类和 SAM 都用 **Imagenette**（ImageNet 的 10 类子集，约 100MB）。

- 有网络：配置里 `data.download: true`，首次运行自动下到 `data/`。
- 离线服务器：先在别处下好 `data/imagenette2-160/` 拷过来，配置里设 `data.download: false`。

> macOS + python.org 的 Python 如果下载报 `SSL: CERTIFICATE_VERIFY_FAILED`，命令前加
> `SSL_CERT_FILE=$(.venv/bin/python -m certifi)`。

---

## 3. 配置文件

所有实验由一个 YAML 配置驱动，现成的在 `configs/`：

| 配置 | 用途 |
|---|---|
| `configs/deit_tiny.yaml` | 分类小模型，开发机快速迭代 |
| `configs/vit_base.yaml` | 分类大模型，服务器完整评估 |
| `configs/deit_tiny_w4a8.yaml` | W4A8 朴素 PTQ 基线（4-bit 权重），研究低位宽精度代价 |
| `configs/deit_tiny_w4a8_advanced.yaml` | W4A8 + 全套高级算法（MSE 截断 + SmoothQuant + AdaRound），与上一行直接对比 |
| `configs/sam_vit_b.yaml` | SAM1 vision encoder 量化（W8A8） |
| `configs/sam_vit_b_w4a8.yaml` | SAM1 W4A8 朴素 PTQ 基线（服务器生产参数：calib 64 / eval 128 / 4×4 网格） |
| `configs/sam_vit_b_w4a8_advanced.yaml` | SAM1 W4A8 + 全套高级算法，与上一行同数据同协议直接对比 |
| `configs/sam3.yaml` | SAM3（facebook/sam3 tracker，PE ViT-L backbone）W8A8，点提示 |
| `configs/sam3_w4a8.yaml` / `sam3_w4a8_advanced.yaml` | SAM3 W4A8 朴素基线 / 全栈高级，成对可比 |
| `configs/sam3_concept.yaml` | SAM3 **文本提示**概念分割（`run_sam3_concept.py`），实例集自一致性 |

### 分类配置字段（以 `deit_tiny.yaml` 为例）

```yaml
model:
  name: deit_tiny_patch16_224          # timm 架构名
  checkpoint: weights/deit_tiny_patch16_224.pth
data:
  root: data
  download: true                       # 离线设 false
  batch_size: 64
  num_workers: 4
  calib_images: 256                    # 校准用多少张图
  calib_batch_size: 32
quant:
  weight:     {bits: 8, symmetric: true,  per_channel: true,  observer: minmax}
  activation: {bits: 8, symmetric: false, per_channel: false, observer: moving_avg}
eval:
  max_batches: null                    # null=全量验证集；小整数=冒烟测试
device: auto                           # auto | cpu | cuda | mps
output_dir: outputs/deit_tiny
```

### SAM 配置字段（`sam_vit_b.yaml`）

```yaml
model:
  name: facebook/sam-vit-base
  checkpoint: weights/sam-vit-base      # 目录
data:
  root: data
  download: true
  calib_samples: 16                     # 校准样本数（正式跑建议 64-128）
  eval_samples: 16                      # 评估样本数
  prompt_grid: 4                        # 每张图 n×n 网格 prompt 点（1 = 只用图像中心一个点）
quant:
  weight:     {bits: 8, symmetric: true,  per_channel: true,  observer: minmax}
  activation: {bits: 8, symmetric: false, per_channel: false, observer: moving_avg}
device: auto
output_dir: outputs/sam_vit_b
```

### 量化方案（`quant` 段）

权重和激活的位宽、粒度、对称性、Observer 全部独立可配，**改配置即可，不用改代码**：

| 字段 | 可选值 | 含义 |
|---|---|---|
| `bits` | 任意整数（常用 8 / 4） | 位宽 |
| `symmetric` | true / false | 对称 / 非对称量化 |
| `per_channel` | true / false | 逐通道 / 逐张量 |
| `observer` | `minmax` / `moving_avg` / `percentile` / `mse` | 统计量收集方式;`mse` 为 MSE 最优截断(网格搜索使量化误差最小的截断范围,抗离群值,低位宽下收益明显) |

想跑 W4A8 就把 `weight.bits` 改成 4（`deit_tiny_w4a8.yaml` 已是现成例子）。任意位宽组合开箱即用。

### 高级 PTQ 算法（可选，分类和 SAM 通用）

在朴素「observer 标定 + 最近舍入」之上，可叠加三个正交的算法，全部配置开关、无需改代码：

```yaml
quant:
  weight:     {bits: 4, symmetric: true,  per_channel: true,  observer: mse}   # ① MSE 最优截断
  activation: {bits: 8, symmetric: false, per_channel: false, observer: mse}

smoothquant:          # ② 激活离群值迁移（校准前，逐输入通道把离群幅值挪进权重）
  enabled: true
  alpha: 0.5          # 迁移强度，0.5 = 论文默认

adaround:             # ③ 自适应权重舍入（校准后，逐层学「向上还是向下舍入」）
  enabled: true
  iters: 1000         # 每层优化步数；越大越准、越慢（GPU 上建议 1000+）
  # lr: 0.01  reg_weight: 0.01  max_tokens: 2048   # 一般不用动
```

| 算法 | 打哪个痛点 | 何时收益大 | 成本 |
|---|---|---|---|
| `observer: mse` | min/max 被离群值撑大 | 低位宽（W4）、有离群值 | 校准时一次网格搜索，几乎免费 |
| SmoothQuant | 激活个别通道幅值过大 | A8 且激活有离群通道 | 一次额外统计前向，免费 |
| AdaRound | 4-bit 最近舍入太粗暴 | W4 及以下 | 每层几百步小优化（分类几分钟，SAM 建议 GPU） |

实测（deit_tiny W4A8 全量验证集，top-1 掉点）：朴素 1.76% → MSE 截断 1.32% → **AdaRound 0.25%**；
SAM ViT-B W4A8（8 图冒烟 IoU）：朴素 0.566 → MSE 0.675 → 全栈组合 **0.705**。
三者可任意组合；执行顺序框架自动保证（SmoothQuant → 校准 → AdaRound）。报告头部会注明该次运行启用了哪些算法。

---

## 4. 使用：分类 ViT

### 一条命令跑完整流水线

```bash
# 开发机冒烟（先把 deit_tiny.yaml 的 eval.max_batches 设成个小整数）
.venv/bin/python scripts/run_classification.py --config configs/deit_tiny.yaml

# 服务器完整评估
python scripts/run_classification.py --config configs/vit_base.yaml
```

按顺序跑 6 个阶段：

1. **fp32 基线**——原始精度
2. **模拟 INT8**——转换 + 校准 + 评估
3. **逐块敏感度**——每个 block 单独量化测精度降幅
4. **混合精度权衡**——保护最敏感的 K 个 block、扫 K 得精度 vs 压缩曲线（依赖第 3 步）
5. **方案消融**——默认方案 vs 权重逐张量 vs 激活对称 vs 不同 Observer
6. **定性可视化**——逐图预测对比网格

### 常用 flag

| Flag | 作用 |
|---|---|
| `--skip-sensitivity` | 跳过逐块敏感度（同时会自动跳过依赖它的混合精度） |
| `--skip-mixed-precision` | 只跳过混合精度 |
| `--mixed-precision-ks 0,1,2,3` | 指定扫哪些 K 值（默认 0,1,2,3,4 + 全 fp32） |
| `--skip-ablation` | 跳过消融矩阵 |
| `--no-qualitative` | 跳过可视化 |
| `--qualitative-samples N` | 可视化抽多少张图（默认 30） |

> **提示**：混合精度只在低位宽（如 W4A8）下才有意义——W8A8 下各 block 都不敏感，曲线是平的。
> 想看有意义的混合精度曲线，用 `configs/deit_tiny_w4a8.yaml`。

---

## 5. 使用：SAM（SAM1 与 SAM3）

```bash
python scripts/run_sam.py --config configs/sam_vit_b.yaml   # SAM1 ViT-B
python scripts/run_sam.py --config configs/sam3.yaml        # SAM3（PE ViT-L backbone）
```

同一个脚本同时支持两代 SAM，由配置里的 `model.family`（`sam` / `sam3`）区分。
SAM3 走 `Sam3TrackerModel`（点提示交互分割入口，与 SAM1 协议同构），其
Perception-Encoder ViT 注意力（RoPE、q/k/v/o 分离）已做显式矩阵乘量化重写，
敏感度按 `backbone.layers.N` + patch_embeddings + 4 个 FPN neck 分支分组（共 37 组）。

> **SAM3 权重是 gated 仓库**：先在 https://huggingface.co/facebook/sam3 申请访问并同意
> 许可，然后在能联网的机器上 `hf auth login` 后执行 `configs/sam3.yaml` 顶部注释里的
> 导出命令，把 `weights/sam3/` 目录拷到离线机器。框架本身永不联网下载权重。
> 也可用非 gated 完整镜像 `Justin331/sam3`（命令同在配置注释里；研究用途）。

### SAM3 文本提示（概念分割）

```bash
python scripts/run_sam3_concept.py --config configs/sam3_concept.yaml
```

与点提示不同的实验轴：每张图用**它自己的类别名**做文本 prompt（如 "gas pump"），模型返回
图中该概念的**所有实例**。fp32 与量化模型的实例集数量可能不同，评估先按 mask IoU 贪心匹配，再报：

| 指标 | 含义 |
|---|---|
| `consistency` | Σ(匹配对 IoU) / max(fp32 实例数, 量化实例数)——漏检和幻检都计 0 分；敏感度/混合精度用这个标量 |
| `detection F1` | 实例数量层面的一致率 |
| `matched IoU` | 只看匹配上的对的 mask 质量 |

量化对象仍然只有共享的 PE vision encoder；文本编码器、DETR 解码器、mask head 全部保持
fp32。文本链路更长（视觉特征还要和文本特征做匹配），预期比点提示协议对量化**更敏感**，
是更严苛的考题。定性图:fp32 实例半透明填充、匹配的量化实例同色细线、未匹配的量化实例红线。

只量化 `vision_encoder`（ViT backbone），`prompt_encoder`/`mask_decoder` 保持 fp32。按顺序跑：

1. **自一致性 IoU**——fp32 mask vs 量化 mask 的相似度（SAM 没有 top-1）
2. **逐块敏感度**（IoU）——每个 vision encoder block 单独量化测 IoU 降幅
3. **混合精度权衡**（IoU）——保护最敏感的 K 层、扫 K 得 IoU vs 压缩曲线
4. **定性可视化**——fp32 vs 量化 mask 叠加网格

**Prompt 覆盖范围**：评估默认对每张图撒 `prompt_grid` × `prompt_grid` 的均匀网格点
（默认 4×4 = 16 个 prompt），衡量量化对**全图各处对象**分割的影响，而不只是中心物体。
被量化的 vision encoder 每张图仍只前向一次（多点只重复很轻的 prompt encoder / mask
decoder），所以网格几乎不增加评估耗时。设 `prompt_grid: 1`（或 `--prompt-grid 1`）
可回到单中心点模式。校准始终用单中心点——vision encoder 的激活与 prompt 点无关，
网格不会改变校准统计量。

### 常用 flag

| Flag | 作用 |
|---|---|
| `--skip-sensitivity` | 跳过逐块 IoU 敏感度（同时跳过混合精度） |
| `--skip-mixed-precision` | 只跳过混合精度 |
| `--mixed-precision-ks 0,1,2,3` | 指定 K 值 |
| `--no-qualitative` | 跳过可视化 |
| `--qualitative-samples N` | 可视化抽多少张（默认 8） |

---

### 保存量化模型 & 无标注推理

实验脚本默认**不保存**量化模型(每次运行重新校准)。加 `--save-quantized` 可把校准 +
AdaRound 的全部成果持久化,之后在任意无标注图片上直接推理,**跳过重校准**:

```bash
# 实验时顺手保存(写入 output_dir/quantized_state.pt + quant_meta.json)
python scripts/run_sam.py --config configs/sam3_w4a8_advanced.yaml --save-quantized

# 之后:对无标注测试集推理(每图输出 masks .npz + 叠加可视化 .png)
python scripts/infer_sam.py --artifact outputs/sam3_w4a8_advanced --images 测试集目录/
# 文本推理:指定要找的概念,输出全图所有实例
python scripts/infer_sam.py --artifact outputs/sam3_w4a8_advanced --images 目录/ --text "gas pump"
```

**同一个 SAM3 产物,点提示和文本推理都能用**:tracker 与文本概念模型共享同一个被量化的
vision encoder,且其校准与提示方式无关,所以点提示实验存的量化参数对文本任务严格有效
——给 `--text` 就自动按概念模型(`Sam3Model`)加载同一份量化状态,不给就走点网格。
(SAM1 产物无文本通路,`--text` 会明确报错。)

原理:所有算法都不改原始权重(scale/zero_point、AdaRound round_offset、SmoothQuant
smooth_scale 全是运行时 buffer),所以产物**只存量化参数**(SAM1 全栈约 88MB,其中
AdaRound 逐权重舍入占大头,已按 {0,1} 压成 uint8),加载 = 原 fp32 checkpoint + convert +
灌 buffer,与保存时的模型**逐位一致**(已验证)。`run_classification.py` 也支持
`--save-quantized`。注意 meta 里记录的 checkpoint 路径是相对项目根目录的,推理机器上
需要有同一份 fp32 权重目录。

---

## 6. 输出产物

都落在配置的 `output_dir`（如 `outputs/deit_tiny/`）。

### 分类

| 文件 | 内容 |
|---|---|
| `report.md` | 人类可读报告：精度表 / 理论压缩表 / 逐块敏感度表 / 混合精度权衡表 / 消融表（结束时也打印到终端） |
| `results.json` | 结构化数据，供二次处理 |
| `qualitative_grid.png` | 逐图 fp32 vs 量化预测对比，被量化"翻转"的样本排前面标红 |
| `qualitative.json` / `.md` | 每张图的预测详情 |

### SAM

| 文件 | 内容 |
|---|---|
| `report.md` | IoU 表 / 理论压缩表 / 逐块敏感度表 / 混合精度权衡表 |
| `results.json` | 结构化数据（含 `mean_iou`/`min_iou`/`per_sample_iou`、敏感度、混合精度） |
| `qualitative_sam_grid.png` | fp32 mask（半透明填充）vs 量化 mask（细边界线）叠加，IoU 最差的排前面标红；网格 prompt 模式下每个点一种颜色，同色填充/线条属于同一 prompt |
| `qualitative_sam.json` | 每个样本逐 prompt 点的 IoU 详情 |

### 怎么读报告（示例：混合精度权衡表）

```
| K protected | Blocks kept FP32          | Top-1  | Avg weight bits | Compression vs FP32 |
| 0           | (none — uniform)          | 92.10% | 4.00            | 8.00x               |
| 1           | blocks.0                  | 95.80% | 4.30            | 7.44x               |
| 2           | blocks.0, blocks.1        | 97.20% | 4.60            | 7.00x               |
```

读法："如果 NPU 只能均匀 W4，精度 92.1%；如果保护最敏感的 1 个 block，精度升到 95.8%，压缩比只
从 8x 降到 7.4x。" 这就是给硬件选型的直接依据。

> **重要**：混合精度表里每一行的精度都是**真实评估**出来的，不是把单块敏感度相加预测的——量化误差
> 跨 block 非线性叠加，敏感度排名只用来决定"先保护谁"。

---

## 7. 常见任务配方

**换量化方案（如跑 W4A8）**：复制一份配置，改 `quant.weight.bits: 4`，改 `output_dir`，重跑。或直接用
`configs/deit_tiny_w4a8.yaml`。

**只想快速看精度、不要敏感度/消融/可视化那些**（省时间）：

```bash
python scripts/run_classification.py --config configs/vit_base.yaml \
  --skip-sensitivity --skip-ablation --no-qualitative
```

**离线服务器**：配置里 `data.download: false`，并预先放好 `data/imagenette2-160/` 和 `weights/`。

**冒烟测试（几十个 batch 验证跑得通）**：把配置的 `eval.max_batches` 设成个小整数（如 5）。

**只跑混合精度、想扫全部 K**：`--mixed-precision-ks 0,1,2,3,4,5,6,7,8,9,10,11,12,13`（分类 13 组，
SAM 14 组；每个 K 多一次完整评估，注意耗时）。

**指定设备**：配置里 `device: cuda`（或 `cpu` / `mps` / `auto`）。

---

## 8. 单项脚本（不跑完整流水线时用）

| 脚本 | 作用 |
|---|---|
| `scripts/evaluate.py --config ...` | 只测 fp32 基线精度 |
| `scripts/quantize.py --config ...` | 单次模拟量化实验（转换→校准→评估），输出 `quantize_result.json` |
| `scripts/qualitative.py --config ... [--num-samples N]` | 只重新生成分类可视化网格 |
| `scripts/qualitative_sam.py --config ... [--num-samples N]` | 只重新生成 SAM mask 可视化网格 |

（可视化脚本产出的图 `run_classification.py`/`run_sam.py` 里已经自动生成了；这两个是想单独重跑或换
样本数时用。）

---

## 9. 典型工作流：给一颗候选 NPU 做量化预研

1. **定基线**：跑 `run_classification.py`（或 `run_sam.py`），看 fp32 → W8A8 精度掉多少。能接受就往下。
2. **看能压多低**：把配置改成 NPU 偏好的低位宽（很多边缘 NPU 要 4-bit 权重），跑 W4A8，看均匀量化下
   精度还行不行。
3. **不行就上混合精度**：看混合精度权衡表——保护几个最敏感的 block，能在多大压缩代价下把精度救回到
   可接受线。
4. **对齐 NPU 的方案**：一旦确定具体 NPU，把 `quant` 段改成它实际要求的方案（很可能是 per-tensor +
   对称 + 特定位宽），重跑上述流程，拿到贴合目标硬件的精度参考——**在碰厂商工具链之前**。

这些产出（精度保持率、敏感度、混合精度曲线、理论压缩比）都**与具体硬件无关、可迁移到任意 NPU**。

---

## 10. 边界（不做什么）

- **所有数字是模拟/理论值**：精度评估可信，但**没有真实加速、真实体积压缩**——那需要具体 NPU 及其
  工具链，超出本项目范围。
- **不测真实延迟 / 吞吐 / 设备内存**。
- **SAM 只量化 vision encoder**，`prompt_encoder`/`mask_decoder` 恒为 fp32；评估是自一致性 IoU，
  不是对标 COCO 的真值 mIoU。
- **敏感度/混合精度是逐 block 粒度**，不是层内更细的粒度；消融是几个代表性变体，不是穷举网格。

这些都是当前"NPU 预研"阶段的既定范围，不是待修的缺陷。
