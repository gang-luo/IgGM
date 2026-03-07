# IgGM Lightning 训练与测试说明


## 0. 关于 `ppi_ckpt / design_ckpt / igso3_buffer` 的说明

- `ppi_ckpt`：建议使用预训练权重；若留空，训练脚本会自动下载并使用官方 `esm_ppi_650m_ab`。
- `design_ckpt`：**可选**；若留空，将从随机初始化的 `DesignModel` 开始训练（适合你说的“主干部分需要训练和优化”场景）。
- `igso3_buffer`：仅用于加速 SO(3) 采样，**可选**。

因此，不再强制你必须提供这三个路径。

---

## 1. 配置管理（YAML）

统一配置文件：`config/train_lightning.yaml`。

集中保存：
- `data`：数据路径、split 文件路径、cluster 文件路径。
- `model`：预训练权重路径与 IGSO3 buffer。
- `optimizer`：学习率、权重衰减、梯度裁剪。
- `loss`：`gamma`、`loss_viol_weight`。
- `metrics`：DockQ 判定阈值。
- `trainer`：`pl.Trainer` 相关超参数。
- `wandb`：api key、project、run name 等。
- `runtime`：输出目录、seed、resume、run_test。

训练主函数 `scripts/train_iggm_lightning.py` 默认读取 YAML，CLI 参数可覆盖。

---

## 2. 数据处理与论文式数据抽取

### 2.1 基础 pt 构建

```bash
python data/prepare_data.py \
  --dataset_name sabdab \
  --raw_root ./data/sabdab/metadata \
  --out_root ./data/sabdab/processed
```

### 2.2 构建训练/验证/测试 prot_ids 与 cluster

`data/prepare_data.py` 新增 split 构建能力，可在处理后直接生成：
- `train_prot_ids.txt`
- `train_prot_cluster.txt`
- `val_prot_ids.txt`
- `test_prot_ids.txt`

并遵循论文时间窗口：
- train: `<= 2022-12-31`
- val: `2023-01-01 ~ 2023-06-30`
- test: `2023-07-01 ~ 2023-12-30`

示例：

```bash
python data/prepare_data.py \
  --dataset_name sabdab \
  --raw_root ./data/sabdab/metadata \
  --out_root ./data/sabdab/processed \
  --build_splits \
  --split_out_dir ./data/sabdab/processed/sabdab/split \
  --cluster_identity 0.95
```

说明：
- 优先调用 `cd-hit` 对重链序列做 95% 聚类；若环境无 `cd-hit`，自动回退到内置 greedy 聚类。
- val/test 会基于 train 重链序列做 95% 相似性去重。

---

## 3. Lightning 数据加载与 cluster 采样

`src/iggm_lightning/data_module.py` 支持读取 `train_ids/val_ids/test_ids` 以及 `train_clusters`。

当前实现改为 **Dataset 侧 Lazy Loading**：
- DataModule 只在 `setup` 阶段建立 `prot_id -> (sample_path, processed_pdb_path)` 轻量索引；
- `_ProteinSampleDataset.__getitem__` 才会按需读取当前样本的 `.pt/.pdb` 并构建模型输入所需 `payload`；
- Dataset 内置 LRU 缓存（`lazy_cache_size`），避免重复解析同一样本；
- `IgGMLightningModule` 只消费 `payload`、执行扩散/前向/损失，保持“数据解析”和“训练逻辑”解耦。

补充说明（模型参数统计）：
- Lightning summary 中只显示 `nn.Module` 子模块，因此会显示 `model` 与 `plm_featurizer`；
- `diffuser` 是采样/扰动调度对象，不是可训练 `nn.Module`，所以不会出现在参数表中；
- 当前训练已默认冻结 `plm_featurizer`（`requires_grad=False` 且固定 `eval` 模式），仅训练 `DesignModel`。

### 3.1 `prepare_data_fromzip.py` 产物在训练中的作用

你列出的几个目录里，当前 Lightning 训练链路**真正强依赖**的是：
- `processed/pdb/*.pdb`：训练时按 batch 读取（懒加载），并基于标准化后的 `H/L/A` 链构建输入。
- `processed/sabdab_debug20/split/*.txt`：用于确定 train/val/test 样本，以及 train cluster 采样。
- `processed/sabdab_debug20/samples/*.pt`：按 batch 读取 `cdr_sequences / sequence_lengths`，用于指标与二阶段掩码采样。

以下文件当前属于“旁路产物/可追溯信息”：
- `samples/*.pt`：可选增强输入；存在时训练会懒加载其中的 `cdr_sequences/sequence_lengths`，不存在时会自动降级为仅用 `processed/pdb`。
- `processed/fasta/*.fasta`：序列备份与可视化排查用，训练主链路不依赖该文件读取。
- `metadata.json`：用于描述数据集边界与路径，split 信息仍以 `split/*.txt` 为准。

### 3.2 metadata 与 split/fasta/pdb 对不上 的根因

`prepare_data_fromzip.py` 里 `metadata.json.sample_ids` 保存的是 `pdb_id`（如 `7x3e`）；
而 split / fasta / pdb 文件名使用的是 `prot_id`（如 `7x3e_H_L_C`）。

这两者不是同一层级 ID：
- `sample_id`（metadata）= 复合物主 ID（PDB）
- `prot_id`（split/训练）= `pdb + H/L/A 链组合`

因此“看起来不一致”是由字段语义不同导致，不是单纯文件损坏。
本仓库现已在 `ProcessedSabdabDataModule` 中补齐这层映射：当 metadata 不含 `entries` 时，优先从 split 的 `prot_id` 重建训练 entry 并匹配 `processed/pdb/*.pdb`，从而与 `prepare_data_fromzip.py` 的输出对齐。

训练阶段若提供 `train_prot_cluster.txt`，每个 epoch 会执行：
- 对每个 cluster 随机抽取 1 个样本
- 使用抽样后的样本集合进行该 epoch 训练

满足论文描述的“每个 epoch 从每个 cluster 随机采样一个样本”。

### 3.3 关于处理后 PDB 只有 H/L/A 链

`prepare_data_fromzip.py` 在导出 `processed/pdb/*.pdb` 时会把链名规范化为 `H/L/A`，并重排残基编号。
因此训练读取 processed PDB 时，不再依赖原始链名（如 `K/F/e` 等），而是固定使用 `H/L/A` 进行解析。

---

## 4. 损失与评估指标

### 4.1 损失（`src/iggm_lightning/losses.py`）

- Stage-1: `L = L_geo + L_frame + L_iFrame + 0.02 * L_viol`
- Stage-2: `L = L_srcv + L_geo + L_frame + L_iFrame + 0.02 * L_viol`

`src/iggm_lightning/lightning_module.py` 现已支持按 epoch 切换两阶段：
- `current_epoch < stage1_epochs`：只训练结构项（关闭 `L_srcv`）；
- `current_epoch >= stage1_epochs`：开启 `L_srcv`。

同时二阶段支持按论文比例做 CDR 掩码混合采样：
- `CDR-H3 : CDR-H1 : CDR-H2 : all-CDR = 4 : 2 : 2 : 2`（可配置）。
- 采样依据来自 `.pt` 中的 `cdr_sequences`（1-based 残基索引）。

### 4.2 指标（`src/iggm_lightning/metrics.py`）

新增并接入验证/测试流程：
- AAR
- RMSD（优先 CDR H3，若缺失索引则退化为全局 CA）
- TM-Score
- GDT-TS
- DockQ
- SR（DockQ > 0.23）

checkpoint 选择改为：
- `monitor = val/tm_score`
- `mode = max`

---

## 5. 训练 / 验证 / 测试

1）按需在 YAML 中填写：
- `model.ppi_ckpt`（可空，空则自动下载官方PPI预训练）
- `model.design_ckpt`（可空，空则随机初始化DesignModel）
- `model.igso3_buffer`（可选）
- `wandb.api_key`（可选）

2）启动：

```bash
python scripts/train_iggm_lightning.py --config config/train_lightning.yaml

python src/train_iggm_lightning.py --config config/train_0306.yaml
```

3）可通过 CLI 临时覆盖：

```bash
python scripts/train_iggm_lightning.py \
  --config config/train_lightning.yaml \
  --max_epochs 5 \
  --run_name iggm-debug \
  --no_wandb
```

---

## 6. 与原仓库兼容性

- 推理入口 `design.py` 不受影响。
- 训练代码集中于 `src/iggm_lightning/` 与 `scripts/train_iggm_lightning.py`。
- 仅新增训练数据抽取、评估指标和 YAML 化配置能力。
