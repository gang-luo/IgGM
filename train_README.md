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

训练阶段若提供 `train_prot_cluster.txt`，每个 epoch 会执行：
- 对每个 cluster 随机抽取 1 个样本
- 使用抽样后的样本集合进行该 epoch 训练

满足论文描述的“每个 epoch 从每个 cluster 随机采样一个样本”。

---

## 4. 损失与评估指标

### 4.1 损失（`src/iggm_lightning/losses.py`）

- Stage-1: `L = L_geo + L_frame + L_iFrame + 0.02 * L_viol`
- Stage-2: `L = L_srcv + L_geo + L_frame + L_iFrame + 0.02 * L_viol`

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
