# IgGM Lightning 训练与测试说明

本仓库新增了基于 `pytorch-lightning` 的训练闭环，保持原始 IgGM 推理与特征构建逻辑不变，并补充论文目标函数拆分。

## 1. 配置管理（YAML）

新增统一配置文件：`config/train_lightning.yaml`。

该文件集中保存：
- 模型超参数与权重路径（`model`）
- 数据路径与 DataLoader 超参数（`data`）
- 训练器参数（`trainer`，对应 `pl.Trainer`）
- 优化器参数（`optimizer`）
- wandb 信息（`wandb`，含 API Key / project / run name）
- 运行控制参数（`runtime`，输出目录、resume、run_test、seed）

训练主函数 `scripts/train_iggm_lightning.py` 默认读取该 YAML，并允许命令行参数覆盖。

## 2. 数据来源与处理

- 官方清洗结果：`data/sabdab/processed/sabdab/metadata.json`
- 对应结构文件：`data/sabdab/metadata/*.pdb`
- 对应 FASTA 参考：`data/sabdab/processed/fasta/*.fasta`

Lightning 数据模块：`src/iggm_lightning/data_module.py`

流程：
1. 优先从 `metadata.json` 读取 entry。
2. 若无 entry，回退读取 `data/sabdab/metadata/sabdab.tsv`。
3. 用 `pdb_id` 在 `metadata/` 下定位 `pdb`。
4. 通过 `IgGM.data.convert_to_example_format.convert_entry_to_sample` 重建单样本。
5. 自动划分 train/val/test（默认 8/1/1）。

## 3. 损失函数实现

实现文件：`src/iggm_lightning/losses.py`

按论文形式实现总损失：

- 第一阶段：
  - `L = L_geo + L_frame + L_iFrame + 0.02 * L_viol`
- 第二阶段（默认开启序列恢复项）：
  - `L = L_srcv + L_geo + L_frame + L_iFrame + 0.02 * L_viol`

说明：
- `L_geo`：由坐标诱导的几何距离监督。
- `L_frame`：对结构模块各层输出进行带 `gamma` 权重的 frame 损失。
- `L_iFrame`：在界面区域的 frame 监督。
- `L_viol`：bond length / bond angle / clash 约束，并跳过重链末端与轻链首端之间的肽键惩罚。
- `L_srcv`：序列交叉熵恢复损失（stage-2）。

## 4. Lightning 训练/验证/测试闭环

入口脚本：`scripts/train_iggm_lightning.py`

关键能力：
- 训练、验证、测试完整闭环。
- 自动保存 `best` 和 `last` checkpoint。
- `--resume` 支持异常中断后从 `last.ckpt` 继续训练。
- 默认记录 `val/loss` 作为最优模型选择指标。
- 集成 wandb 记录 loss 曲线；可用 `--no_wandb` 切 CSV logger。

## 5. 运行示例

先编辑 `config/train_lightning.yaml` 中的权重路径与 wandb 信息，然后执行：

```bash
python scripts/train_iggm_lightning.py --config config/train_lightning.yaml
```

通过 CLI 临时覆盖单个参数示例：

```bash
python scripts/train_iggm_lightning.py \
  --config config/train_lightning.yaml \
  --max_epochs 5 \
  --run_name iggm-debug \
  --no_wandb
```

## 6. 与原始仓库的一致性

- 模型正向仍基于 `DesignModel.featurize + diffuser.run + DesignModel.forward`。
- 不改动推理入口 `design.py` 与官方样例使用方式（见 `README.md`）。
- 新增训练代码位于 `src/iggm_lightning/` 与 `scripts/train_iggm_lightning.py`，与原部署逻辑解耦。
