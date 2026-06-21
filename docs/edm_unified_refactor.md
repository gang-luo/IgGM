# EDM 统一架构重构说明

针对 FR 刚体（trsl/rota）与 CDR 全原子三个模态的去噪参数化做了统一，消除此前 trsl/rota 训练震荡不收敛的问题。前向加噪（`xt = x0 + σ·ε`、IGSO3 旋转加噪）与 dataloader 未改动，仍是合法扩散模型。

## 第一性原理诊断（结论）

- 单样本固定 seed 可完美过拟合，但单样本随机 seed 时 trsl 卡 0–1、rota 不收敛 → 瓶颈在**参数化**，非容量/数据量。
- 旧 trsl 损失为 F 空间 MSE（等价 `1/c_out²` 加权），强迫浅层 MLP 在低 σ 复现 `−σd/σ·u` 的发散乘性增益，结构上做不到 → 撞墙。
- 旧 rota 用残差/复合（`delta @ rota_xt`），要求 MLP 在整个 SO(3) 上做矩阵复合（BCH），拼接式 MLP 无法表达 → 坍缩到不旋转。

## 统一方案

| 模态 | 参数化 | 损失 |
|------|--------|------|
| trsl（欧氏） | EDM x0-prediction：`c_skip·xt_c + c_out·F_θ + μ` | x0 空间归一化 MSE（÷σd²，均匀加权） |
| CDR（欧氏） | EDM x0-prediction（原已是该配方） | x0 空间归一化 MSE（÷cdr_scale²） |
| rota（SO(3)） | 直接预测干净帧（6D + Gram-Schmidt），无复合 | 测地损失 |

要点：SO(3) 上无高斯 EDM 预条件，clean-frame 预测即群上的 x0-prediction。三模态损失均为 σ 无关的 x0/测地空间，不再有 `1/c_out²` 奇异目标。

## 代码改动

- `IgGM/model/arch/core/module/fr_cdr_blocks.py`：`FRBranch.forward` 固化为 trsl=EDM 组装、rota=clean-frame；删除未用的 `_axis_angle_to_matrix`、注释 x0 块、未用参数 `trsl_scale`/`fr_sigma_rota` 及注释返回键。
- `IgGM/model/arch/core/module/structure_module.py`：FR 调用同步去掉 `trsl_scale`/`fr_sigma_rota`；移除随之失效的 `fr_sigma_rota`/`trsl_scale`/`cdr_scale` 解包。
- `src/iggm_lightning/losses.py`：启用 CDR x0 空间损失（逐层线性加权 `cdr_all_atom_weight`）；移除低 σ 偏置 `weight_factor`，CDR 用固定权重；`_backbone_mse_layer` 恢复 `2·rota + trsl`，删除注释旧实现。
- `src/iggm_lightning/lightning_module.py`：移除已删除的 `weight_factor` 日志项。

## 验证

- 四个改动文件 `py_compile` 通过；FR 调用与 `_cdr_all_atom_mse` 输入输出对齐。
- 建议先单样本重跑确认 trsl/rota/CDR 三者均收敛到低位，再上 200 样本规模化。
