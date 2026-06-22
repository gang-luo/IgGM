# EDM 统一架构 + A1/A3 改进说明

活动训练树为 `src/iggm_lightning/`（`zz_dish/` 为旧拷贝，未改）。前向加噪与 dataloader 未改动，仍是合法扩散模型。

## 参数化结论（第一性原理）

EDM 组装 `pred = c_skip·xt + c_out·F_θ` 要求轻量 MLP 复现 `coeff(σ)·xt`（σ 与输入的乘性耦合 + 低 σ 的 1/c_out 奇异），拼接式 MLP 表达不了 → trsl/cdr 震荡；不带组装的 rota 完美。x0 空间损失只改梯度加权、不改网络须表达的函数，故无法绕开该障碍。

**统一为直接 x0-prediction**：trsl/cdr/rota 三模态均直接输出干净量（trsl=归一化 clean 平移，cdr=归一化 clean 局部坐标，rota=clean frame 6D）。保留 `c_in` 输入归一化（无害），永久弃用 `c_skip/c_out` 输出组装。

## 本次改动

- **A1 界面感知池化（fr_cdr_blocks.FRBranch）**：新增 `_interface_pool`，按抗体残基到抗原 CA 的 softmin 距离加权池化，得到 `iface_feat`（64 维），拼入 trsl/rota 两个头。让位姿头看到"在抗原何处对接"，而非仅抗体均值。无新增跨模块 plumbing（复用已有 antigen_mask/curr_coords）。
- **FR 清理**：trsl 头固化为直接 x0-prediction；删除已弃用的 `trsl_xt_centered`/`fr_c_skip`/`fr_c_out` 形参与注释组装块；structure_module 同步去掉对应 unpack 与传参。
- **A3 min-SNR 损失加权（losses）**：`IgGMLossConfig.use_snr_weight`（默认 False）、`snr_gamma=5.0`。开启后按噪声水平缩放总损失，平衡各 σ 梯度。默认关闭，不影响当前单样本过拟合。

## A2 / A4

- **A2 已撤销**：CDR→FR 信息流已由主干 `sfea_after_cdr → sfea_tns →（下一层）percpt_xt + fr_branch` 天然承载，且 `res_proj`+池化的表达力严格强于"先池化再线性投影"。显式 `cdr_fb_proj` 在信息上零增益、表达力上是子集、仅多一条冗余捷径梯度，故移除。
- **A4 训练期 self-conditioning（lightning_module）**：保留。`self_cond_prob=0.5`，50% step 先 no-grad 前向得 x0_hat，以 `inputs_addi`（step 全 0，keys=sfea/pfea/cord）回灌；另 50% 走 cold-start。**logt 通道已随序列预测删除**。

## 序列(AA)预测逻辑删除

结构层面已用虚拟原子(N/O 邻域计数)解码氨基酸，故删除残基类型预测分支：

- `model.py`：移除 `norm_aa`、`aa_pred`、`logt_tns_aa` 计算与 `outputs['1d']`；移除 self-conditioning 的 `logt` 通道及其 `linear-lt-sd` 投影；删除随之失效的 `RESD_NAMES_1C` import、`sfea_tns_st` 解包改 `_`、更新 docstring。
- `lightning_module.py`：A4 回灌字典去掉 `logt`。
- 保留 `da_pred`（2D 几何 cb/om/th/ph，非序列预测）。
- **deploy 未改**（`ab_design.py`/`base_designer.py` 仍引用 `outputs['1d']`）：按既定选择不动训练外路径；如需跑原始 deploy 序列解码需另行适配。

## A3 默认值修正

`use_snr_weight` 默认改回 **False**（曾被误置 True，导致单样本随机 timestep 下 loss 标量逐 step 抖动、trsl 峰值偏高——是加权重塑曲线，非拟合失败）。多样本阶段再设 True。

## 校验

- `fr_cdr_blocks.py` / `structure_module.py` / `losses.py` 均 `py_compile` 通过；FR 调用签名对齐，无残留 `fr_c_skip/fr_c_out`。
- 建议先单样本过拟合确认 trsl/cdr/rota 三者收敛（A1 应加速 trsl/位姿、A3 保持关闭），再上多样本并开启 A3。
