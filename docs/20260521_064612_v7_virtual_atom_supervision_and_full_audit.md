# 2026-05-21 v7 virtual atom supervision + full pipeline audit

## 对你问题1的结论（atom14 虚拟原子）
你说的是对的：在本课题设定里，atom14 不只是“真实原子mask”，还承载了通过 N/O 邻域虚拟原子数量表征残基类型的信号。
因此 CDR 去噪分支应当允许对虚拟原子坐标进行更新与监督，而不是仅限真实存在原子。

## 本次实现策略（双mask解耦）
- `loop_atom_valid_mask`：保留为数据真实 atom-existence（用于物理真实性检查/可解释分析）。
- `loop_atom_supervise_mask`：新引入为“残基有效即14原子全开”的监督掩码（用于 CDR 局部去噪学习）。

这样避免了二选一冲突：
- 既保留真实 atom14 信息；
- 又满足虚拟原子需要参与去噪更新与监督的课题设定。

## 代码改动
1. `Diffuser._run_fr_cdr_sync`
   - 构建 `loop_atom_supervise_mask` 并用于 local 加噪、loop local 提取与重建。
   - 同时保留原始 `loop_atom_valid_mask`。
2. `DesignModel.__extract_region_metadata`
   - 透传 `loop_atom_supervise_mask` 到 structure 模块。
3. `StructureModule.forward`
   - CDR 分支输入 mask 改为 `loop_atom_supervise_mask`（若不存在则回退 `loop_atom_valid_mask`）。
4. `IgGMPaperLoss`
   - `loss_cdr` 使用 `loop_atom_supervise_mask`（回退兼容旧字段）。

## 对你问题2：继续深入后的潜在风险点（非loss视角）
1. **FR head 信息瓶颈**：全局 mean pooling 对复杂抗原条件可能表达不足。
2. **CDR head 零初始化启动慢**：`coord_head` 零初始化在早期易退化为近似 copy-noise。
3. **共享感知特征梯度冲突**：FR（全局SE3）与CDR（局部原子）目标耦合可能互相拉扯。
4. **加噪-去噪目标一致性**：已修 VP trsl；建议继续核对 rotation 左/右乘 convention 与 loss 使用的误差定义是否完全同构。
5. **优化权重时序偏置**：低噪声步梯度主导仍需按 timestep 分桶监控，而非仅看总体均值。

## 建议最小验证闭环
- 日志：`delta_trsl`, `delta_rota(log map)`, `pred_x0_local-xt_local` 按 timestep 分桶。
- A/B：`loop_atom_supervise_mask` on/off 对 seq-recovery、bond、backbone稳定性的影响。
- 梯度：共享主干处 FR vs CDR 梯度范数比例与夹角（冲突度）。
