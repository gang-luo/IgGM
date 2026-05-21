# 2026-05-21 v6 潜在问题再分析（继续排查）

## 本轮新增关键发现

### 问题 A：CDR atom14 有效原子 mask 被“全开残基原子”覆盖
- 在 `Diffuser._run_fr_cdr_sync` 里，`loop_atom_valid_mask` 曾被替换为 `loop_valid_res_mask.unsqueeze(-1).expand_as(...)`。
- 在 `StructureModule.forward -> cdr_fusion_block` 调用时，也再次把 atom mask 改成“只要残基有效就全原子有效”。

#### 为什么是高风险
1. **与 atom14 表示假设冲突**：Gly、缺失侧链、虚拟位点的真实 atom existence 被抹平。
2. **局部去噪目标失真**：CDR head 会在本不应存在的原子位点上学习坐标更新。
3. **几何损失噪声升高**：bond / smooth lddt 在掩码层面受到污染，间接影响 backbone 分支的梯度稳定。

#### 本次修复
- 保留并贯穿原始 `loop_atom_valid_mask`，不再“全开”。

---

## 仍建议重点检查的潜在问题（未在本次直接改）

### 问题 B：FR 头表达瓶颈
- `FRBranch` 的刚体更新来自抗体全局 mean pooling，可能不足以表达复杂抗原条件下的 SE(3) 修正。
- 若 `delta_trsl`/`delta_rota` 长期接近 0，说明 head 学习受限而非 loss 权重问题。

### 问题 C：CDR 头初期近似“复制 x_t”
- `coord_head` 全零初始化 + EDM `c_skip` 结构在初期会让 `pred_x0_local ≈ c_skip*x_t`。
- 如果 FR 头同时不稳定，系统会在早期形成“FR 抖动 + CDR 保守复制”的耦合状态。

### 问题 D：FR/CDR 共享感知特征的梯度冲突
- 全局刚体任务与局部原子任务可能在 shared representation 上方向冲突。
- 表象：某分支 loss 降而另一分支持续振荡。

## 建议验证指标（短跑即可）
1. `valid_atom_ratio_loop = loop_atom_valid_mask.float().mean()`（确保不再异常接近1.0）。
2. `||delta_trsl||`、`||log(R_delta)||` 随 timestep 分桶统计。
3. `||pred_x0_local - x_t_local||` / sigma 分桶统计。
4. FR vs CDR 分支梯度范数比值（共享主干输入处）。
