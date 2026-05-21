# 2026-05-21 v4 pipeline re-audit and actionable fix

## 重新梳理后确认的关键点
1. FR 扰动 `trsl_xt` 在 `Diffuser` 中使用了 `trsl_orig + sigma*eps`，但同模块已有注释版实现与标准 VP 式是 `sqrt(alpha_bar)*trsl_orig + sigma*eps`。
2. 训练时 backbone 目标在 loss 端监督的是去噪到 `trsl_orig`，如果前向加噪分布定义不一致，会使不同时间步目标难以学习成统一映射。
3. FR 分支是多层迭代，但 backbone 损失只监督最后一层，早期层缺少直接约束，容易把误差压力集中到末层，表现为 `loss_rota/loss_trsl` 大幅震荡。

## 本次修复
- 修复 FR 平移加噪方程为 VP 一致形式：
  - `trsl_xt = sqrt(alpha_bar_trsl) * trsl_orig + sigma_trsl * eps`
- backbone loss 改为层间监督：
  - 对 `outputs['3d']['rota']` 全层平均
  - 对 `outputs['3d']['trsl']` 全层平均
  让每一层都学习“去噪朝向 x0”，提升迭代链条稳定性。

## 为什么这比继续调loss权重更本质
- 先保证 forward noising distribution 与 denoising objective 一致，再谈权重。
- 先保证多层迭代每层有梯度约束，再谈末层数值抖动。

## 建议的下一步检查（不改代码也能做）
1. 记录 `||trsl_xt - sqrt(alpha_bar)*trsl_orig|| / sigma_trsl` 的分布，应接近标准正态尺度。
2. 分层记录 `loss_trsl@layer_i`、`loss_rota@layer_i`，看是否单调改善。
3. 按 timestep 分桶记录 backbone loss（低/中/高噪声），检查是否仍被低噪声桶主导。
