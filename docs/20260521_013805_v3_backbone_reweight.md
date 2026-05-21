# 2026-05-21 v3 backbone reweight

## 问题复盘
在已修复 FR 迭代状态传递后，`loss_backbone`（特别是 `loss_trsl`）仍明显震荡，说明不只架构流问题，还存在训练目标权重动态过激。

## 代码级原因
- backbone loss 使用 `1/sigma^2` 型加权；在小噪声步（t 接近 0）时，权重会急剧增大。
- 即便 `sigma clamp_min(5e-2)`，`1/sigma^2` 上界仍可到 400，单个 batch 的低噪声样本会主导梯度，表现为 rota/trsl 震荡。

## 修改
- `loss_rota` 改为 `sq_rota * clamp(1/sigma_rota^2, max=64)`
- `loss_trsl` 改为 `sq_trsl * clamp(1/sigma_trsl^2, max=25)`

## 预期
- 降低低噪声步对 backbone 梯度的“硬主导”，让多时间步样本共同驱动优化；
- 缓解 `loss_trsl` 与 `loss_backbone` 的高频大幅抖动，提升收敛可控性。
