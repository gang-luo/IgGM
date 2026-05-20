# 2026-05-20 v1 loss stability fix

## 入口
- 训练入口：`python src/train_iggm_lightning.py --config config/train_0518_signleGPU.yaml`

## 关键问题（基于代码路径）
1. backbone 平移/旋转损失的噪声标度使用不一致：旋转分支未按 `sigma_rota` 归一化，平移分支以全局 MSE 后再除 `sigma_trsl^2`，导致量纲和权重耦合不稳定。
2. bond loss 在有效掩码为空时直接调用 `F.mse_loss(..., reduction='mean')`，会产生 NaN/不稳定梯度。
3. loss 文件中含有硬编码磁盘写入调试逻辑（每 50 step 保存到绝对路径），会引入额外 IO 干扰并污染训练流程。

## 修改
- 将 `loss_rota` 改为 SO(3) 切空间平方误差后按 `sigma_rota^2` 归一化。
- 将 `loss_trsl` 改为逐样本平方位移按 `sigma_trsl^2` 归一化后求均值。
- 对 `sigma_rota/sigma_trsl` 加下限 clamp（5e-2）防止极小噪声导致梯度爆炸。
- `bond loss` 增加空掩码分支，空集返回 0 张量，避免 NaN。
- 删除训练时周期性 `torch.save` 的硬编码调试输出。

## 预期影响
- 缓解 `loss_backbone` 在 1.2~1.8 区间长时间震荡，尤其是 `loss_trsl` 高位振荡（~12）。
- 降低 `loss_bond` 在 0.1~3 间随机跳变，优先消除“空掩码均值”带来的非物理波动。
