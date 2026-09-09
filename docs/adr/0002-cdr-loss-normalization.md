# CDR 损失只对坐标齐次项做 cdr_scale² 归一化

CDR 头输出的是无量纲的 `x0_norm`，由 `fr_cdr_blocks.py:855` 的 `pred_x0_local = x0_norm * cdr_scale + cdr_mu` 变成埃。`_cdr_grouped_atom_mse` 除以 `cdr_scale²`，因此它恰好是 `x0_norm` 空间里的 MSE —— 这就是 EDM/Karras 预处理，作用是让回传到 `coord_head` 的梯度与数据尺度无关（`cdr_scale` 重标定时不必改学习率），并让零初始化的 `x0_norm=0` 成为"预测均值"这个有定义的起点。

**关键点是 `cdr_scale²` 出现在梯度比里，不只在损失值里。** 令 `u = x0_norm`、`x = s·u`：

```
d(loss_cdr)/du  ~ err_phys / s          d(loss_bond)/du ~ s · bond_err
|g_bond| / |g_cdr| = s² · bond_err / err_phys
```

`s=6` 即 36 倍放大。2026-09-01 实测：`bond_weight=1.0` 且 bond 用物理 Å² 时，C=O 键长误差从 0.218 Å 改善到 0.036 Å，但 CDR 局部误差从 0.275 劣化到 1.072 Å、拟合样本 `aar_cdr` 从 0.83 崩到 0.11。损失值给不出任何预警（bond 当时是 0.001，比 `loss_cdr` 还小）。

**采取的规则：只对坐标齐次项做 `/s²`。** 键长是坐标的二次齐次函数，`d_norm = d/s` 给出 `(d_norm - d*_norm)² = (d - d*)²/s²`，所以除 `s²` 是精确的单位换算，不是调参。改后实测通过：局部误差 0.272/0.294/0.360 Å（三列均优于无 bond 的基线）、`aar_cdr` 0.9362、C=O 0.0558 Å，且 `inflated(slot2)` 类错误清零。代价是 FR 吸收率从 0.412 轻微降到 0.395（仍在 0.35 线上）。

**这条规则不能推广。** `loss_smooth_lddt` 的 sigmoid 阈值 0.5/1/2/4 和 `cutoff=15` 是写死的埃，把输入换成归一化单位等于把阈值悄悄改成 3/6/12/24 Å，那不再是 lDDT；`loss_seq` 根本不是坐标的函数。这两项非齐次，不存在正确的 `s` 幂次，**只能按实测梯度范数定权**（`lightning_module._log_grad_balance`）。实测教训：`smooth_lddt` 的梯度比是 0.416（`w_balanced≈2.4`），而按损失值估算会得出"20×"——方向相反，因为它有界且饱和，值大而梯度小。

**逃生口**：若将来把 lDDT 的阈值也改成随 `cdr_scale` 缩放的形式，它就变成齐次项、可以纳入 `/s²`；但那样它衡量的就不是标准 lDDT，跨文献不可比。

**注意**：`_log_grad_balance` 必须在 `coord_head` 离开零初始化之后取样（当前 step 50）。step 0 时 `x0_norm≡0`，所有原子塌在 `cdr_mu` 一点，原子间距离全为 0，任何距离类项的梯度恒为 0。