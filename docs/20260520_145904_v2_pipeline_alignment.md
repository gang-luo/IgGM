# 2026-05-20 v2 pipeline alignment review

## 针对模块
- `IgGM/model/arch/design_model/model.py`
- `IgGM/model/arch/core/module/structure_module.py`
- `IgGM/model/arch/core/module/fr_cdr_blocks.py`

## 代码级流水梳理（加噪/去噪）
1. `Diffuser.run` 先构造全局 FR 噪声状态 `(rota_xt, trsl_xt)`，并构造 CDR 局部噪声坐标 `noisy_loop_local_coords`。
2. `DesignModel.__forward_impl` 组织特征后调用 `StructureModule.forward` 进行 3D 去噪。
3. `StructureModule.forward` 每层顺序：
   - `percpt_xt` 感知当前 `curr_coords`；
   - `FRBranch` 用 `(rota_xt,trsl_xt)` 与特征预测更新后的刚体；
   - `CDRFusionBlock` 在 FR 更新后重建锚点局部框架，预测 CDR `pred_x0_local` 并写回全局。
4. `losses.py` 对最终层输出监督：
   - backbone: `outputs[rota,trsl][-1]` 对 `anchor_frame_meta[rota_orig,trsl_orig]`
   - cdr: `loop_cords[-1]` 对 `clean_loop_local_coords`
   - bond/smooth_lddt: 在 merged 全局坐标上计算。

## 发现的关键架构错位
- 在 `StructureModule.forward` 中，`rota_xt/trsl_xt` 在多层循环里一直固定为初始噪声态，未使用上一层预测结果进行迭代更新。
- 这会导致：
  - FR 分支每层都在“同一个 x_t”上重复回归，而不是逐层去噪；
  - 与 CDR 分支“每层接收更新后 FR 坐标再建局部框架”的行为不一致；
  - 训练时 FR 目标梯度难以形成稳定的分层纠错链条，表现为 `loss_rota/loss_trsl` 长期震荡。

## 本次修复
- 在每层结束后执行：`rota_xt = rota`, `trsl_xt = trsl`，让下一层从当前估计继续 refinement。
- 同时修复 `sigma_t` 维度处理，支持标量/按样本输入，避免 `.view(1)` 在 batch>1 时的潜在错配。
- 对 `rota_xt/trsl_xt` 做 batch 维展开与 device/dtype 对齐，减少隐式广播副作用。

## 预期
- FR 去噪从“重复单步回归”变为“逐层迭代逼近 x0”；
- backbone 损失收敛更平滑，尤其是 translation 项。
