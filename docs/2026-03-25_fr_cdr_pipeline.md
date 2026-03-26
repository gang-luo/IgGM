# 2026-03-25 FR/CDR 同步扩散训练流水线说明

## 1. 数据准备（`data/prepare_data_fromzip.py`）

1. 从 SAbDab 原始元数据与结构文件构建样本：
   - 统一重排链并输出 `processed_pdb`（H/L/A）。
   - 生成 `sample.pt`，保存序列、链长、CDR 序列索引等信息。
2. 核心区域元数据由 `build_antibody_region_metadata` 生成：
   - `fr_mask / cdr_mask / loop_masks`
   - `loop_type_ids`
   - `loop_global_res_indices`
   - `loop_valid_res_mask / loop_atom_valid_mask`
   - `loop_occ_target / loop_true_len / loop_lmax`
3. 这些 metadata 在训练时通过 DataModule 注入到 `prot_data_curr`，成为扩散与结构模块的直接输入。

## 2. 训练输入构建（`src/iggm_lightning/data_module.py` + `Diffuser`）

### 2.1 DataModule
- 每次 `__getitem__` 随机采样 `step in [1, n_steps]`。
- payload 中 `prot_data_curr` 包含：
  - clean 结构：`seq/cord/cmsk`
  - 设计 mask：`mask_design`
  - 抗体/抗原信息：`asym_id, a-cord, a-cmsk, epitope, contact`
  - loop region metadata（上节）

### 2.2 加噪（`Diffuser._run_fr_cdr_sync`）
- **序列噪声**：按 transition matrix 得到 `seq-p`。
- **FR 噪声**：整块 FR 共享一个刚体扰动（旋转 + 平移）。
- **CDR 噪声**：
  1. 先按当前 clean anchor 构造 loop-local clean 坐标 `clean_loop_local_coords`。
  2. 在 local 坐标加高斯噪声得到 `noisy_loop_local_coords`。
  3. 再基于 noisy FR anchor 回装配到 global。
- 最终得到 `cord-p/cmsk-p` 与扩散监督字段：
  - `clean_fr_reference`
  - `clean_loop_local_coords`
  - `noisy_loop_local_coords`
  - `anchor_frame_meta`

## 3. 去噪主干与双分支同步（`StructureModule`）

`structure_mode=fr_cdr_sync` 时，每层执行：
1. IPA 更新 residue single state。
2. FrameAngleHead 更新 `quat/trsl/angl` 主状态。
3. 从主状态重建当前层全局底板坐标（full-atom）。
4. FRRigidHead 预测 FR 刚体去噪变换，得到 FR 当前坐标。
5. LoopFrameBuilder 基于 FR 当前坐标和左右 anchor 动态构建每个 loop 的 local frame。
6. gather loop residue `sfea`，与 `loop_xt_local + self-conditioning + timestep + type/pos/mask` 输入 CDRLoopHead。
7. CDRLoopHead 输出：
   - `pred_x0_local`
   - `pred_occupancy_logits`（保持 prefix/monotone）
8. LoopStateTransition：`loop_xt_local -> loop_xnext_local`。
9. local->global 并用 FRCDRMerger 合并为 `merged` 全局全原子坐标。
10. LoopFeatureFeedback 把 loop 几何更新回写 `sfea_tns`，进入下一层。

最终输出 `fr/cdr/merged`。

## 4. 关键张量与维度约定

- `loop_xt_local`: `[B, N_loop, Lmax, 14, 3]`
- `loop_self_cond_x0_local`: `[B, N_loop, Lmax, 14, 3]`
- `loop_frame_rota`: `[B, N_loop, 3, 3]`
- `loop_frame_trsl`: `[B, N_loop, 3]`
- `pred_x0_local`: `[B, N_loop, Lmax, 14, 3]`
- `pred_occupancy_logits`: `[B, N_loop, Lmax]`
- `merged.coords`: `[B, L, 14, 3]`

## 5. 损失（`src/iggm_lightning/losses.py`）

`loss_mode in {fr_cdr_boltz, fr_cdr_iggm, boltz_style, iggm_style}` 时：

1. **FR 损失**
   - FR 坐标 MSE（`pred_coords vs clean_fr_reference`）
   - 刚体旋转 MSE（`pred_rota vs target_rota`）
   - 刚体平移 MSE（`pred_trsl vs target_trsl`）
2. **CDR local 坐标损失**
   - `pred_local_coords vs clean_loop_local_coords`（atom mask 加权）
3. **Occupancy 损失**
   - `BCEWithLogits(pred_occupancy_logits, loop_occ_target)`（valid mask 加权）
4. **几何正则项**
   - seam endpoint loss
   - loop clash penalty
5. **merged 全局坐标损失**
   - `merged.coords vs cord-o`（cmsk 加权）

最后按风格权重汇总总损失。

## 6. 推理链路（`IgGM/deploy/ab_design.py`）

- 当 `fr_cdr` bundle 包含 `merged.coords` 时，优先使用该结果作为全局坐标。
- occupancy 仍用 prefix 逻辑转预测长度，用于导出时的 loop 截断与 mask 更新。
- 若无 merged，才走 legacy 的 local->global 重建分支。

## 7. 本轮修正点（针对上次实现问题）

1. 修正了 region metadata 的批维扩展逻辑：对 `[N_loop,...]` 输入自动扩展到 `[B,N_loop,...]`，避免维度错配。
2. 修正了混合精度下的赋值 dtype 问题：FR 刚体赋值和 loop feedback scatter 前统一按目标 tensor dtype 转换。
3. 修正了 loop feedback 的无效索引处理：仅对 `valid & idx>=0` 条目 scatter，避免 padded index 干扰。
