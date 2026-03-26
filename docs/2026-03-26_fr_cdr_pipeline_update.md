# 2026-03-26 FR/CDR 同步扩散实现更新（基于训练链路复核）

## 1. 先验复核结论（数据->训练->模型）

1. 数据侧由 `prepare_data_fromzip.py` 产出可训练样本，并依赖 `build_antibody_region_metadata` 固定生成 loop 索引/掩码张量。  
2. 训练侧 `ProcessedSabdabDataModule` 在 `payload['prot_data_curr']` 中直接注入这些 region metadata，并在 `__getitem__` 随机采样扩散步 `idx_step`。  
3. `train_iggm_lightning.py` 将 `idx_step`、`prot_data_curr` 交给 `Diffuser`，得到 `cord-p` 与 `noisy_loop_local_coords` 等监督字段后送入 `DesignModel`/`StructureModule`。

因此，`StructureModule(fr_cdr_sync)` 必须正确处理**非 batch 形式**的 loop metadata（典型 `[N_loop,...]`），并与混合精度训练 dtype 保持一致。

---

## 2. FR 分支更新（本次重点）

### 2.1 问题
旧改造中 FR 刚体头只看 `sfea_tns + fr_mask`，没有融合 `sfea_tns_init / encd_tns / quat_tns / trsl_tns`，且刚体坐标变换在 `StructureModule` 外部执行，信息流不完整。

### 2.2 新实现
`FRRigidHead.forward` 改为显式输入：
- `sfea_tns`
- `sfea_tns_init`
- `encd_tns`
- `quat_tns`
- `trsl_tns`
- `fr_mask`
- `fr_base_coords_global`

流程：
1. residue-level 融合特征（含 time/position 编码和 frame state）。
2. FR mask 池化得到全局 FR token。
3. 预测刚体 `quat/trsl/rota`。
4. **在 head 内部直接应用刚体到 FR 坐标**，输出 `updated_coords`。
5. 输出 `delta_sfea` 反馈到 `sfea_tns`，供 CDR 分支使用。

这使 FR 分支逻辑更接近 `fa` 的“状态+几何联合更新”范式。

---

## 3. CDR 分支更新（本次重点）

### 3.1 问题
旧版把 timestep 仅当外部 embedding 信号注入，且 token/atom 融合过弱。

### 3.2 新实现（参考 BoltzGen 思路做轻量整合）
`CDRLoopHead` 改为 token/atom 两级去噪：

1. **single conditioning**
   - 输入 `loop_sfea + loop_encd + loop_type + local_pos`
   - 得到 `token_cond`

2. **atom encoder**
   - 输入 `x_t`, `self_cond`, `delta(x_t-self_cond)`, `token_cond`, `atom_mask`
   - 得到 `atom_feat`

3. **token trunk**
   - `token_cond + token_from_atom` 融合后更新 `token_feat`

4. **atom decoder + 坐标头**
   - `atom_feat + token_feat + x_t` -> `r_update`
   - `pred_x0_local = x_t + r_update`

5. **occupancy**
   - 从 `token_feat` 预测并保持 monotone/prefix logits。

其中 `loop_encd` 来自主干 `encd_tns` 的 loop gather，语义上对应扩散步/噪声强度编码。

---

## 4. StructureModule 同步闭环（每层）

1. IPA 更新 `sfea_tns`。  
2. FA 更新 `quat/trsl/angl`。  
3. 根据主状态重建 `fr_base_coords`。  
4. FRRigidHead 内部完成 FR 刚体去噪与 FR 坐标更新，并返回 `delta_sfea`。  
5. 基于 FR 更新坐标构建 loop anchor frame。  
6. gather `loop_sfea` 与 `loop_encd`。  
7. CDRLoopHead 做 loop-local 全原子去噪，输出 `pred_x0_local`。  
8. LoopStateTransition 生成下一层 `loop_xt_local`。  
9. local->global + FRCDRMerger 得到 `merged`。  
10. LoopFeatureFeedback scatter 回 `sfea_tns`。  

最终输出 `fr/cdr/merged`，并作为 loss / deploy 的统一接口。

---

## 5. 稳定性修复

1. **维度修复**：统一 batch 扩展逻辑，对 `[N_loop,...]` 自动扩展到 `[B,N_loop,...]`。  
2. **AMP 修复**：FR 坐标写回和 feedback scatter 前强制对齐目标 dtype，避免半精度 in-place 赋值错误。  
3. **索引修复**：feedback 仅对 `valid & idx>=0` 位置 scatter，避免 padded 索引污染。

---

## 6. 与当前加噪/损失的一致性

- 加噪仍由 `Diffuser._run_fr_cdr_sync` 提供：`clean_fr_reference`, `clean_loop_local_coords`, `noisy_loop_local_coords`。  
- 去噪后输出仍对齐 `IgGMPaperLoss` 中 FR/CDR/occupancy/seam/clash/merged 监督项。  
- 推理仍优先使用 `merged.coords`。
