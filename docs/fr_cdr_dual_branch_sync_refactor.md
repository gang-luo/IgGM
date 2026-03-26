# FR刚体扩散 + CDR Anchor-Local全原子扩散：一次性结构性重构说明

## 1) 旧实现的根本问题

旧 `fr_cdr_sync` 路径虽然有 FR 和 CDR 两个头，但 CDR 本质仍是 `sfea -> 一次性local坐标`，没有显式 loop diffusion state：
- 没有把 `loop_xt_local` 作为分层递推状态。
- 没有显式 self-conditioning `loop_self_cond_x0_local`。
- 没有 timestep embedding 接入 CDR 头。
- `StructureModule` 的 fr_cdr 分支没有保持 `quat/trsl/angl` 主链逐层更新并与 loop 分支同步闭环。
- `_build_fr_cdr_outputs` 只做打包，未承担 local/global 变换和 merged 正式输出组织。

## 2) 新双 branch 同步扩散设计

当前改造后，`StructureModule.forward(structure_mode=fr_cdr_sync)` 每层执行：
1. IPA + FrameAngleHead 更新主干 `quat/trsl/angl`。
2. 从主干参数重建当前层全局底板坐标（FR base）。
3. FRRigidHead 预测刚体变换并得到当前层 FR/global 底板。
4. LoopFrameBuilder 基于当前 FR/global 动态构建 loop anchor frame。
5. gather `loop_sfea` + `loop_xt_local` + `loop_self_cond_x0_local` + timestep + type/pos/masks 进入 CDRLoopHead。
6. CDRLoopHead 输出 `pred_x0_local` 与 monotone occupancy logits。
7. LoopStateTransition 用 `x_t + x0_pred + alpha_prev/curr` 生成下一层 `loop_xt_local`。
8. local->global 装配并通过 FRCDRMerger 得到 `merged` 全局全原子。
9. LoopFeatureFeedback 将 loop 去噪信号 scatter 回全链 `sfea_tns`，供下一层主干继续使用。

## 3) 状态定义

### 全局主状态
- `quat_tns`
- `trsl_tns`
- `angl_tns`

### CDR loop state
- `loop_xt_local`: `[B, N_loop, Lmax, 14, 3]`
- `loop_self_cond_x0_local`: `[B, N_loop, Lmax, 14, 3]`
- `loop_frame_rota`: `[B, N_loop, 3, 3]`
- `loop_frame_trsl`: `[B, N_loop, 3]`
- occupancy / valid / atom mask / indices / type ids

## 4) 模块职责与I/O

- `loop_frame_builder.py`：从 FR/global 坐标和 anchor index 构建每个 loop frame。
- `loop_coord_converter.py`：`global_to_local_loop_coords` 与 `local_to_global_loop_coords`。
- `loop_state_encoder.py`：显式编码 `xt/selfcond/delta + timestep + loop type/pos/mask`。
- `loop_geom_updater.py`：融合 `loop_sfea + loop_state_feat` 输出 `pred_x0_local` 和 occupancy。
- `loop_state_transition.py`：x0-parameterized 递推 `loop_xnext_local`。
- `loop_feature_feedback.py`：loop denoise 信息回写 `sfea_tns`。
- `fr_cdr_merger.py`：FR底板 + CDR覆盖，得到 `merged_coords_global`。

## 5) 关键shape

- `loop_sfea`: `[B, N_loop, Lmax, C_s]`
- `loop_state_feat`: `[B, N_loop, Lmax, C_loop]`
- `pred_x0_local`: `[B, N_loop, Lmax, 14, 3]`
- `pred_occupancy_logits`: `[B, N_loop, Lmax]`
- `pred_loop_global`: `[B, N_loop, Lmax, 14, 3]`
- `merged.coords`: `[B, L, 14, 3]`

## 6) local frame定义一致性

训练 target 仍来自扩散器侧 `clean_loop_local_coords`，运行期 frame 统一由 anchor 定义；装配使用相同 anchor frame 反变换，保证 target / 推理装配使用同一局部坐标系定义。

## 7) 修改文件清单

- `IgGM/model/arch/core/module/structure_module.py`
- `IgGM/model/arch/core/module/cdr_loop_head.py`
- `IgGM/model/arch/core/module/loop_frame_builder.py`
- `IgGM/model/arch/core/module/loop_coord_converter.py`
- `IgGM/model/arch/core/module/loop_state_encoder.py`
- `IgGM/model/arch/core/module/loop_geom_updater.py`
- `IgGM/model/arch/core/module/loop_state_transition.py`
- `IgGM/model/arch/core/module/loop_feature_feedback.py`
- `IgGM/model/arch/core/module/fr_cdr_merger.py`
- `IgGM/model/arch/design_model/model.py`
- `IgGM/deploy/ab_design.py`

## 8) 为什么这是最短路径闭环实现

- 不推翻 Evoformer 与 FrameAngle 主链，仅在 `fr_cdr_sync` 路径插入同步 loop diffusion pipeline。
- CDR 直接用 local x0-pred 递推，不引入独立长度模型。
- 通过最小必要新增模块完成 frame构建、状态递推、反馈与装配，形成单次 forward 端到端闭环。
