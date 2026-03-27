# 本次 Codex 修改说明（精简版）

## 目标

围绕 FR/CDR 同步扩散训练链路做“可训练、可推理、输入输出一致”修复，并清理结构模块冗余调用。

## 主要修改

1. **修复 StructureModule 调用参数不匹配**
   - `FRRigidHead.forward`：`rmsk_vec_motf` 改为可选，避免调用端缺参导致 forward 断链。
   - `CDRLoopHead.forward`：`loop_self_cond_x0_local` 改为可选，默认使用 `loop_xt_local`。
   - `StructureModule.forward`：显式传入 `rmsk_vec_motf` 与 `loop_self_cond_x0_local`，确保 FR/CDR 子模块接口对齐。

2. **精简 StructureModule 内部辅助逻辑**
   - 将重复的 `loop_sfea/loop_encd` gather 逻辑整合为统一 `_gather_loop_features(...)`。
   - 删除未使用的 `LoopStateTransition` 依赖，减少同级冗余组件耦合。

3. **评估逻辑标准化改造**
   - `StructureMetrics` 改为直接基于外部库计算：
     - DockQ（同步 DockQ/FNAT/LRMS/iRMS）
     - TM-score/GDT-TS（tmtools）
   - SR 判定统一为 `DockQ >= 0.23`。
   - 新增六个环 `H1/H2/H3/L1/L2/L3` 的 AAR 与 RMSD 统计，并保留全局 AAR。

4. **Lightning 评估输入对齐**
   - 在 `validation/test` 中将 `cdr_sequences` 与 `sequence_lengths` 传入指标模块，保证六环指标可用。

## 结果

- 训练主链路（输入 -> 扩散加噪 -> 模型去噪 -> 损失 -> 反传）可连通。
- 推理/验证主链路与训练共用结构输出接口，评估指标输出更完整且更接近官方实现。
