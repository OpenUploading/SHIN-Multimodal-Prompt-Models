# SHIN：时间 CNN / 空间 SGFormer 双 Prompt 门控注入 CBraMod K/V

## 1. 本轮实验结论

本轮实现 MI 与 MA 两个任务。EEG 在进入 CBraMod 前统一执行物理微伏 `/100`，与 CBraMod 原论文预训练尺度一致。MI 固定训练 50 轮；MA 最大 50 轮，并按 Fusion validation accuracy 使用 patience=15 的早停。CBraMod 预训练主干始终保持冻结，只训练：

- 参数独立的 HbO、HbR 时序编码器与 SGFormer 图编码器；
- CBraMod 最后 4 个 CrissCross block 的空间/时间 K/V 交叉注意力适配器；
- 8 个零初始化标量门（4 层 × 空间/时间）；
- EEG/Fusion 共享的 official all-patch 分类头；
- fNIRS 辅助分类头。

CBraMod 官方源码与官方预训练权重均不修改。

## 2. 为什么使用 36 个 prompt tokens

fNIRS 每个 trial 表示为 36 节点图，每个节点包含 HbO、HbR 两条 100 点时间序列及三维探头几何信息。HbO 与 HbR 不再作为两通道送入同一个首层卷积，而是分别进入参数完全独立、结构相同的时序编码器：

`HbO [B×36,1,100] -> Conv1d -> Conv1d -> Pool -> 64维`

`HbR [B×36,1,100] -> Conv1d -> Conv1d -> Pool -> 64维`

随后按固定 `[HbO,HbR]` 顺序拼接为 128维节点时序特征，再与几何编码、节点身份编码相加并进入 SGFormer。两个分支不共享卷积参数，直到拼接前都不会混合。两条分支合计参数量与原联合时序编码器近似，便于公平比较。经过 SGFormer 后，保留 36 个节点表示，并投影为：

`P ∈ R^(B×36×200)`

本轮进一步把送入 CBraMod 的时间、空间 Prompt 分开：

- 时间 Prompt：`HbO独立CNN ⊕ HbR独立CNN -> shared projection -> [B,36,200]`；不包含几何、Node-ID 或 SGFormer；
- 空间 Prompt：`temporal concat + geometry + Node-ID -> SGFormer -> shared projection -> [B,36,200]`；
- 空间 CrissCross Attention 只使用空间 Prompt 的 K/V；
- 时间 CrissCross Attention 只使用时间 Prompt 的 K/V。

两路共用同一个 128→200 投影层，因此与上一轮“共享 SGFormer Prompt”保持相同参数量，性能差异主要对应 Prompt 内容拆分。

不能在进入交叉注意力前池化为单个 token。若 K/V 只有一个 token，attention softmax 恒为 1，Q 与 K 不再影响权重，机制会退化成门控广播 V。本实现使用 36 个节点 tokens，确保 EEG query 能对不同 fNIRS 图节点形成非平凡注意力分布。

图级均值 `mean(P, dim=node)` 只供 fNIRS 辅助分类头使用。

SGFormer 严格保留论文的两条并联路径：

- 一层、单头的全局线性注意力，在一次传播中建模任意 fNIRS 节点对；
- 一个浅层局部 GCN，继续使用 36 节点、97 边的解剖图结构。

全局注意力内部使用 Frobenius 归一化 Q/K，注意力残差权重 `beta=0.5`；最终输出为 `0.2 × global + 0.8 × local_GCN`。原先串联的两层 GCN 已删除。

## 3. 注入位置和张量流程

CBraMod 输入：

`EEG_uV [B,30,2000] -> EEG_uV / 100 -> [B,30,10,200]`

只在 0-based 第 8、9、10、11 层，也就是最后 4 个 CrissCross blocks 注入 prompt。

空间分支：

- EEG query：`Q_s [B×10,30,100]`
- fNIRS key/value：`K_s,V_s [B×10,36,100]`
- 输出：`CA_s [B,30,10,100]`

时间分支：

- EEG query：`Q_t [B×30,10,100]`
- fNIRS key/value：`K_t,V_t [B×30,36,100]`
- 输出：`CA_t [B,30,10,100]`

每个 block 的残差更新为：

`Y_s = SA_s(EEG) + tanh(alpha_s) × CA_s(Q_s,K_s,V_s)`

`Y_t = SA_t(EEG) + tanh(alpha_t) × CA_t(Q_t,K_t,V_t)`

空间和时间分别使用独立的 K/V 线性投影及交叉注意力。所有 `alpha_s`、`alpha_t` 精确初始化为 0。

## 4. 零门控的意义

初始门值为 0 时，交叉注意力更新完全关闭，所以 Fusion 路径在数学上与原始 EEG 路径一致。训练首先学习门值是否应放行 fNIRS 信息，再逐步让 K/V 适配器参与优化，避免随机初始化的多模态模块一开始破坏预训练 EEG 表征。

由于还有独立的 fNIRS 辅助分类损失，图编码器在第一个 step 也能得到梯度。

## 5. 数据和损失

- MI：左手 vs 右手；session 0/2/4。
- MA：心算 vs 静息；session 1/3/5。
- 受试者切分沿用既定协议，训练统计量不使用验证/测试数据。
- EEG：物理微伏、CAR、因果五阶 Butterworth 0.3–50 Hz，然后除以 100。
- fNIRS：OD、MBLL、HbO+HbR、0.01–0.1 Hz、-5 到 -2 秒基线。
- 事件按 `subject/task/session/trial/start` 严格对齐。
- 总损失：`L = L_eeg + L_fnirs + L_fusion`。
- best checkpoint：按 Fusion validation accuracy 选择。

## 6. 训练参数

- MI：固定 50 epochs，不早停
- MA：最大 50 epochs，Fusion validation accuracy 连续 15 轮无提升时早停
- Batch size：4
- Graph / adapter / head learning rate：均为 `1e-4`
- Weight decay：`1e-4`
- Seed：1
- CBraMod：全部 50 轮冻结，无解冻阶段
- Mixed precision：关闭（首轮先控制变量）

注意：共享 all-patch 分类头约 1.204 亿参数，是主要显存占用和过拟合风险来源；共享头避免 EEG/Fusion 各复制一份。若显存不足，优先把 batch size 从 4 降到 2，不改变其他参数。

## 7. 已完成的诊断

MI 与 MA 均已使用 train/val/test 各 1 名受试者完成小样本诊断：

- 每个 split 60 trials，标签 30/30；
- EEG 与 fNIRS 各 session 事件序列全部逐 trial 对齐；
- NaN 与 Inf 均为 0；
- CBraMod checkpoint 严格载入 211 个键，missing/unexpected 均为空；
- HbO、HbR 分支形状分别为 `[1,36,64]`，按 `[HbO,HbR]` 拼接为 `[1,36,128]`；
- 两个时序编码器共享参数数为 0，输出全部有限；
- 空间 Prompt 与时间 Prompt 的形状均为 `[1,36,200]`；
- 时间 Prompt 仅来自 HbO/HbR 独立 1D-CNN 拼接，空间 Prompt 来自 SGFormer；二者内容确实不同：MI/MA 平均绝对差分别为 `0.6011/0.6161`；
- 两路 Prompt 共用一个 128→200 投影层（26,056 参数），总参数量与上一轮共享 Prompt 版本相同：`126,120,036`；
- 初始 8 个门值全部为 0；
- 零门控时 EEG/Fusion logits 最大绝对差：0；
- 临时令门值为 0.01 后执行一次反向传播：108 个可训练张量均有有限梯度。

完整诊断位于：

- `local_diagnostic_output_split_temporal_cnn_spatial_sgformer_mi/diagnostics.json`（MI）
- `local_diagnostic_output_split_temporal_cnn_spatial_sgformer_ma/diagnostics.json`（MA）

## 8. 运行方式

先双击：

`RUN_DIAGNOSE.cmd`

该脚本依次检查 MI、MA，每个 split 只取 1 名受试者，不进行长训练。

诊断通过后双击：

`RUN_MI_MA_FROZEN50.cmd`

该脚本依次执行 MI、MA 正式 50 轮训练，输出到：

`<OUTPUT_DIR>`

每个实验目录会包含配置、环境、数据诊断、训练历史、最佳权重、trial 预测、受试者指标、summary 和实验记录。
