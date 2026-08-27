# fNIRS-T通用Prompt融合：Step 1与Step 2

本项目不再使用CBraMod专有的空间/时间CrissCross注入。EEG主干输出统一展平为300个token；fNIRS-T产生统一Prompt；仅在EEG主干最终隐藏层之后使用一次通用残差Cross-Attention。

## 公共融合接口

完整10秒EEG进入冻结CBraMod，得到：

\[
Z_E\in\mathbb{R}^{B\times30\times10\times200}
\rightarrow
\tilde Z_E\in\mathbb{R}^{B\times300\times200}.
\]

Prompt通过标准Cross-Attention注入：

\[
\Delta Z=\operatorname{MHA}(Q=\operatorname{LN}(\tilde Z_E),
K=\operatorname{LN}(P),V=\operatorname{LN}(P)),
\]

\[
Z_F=\tilde Z_E+\tanh(g)\operatorname{LN}(\Delta Z).
\]

门控 \(g\) 从0开始，因此初始Fusion和固定EEG参考严格相等。迁移到CodeBrain、LaBraM或其他Transformer时，只需把其隐藏状态适配为 \`[B,N,D]\`，无需定义空间/时间两套注入。

## Step 1：单一统一Prompt

fNIRS-T把HbO/HbR从第一层联合编码。36个通道token与32个重叠五节点局部patch token分别经过Transformer；局部patch均值广播到36个节点，与通道特征拼接，形成标准fNIRS-T输出：

\[
T_{joint}\in\mathbb{R}^{B\times36\times128}.
\]

经过 \`128→200\` 投影，得到36个统一Prompt：

\[
P=\operatorname{Proj}(T_{joint})\in\mathbb{R}^{B\times36\times200}.
\]

## Step 2：三源加权多Prompt

## Step 3：整体学习Context + 当前样本Prompt

- 不再单独拆分 Universal Prompt 与 Task-specific Prompt；在每个任务的一次整体训练中，共同学习 `8` 个静态 Context Token。
- 当前0～10秒fNIRS由fNIRS-T编码，只将36个Joint Token作为动态样本Prompt。
- 最终Prompt为 `[8个learned context ; 36个sample tokens]`，共44个Token，直接拼接后作为通用交叉注意力的K/V。
- Context Token对同一任务的所有trial共享；Sample Token随trial变化。
- 不使用Router、熵损失或三组Prompt竞争权重；fNIRS辅助头只读取动态样本Token，避免静态Context替代样本证据。
- CBraMod主干与任务对应的固定分类头保持冻结，输入协议继续使用EEG/fNIRS的0～10秒窗口。

### MA Step 2 / Step 3 多seed

运行 `RUN_MA_STEP2_STEP3_SEEDS2_3.cmd` 可补齐MA的seed=2、3。脚本保持seed=1固定EEG参考头和其余协议不变，只改变Prompt训练的随机种子；已存在 `summary.json` 的目标会自动跳过。

多Prompt不是复制同一个张量，而是保留fNIRS-T中三种不同信息：

1. \`channel\`：36个通道路径token，每个对应一个fNIRS节点；
2. \`patch\`：32个局部路径token，每个对应连续5节点的重叠局部区域；
3. \`joint\`：36个通道token与局部路径全局摘要的联合表示。

分别投影到200维后沿token维拼接：

\[
P_{cat}=[P_{channel};P_{patch};P_{joint}]
\in\mathbb{R}^{B\times104\times200}.
\]

一个小型路由器根据三路全局摘要为每个trial产生：

\[
\alpha=\operatorname{softmax}(R(X_F))\in\mathbb{R}^{B\times3}.
\]

不提前对Prompt求和，而是在Cross-Attention logits中加入每个来源的对数先验：

\[
A=\operatorname{softmax}
\left(\frac{QK^T}{\sqrt d}+\log\alpha_m-\log N_m\right).
\]

\`-log N_m\`消除36/32/36 token数量差异，保证初始每个来源的总先验质量均为1/3。路由器最后一层全零初始化，所以Step 2从严格均匀权重开始。使用小权重熵惩罚鼓励后期产生选择性，并记录均值、标准差、极值及有效Prompt数。

## 固定参考与训练

- 数据协议保持主报告：train sub-1–19，val sub-20–24，test sub-25–29；完整10秒输入。
- EEG物理微伏除以100。
- 直接加载已有seed=1任务特定EEG分类头；CBraMod和分类头全程冻结。
- 只训练fNIRS-T、Prompt投影/路由、通用Cross-Attention、门控和fNIRS辅助头。
- 损失为 \`Fusion CE + fNIRS CE + 路由熵项\`；不优化EEG分支。
- Step 1和Step 2使用完全相同的固定EEG参考，因此可以直接比较Fusion增益。

## 运行

默认顺序运行Step 1与Step 2的MI和MA：

\`\`\`bat
RUN_STEP1_STEP2_MI_MA.cmd
\`\`\`

也可以只运行其中一步：

\`\`\`bat
RUN_STEP1_STEP2_MI_MA.cmd step1
RUN_STEP1_STEP2_MI_MA.cmd step2
\`\`\`

结果写入：

\`<OUTPUT_DIR>\`
