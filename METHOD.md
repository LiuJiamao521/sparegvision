# SpaRegVision主算法设计

## 1. 问题定义

对gene \(g\)，输入gene active mask \(M_g\)、tissue mask \(T\)、候选enhancer maps \(E_1,\ldots,E_{n_g}\)，以及训练阶段可见、测试空间块被遮盖的gene context。模型学习：

\[
\{E_1,\ldots,E_{n_g}\},M_g,T,G_g^{context}
\longrightarrow \hat G_g,A_g,
\]

其中 \(A_g\in\mathbb R^{n_g\times H\times W}\) 是显式的enhancer × spatial attribution tensor。模型必须支持不定数量的候选enhancers，并对输入集合的排列保持等变性。

## 2. 第一阶段边界

- 不预测`P(enhancer regulates gene)`；
- 不把correlation、RVS或divergence当作causal label；
- 不以随机spot split作为主要验证；
- 不用post-hoc Grad-CAM替代原生attribution；
- 不仅凭attention weight宣称生物学重要性；
- 不以overall R²提升作为空间互补性的唯一证据。

## 3. 数据与验证

每个gene首轮保留5–32个候选enhancers。超过上限时只依据训练区之外也可获得的link先验筛选，不能根据测试区RNA表现选择。

每个输入提供normalized activity、tissue/observation mask、可选spot density和sigma 0/1/2多尺度图。归一化参数只能从训练区域估计，组织外未观测值与组织内真实零值必须区分。

验证分三层：

1. within-slice spatial block CV；
2. leave-one-slice/sample-out；
3. cross-dataset transfer。

所有候选筛选、归一化和超参数选择都在训练fold内部完成。

## 4. Baselines

- B0 spatial mean/null predictor；
- B1 best single enhancer，且只能用training/validation选择；
- B2 global non-negative Elastic Net：\(\hat G(s)=b+\sum_iw_iE_i(s)\)；
- B3 hand-crafted SpaRegVision指标；
- B4 capacity-matched neural control，每个enhancer仍只有global weight。

B4用于判断收益是否真正来自spatial attribution，而不只是更大的模型容量。

## 5. 主模型：Spatial Attribution Set Network

### 5.1 Encoders与set interaction

Gene context encoder读取masked gene map、active mask和tissue context；shared enhancer encoder用完全共享的参数处理每个 \(E_i\)，同时保留global token和spatial feature map。

带padding mask的Set Transformer或permutation-equivariant attention处理enhancer tokens。模块输出必须保留enhancer维度，不能过早pool成单个global token。

### 5.2 Spatial attribution decoder

对每个enhancer解码一个logit map \(Q_i(s)\)，在enhancer维度做masked softmax，并额外保留background/residual通道：

\[
A_i(s)=softmax_i(Q(s)/\tau),\qquad \sum_iA_i(s)+A_0(s)=1.
\]

\(A_0\)表示候选enhancers无法解释的贡献，防止模型强行把所有RNA归因给候选集合。

### 5.3 Gene reconstruction

首版采用可解释的局部mixture：

\[
\hat G(s)=b(s)+\phi\left(\sum_iA_i(s)V_i(s)\right),
\]

其中 \(V_i(s)\) 是enhancer feature的非负局部变换。模型同时输出gene prediction、attribution maps、background map、enhancer contribution maps及reconstruction residual。

## 6. Loss

\[
\mathcal L=\lambda_{rec}\mathcal L_{Huber}
+\lambda_{ssim}\mathcal L_{SSIM}
+\lambda_{grad}\mathcal L_{gradient}
+\lambda_{tv}\mathcal L_{TV}(A)
+\lambda_{ent}\mathcal L_{entropy}
+\lambda_{stab}\mathcal L_{stability}.
\]

训练test block必须完全遮盖。TV只轻度鼓励区域连续；entropy项通过simulation选择方向与强度。对坐标微扰、信号噪声和候选集合置换，prediction与attribution应稳定。

## 7. 空间调控类型的操作性定义

以贡献图 \(C_i(s)=A_i(s)V_i(s)\) 为基础：

- redundant：贡献图高度重叠，single ablation可被另一个enhancer补偿，而joint ablation损失明显更大；
- complementary：不同enhancers具有高unique coverage，regional ablation只在各自domain造成显著损失；
- dominant：一个enhancer在多数active locations和domains中贡献最大；
- regional switching：\(argmax_i C_i(s)\)在连续空间domain之间稳定变化，并可跨fold复现。

这些结论不能仅由raw attention得出，必须由消融支持。

## 8. Ablation

Global enhancer ablation：

\[
\Delta_i=\mathcal L(\hat G_{-i},G)-\mathcal L(\hat G,G).
\]

Regional ablation：

\[
\Delta_{i,d}=\mathcal L_d(\hat G_{-i},G)-\mathcal L_d(\hat G,G).
\]

同时比较single、pair、full set和逐步删除曲线，以量化冗余、互补与饱和效应。

## 9. Simulation benchmark

在真实tissue mask和空间自相关噪声上模拟：

1. single dominant；
2. redundant；
3. complementary non-overlapping domains；
4. partially overlapping/regional switching；
5. unrelated enhancers与missing true factor。

模拟必须保存ground-truth attribution。模型若不能在simulation中区分redundant与complementary，不进入真实数据的生物学解释。

## 10. 评价指标

Prediction：held-out-domain R²、Pearson/Spearman、masked SSIM、hotspot Dice和gradient similarity。

Attribution：simulation attribution recovery、regional ablation specificity、跨fold/seed稳定性、permutation equivariance，以及unrelated enhancers的background allocation。

Complementarity：best-single到global-multi的gain、global-multi到spatial-multi的gain、unique contribution coverage、contribution overlap、regional switching和set-ablation curve。

所有指标按gene报告，再汇总paired effect size、bootstrap CI和多重检验校正结果；案例图不能替代全体统计。

## 11. 成功标准

当前算法需同时满足：

1. spatial model在held-out domains稳定优于best-single和global Elastic Net；
2. simulation中恢复complementary enhancer的已知贡献区域；
3. enhancer permutation不改变prediction，attribution随输入正确置换；
4. unrelated enhancers主要被background通道拒绝；
5. regional ablation与原生attribution一致；
6. 真实数据关键结果跨fold、seed和样本可复现。

只有预测提高但attribution不稳定时，只能声称非线性模型预测更好，不能声称发现spatial enhancer complementarity。

## 12. 实施里程碑

1. 冻结gene集合、candidate规则、spatial splits、baselines和五类simulation；
2. 实现shared encoder、set interaction、explicit attribution decoder和background channel；
3. 完成与best-single、Elastic Net、capacity-matched control的严格比较；
4. 通过simulation后再进行真实数据genome-wide ranking、调控类型分类和跨样本复现。

