# weMERFISH 空间调控差异分析算法说明（供其他 AI 生成算法图）

本文档总结当前项目中用于 weMERFISH 空间双组学分析的核心算法流程，目标是让其他 AI 能够据此绘制一张清晰的算法示意图。

适用范围：当前 `results/weMERFISH/` 中的全基因组结果，以及单基因 locus-longpdf 可视化流程。

## 1. 分析目标

核心目标不是寻找“开放性最高”的 enhancer，而是寻找：

- 与目标基因存在初始 link 关系的 candidate enhancers；
- 在空间上与基因表达模式存在结构化差异的 enhancers；
- 进一步把这种 enhancer-level 差异聚合到 gene-level，得到 genome-wide gene ranking。

一句话概括：

> 先用原始 gene-peak link 限定候选 enhancer，再用 gene-specific latent state 约束空间比较，最后用结构化差异分数衡量 enhancer 与 gene 的空间不一致性。

---

## 2. 输入数据

当前 weMERFISH 分析使用三类输入。

### 2.1 RNA 空间表达

文件：

- `weMERFISH_combined_C_6s_E1_rescaled_z.h5ad`

包含：

- measured RNA expression (`X`)
- imputed RNA expression (`obsm/X_imputed`)
- 空间坐标

### 2.2 ATAC 空间开放性

文件：

- `weMERFISH_spatial_ATAC_C_6s_E1.h5ad`

包含：

- peak × cell ATAC matrix
- peak genomic coordinates (`chrom/start/end`)
- 3D spatial coordinates (`spatial_rescaled_z`)

### 2.3 基因组注释

文件：

- `Danio_rerio.GRCz11.113.gtf`

用于提取：

- gene start/end
- strand
- TSS
- 全染色体所有 annotated TSS

---

## 3. 总体流程

可以把整个算法拆成 6 个模块。

1. gene-specific candidate peak collection
2. gene-specific latent state construction
3. positive gene-peak link filtering
4. enhancer-vs-gene spatial difference scoring
5. gene-level aggregation
6. single-gene locus visualization

建议其他 AI 在画图时，把这 6 个模块画成主干流程。

---

## 4. 模块 1：candidate peak collection

### 4.1 输入

- 一个目标基因 `g`
- gene 的 chromosome / TSS
- 全 genome peak coordinates
- 全 chromosome annotated TSS

### 4.2 规则

对每个基因，只保留满足以下规则的 ATAC peaks：

- 与目标基因位于同一条染色体
- peak center 距离目标 TSS 在 `±100 kb` 以内
- peak center 不落在任意 annotated TSS 的 `±2 kb` 范围内

### 4.3 输出

- 该基因的 candidate peaks 集合

### 4.4 图示建议

画成一个基因 TSS 居中，左右 `100 kb` 窗口，近 TSS exclusion zone 单独标灰。

---

## 5. 模块 2：gene-specific latent state construction

这是整个流程最核心的一步，因为后续 enhancer 比较都建立在 gene-specific latent mask 上。

### 5.1 输入

- 目标基因的 RNA expression across all spots/cells
- 每个 cell 的 3D spatial coordinates

### 5.2 步骤

#### Step 1: 取 gene raw expression

优先使用 measured RNA；若 measured 不存在，则退回 imputed RNA。

#### Step 2: min-max normalization

对该基因表达做：

- `x_norm = (x - min) / (max - min)`

目的：不同基因进入统一 0–1 标尺。

#### Step 3: two-state fitting

用两状态模型把表达分成：

- low state
- high state

初始化：

- 25% quantile
- 75% quantile

迭代方式：

- 按最近中心重新分配 spot
- 更新两个中心
- 进行约 25 轮迭代

高表达状态定义为 active。

输出：

- raw binary state mask

#### Step 4: spatial regularization in 3D

为了去掉孤立噪声点，在 3D 空间上进行 kNN majority voting：

- k = 8 neighbors（实际用 k+1 因为包含 self）
- 3 rounds
- majority threshold = 5

输出：

- final latent state mask

### 5.3 输出

- 每个基因一个 binary latent mask
- active fraction
- low/high state centers

### 5.4 图示建议

可以画成：

`gene expression -> normalize -> two-state split -> 3D spatial smoothing -> latent mask`

---

## 6. 模块 3：positive gene-peak link filtering

这一步的作用不是最终评分，而是把候选 enhancer 进一步筛成“与基因表达初步一致”的一组 peaks。

### 6.1 输入

- gene expression vector
- candidate peak ATAC matrix

### 6.2 统计量

对每个 peak 计算：

- Spearman correlation between gene expression and peak accessibility
- one-tailed positive test

实现方式：

- 先把 gene 和 peak 分别转 rank
- 再用 Pearson on ranks 实现 Spearman rho
- 再用 t-statistic 和 one-tailed p-value 评估正相关显著性

### 6.3 保留规则

保留：

- positive link
- `p < 0.01`

### 6.4 输出

- significant positive candidate peaks
- 这些 peaks 会进入后续空间差异评分

### 6.5 图示建议

可以画成：

`candidate peaks -> rank correlation test -> significant positive peaks`

---

## 7. 模块 4：enhancer-vs-gene spatial difference scoring

这是最终的 enhancer-level ranking 模块。

### 7.1 输入

对每个 retained enhancer：

- gene expression
- gene latent mask
- enhancer raw ATAC

### 7.2 构造 gene control template

首先定义基因控制模板：

- `control = gene_expression × latent_mask`

解释：

- 只在 gene active 的区域看 enhancer
- gene 不表达的区域，即使 enhancer 开放，也不应该被重点关注

### 7.3 构造 latent enhancer map

对 enhancer：

- `raw_enhancer = log1p(ATAC)`
- `latent_enhancer = raw_enhancer × latent_mask`

### 7.4 形状比较前标准化

为了比较空间“形状”而非绝对强度：

- gene control map 和 enhancer latent map 分别做自己的 `p99 clipping`
- 再各自 rescale 到 `[0, 1]`

记为：

- `x = normalized gene control`
- `y = normalized enhancer latent`

### 7.5 三种差异视图

然后计算 3 个空间差异视图。

#### View A: direct difference

- `diff = y - x`

#### View B: log fold-change

- `logFC = log2((y + eps) / (x + eps))`

#### View C: residual difference

先拟合：

- `y ~ x`

然后取：

- residual = observed enhancer - fitted enhancer

### 7.6 每个视图的结构化评分

对每种视图，提取 3 类信息：

1. difference magnitude
2. spatial autocorrelation (`Moran's I`)
3. hotspot concentration

其中 hotspot concentration 的实现为：

- 取绝对差异值最高的 top 10% 区域
- 在 3D kNN 图上做 connected component
- 取最大热点连通块占比

因此每个视图都会得到一个结构化分数：

- `diff_structure_score`
- `logfc_structure_score`
- `residual_structure_score`

### 7.7 主 enhancer divergence score

最后组合成总分：

- `difference_structure_score = 0.4 * diff_structure_score + 0.25 * logfc_structure_score + 0.35 * residual_structure_score`

解释：

- direct difference 权重最高
- residual 其次
- logFC 作为辅助项

### 7.8 附加量

另外还会记录：

- `latent_state_rho`：gene control 与 latent enhancer 的空间相关性

现在这个量主要用于可视化参考，不再作为中间主轨道显示。

### 7.9 输出

每个 enhancer 至少有以下结果：

- initial gene-peak rho
- latent_state_rho
- difference_structure_score
- genomic coordinate
- panel id (`E1`, `E2`, ...)

---

## 8. 模块 5：gene-level aggregation

### 8.1 输入

一个基因的多个 retained enhancers 的 `difference_structure_score`

### 8.2 先做 enhancer 数量过滤

只有当一个基因至少保留一定数量 enhancer 时，才进入 gene ranking。

当前设置：

- `MIN_ENHANCERS_FOR_RANK = 5`

### 8.3 gene-level summary

对该基因所有 enhancer score 汇总：

- median
- mean
- top3 mean
- max
- std

### 8.4 主 gene-level ranking score

当前全基因组 ranking 主要使用：

- `absolute_divergence_score = 0.6 * median_difference_structure_score + 0.4 * top3_mean_difference_structure_score`

解释：

- median 代表整体差异水平
- top3 mean 代表最强几个 enhancer 的差异强度
- 这个定义比“相对 top10% 比例”更稳定，更适合 cross-gene ranking

### 8.5 输出

两个全基因组结果表：

- gene-level genomewide table
- enhancer-level genomewide table

---

## 9. 模块 6：single-gene long-pdf visualization

这一步不是评分本身，而是解释结果给人看的专门输出。

### 9.1 图的组成

当前单基因 PDF 包括：

1. 基因组 axis
   - gene box
   - TSS arrow
   - enhancer boxes (`E1, E2, ...`)
2. diffS summary track
   - 每个 enhancer 一根细竖线
   - 顶端一个菱形 marker
   - x 顺序与下方 enhancer panel 一一对应
3. spatial panel rows
   - 第一行：raw RNA / raw enhancer ATAC
   - 第二行：latent RNA / latent enhancer ATAC

### 9.2 注意

当前中间的 diffS 轨道已经不再按基因组坐标对齐，而是按下方空间图列顺序对齐。这样读者可以从 diffS 直接往下对应具体 enhancer 空间图。

---

## 10. 建议其他 AI 生成算法图时的结构

如果让其他 AI 画“算法流程图”，建议画成 3 层结构。

### Layer 1: input layer

- spatial RNA
- spatial ATAC
- GTF annotation

### Layer 2: computation layer

按顺序画成：

1. target gene selection
2. ±100 kb candidate peak collection
3. TSS exclusion
4. gene latent state construction
5. positive gene-peak correlation filtering
6. latent enhancer masking
7. three spatial difference views
8. structured difference scoring
9. gene-level aggregation

### Layer 3: output layer

- enhancer-level divergence score
- genome-wide gene ranking
- single-gene locus PDF

---

## 11. 推荐给作图 AI 的一句话 prompt

如果要直接把这套方法交给其他 AI 画图，可以用下面这句作为 prompt 基础：

> Draw a clean methodological schematic for a spatial multi-omics pipeline that integrates spatial RNA, spatial ATAC and gene annotation. For each target gene, collect candidate peaks within ±100 kb while excluding ±2 kb around any annotated TSS, build a gene-specific latent active mask from RNA by min-max normalization, two-state fitting and 3D kNN majority smoothing, retain positively correlated gene-peak links, mask enhancer ATAC by the gene latent state, compare enhancer and gene control maps using difference, log fold-change and residual views, summarize each view by magnitude + Moran’s I + hotspot concentration, combine them into an enhancer spatial divergence score, and aggregate enhancer scores into a genome-wide gene divergence ranking. Show the single-gene output as a locus axis plus a diffS track plus raw/latent spatial maps.

---

## 12. 当前项目中最重要的 3 个脚本

如果其他 AI 需要把图和实现对应起来，最相关的是：

- `scripts/compute_gene_peak_links_wemerfish.py`
- `scripts/score_wemerfish_genomewide_gene_divergence.py`
- `scripts/plot_wemerfish_gene_locus_spatial_longpdf.py`

---

## 13. 文档用途声明

这份文档的用途是：

- 帮助其他 AI 生成算法图
- 帮助人快速理解当前 weMERFISH 分析逻辑
- 不是代码 API 文档，也不是逐函数说明
