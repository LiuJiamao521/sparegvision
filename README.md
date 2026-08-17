# SpaRegVision

SpaRegVision研究**空间分辨的enhancer互补性**：给定一个gene及其候选enhancer集合，学习每个enhancer在不同空间位置的贡献，并重建该gene的空间表达图案。

第一阶段只解决：

```text
candidate enhancer set + gene active mask + tissue context
                         ↓
              predicted gene spatial map
                         +
          enhancer × spatial attribution maps
```

第一阶段不训练`regulatory-link probability`。现有peak–gene links、RVS和divergence score只用于候选集合构建、baseline或弱监督辅助，不作为因果真值。

## 核心假设

> Spatial multi-enhancer models can identify spatially complementary enhancer activity that cannot be resolved by pairwise correlation or a global additive model.

主benchmark为：

```text
Best single enhancer
        <
Global multi-enhancer linear model
        <
Spatial multi-enhancer attribution model
```

主要终点是held-out spatial domain上的重建性能和区域贡献恢复能力，而不是随机spot上的overall R²。

## 目录

- `docs/DESIGN.md`：科学问题、模型、loss、消融和验证方案；
- `configs/default.yaml`：首轮实验的冻结配置；
- `docs/RESULTS_SCHEMA.md`：所有运行结果的目录和表格规范；
- `results/`：运行产物的唯一根目录。

## 阶段边界

1. 传统空间指标作为可解释基线；
2. 当前主算法：learned pairwise representation + explicit multi-enhancer spatial decomposition；
3. 后续：获得可信实验标签后再研究causal regulatory-link prediction。

