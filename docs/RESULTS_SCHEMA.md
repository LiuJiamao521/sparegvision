# SpaRegVision结果目录规范

所有SpaRegVision运行结果必须写入`results/`；`plot/`仅保留历史分析产物。

```text
results/<dataset>/<run_id>/
├── config.resolved.yaml
├── manifest.json
├── splits/
├── checkpoints/
├── predictions/
├── attribution/
├── metrics/
├── figures/
└── logs/
```

`manifest.json`必须记录代码版本、输入校验信息、candidate规则、空间splits、normalization来源、seed/fold、模型参数量，以及test region的遮盖方式。

## 必需结果表

- `metrics/per_gene_per_fold.tsv`：逐gene、fold和model的held-out metrics；
- `metrics/baseline_comparison.tsv`：best-single、global-multi和spatial-multi的配对比较；
- `metrics/complementarity_scores.tsv`：additive gain、spatial decomposition gain、redundancy、complementarity和switching；
- `enhancer_evidence.tsv`：global concordance、regional concordance、specificity、spatial-shift q-value与enhancer class；
- `gene_complexity.tsv`：core/regional enhancer数量、domain diversity与gene-level complexity score；
- `attribution/enhancer_attribution_summary.tsv`：每个enhancer的贡献、unique coverage和稳定性；
- `attribution/regional_ablation.tsv`：gene × enhancer × domain的ablation delta；
- `attribution/set_ablation.tsv`：single、pair、full set及逐步删除曲线；
- `predictions/spatial_maps.npz`：prediction、residual和mask；
- `attribution/attribution_maps.npz`：原生attribution与contribution maps。

## 报告规则

- 所有模型必须在相同test domains上比较；
- best-single只能用training/validation选择；
- 同时报告逐gene结果和跨gene paired summary；
- attribution结论必须有regional ablation支持；
- 未通过simulation attribution recovery时，不解释真实数据attribution。

