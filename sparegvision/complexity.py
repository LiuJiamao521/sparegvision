from __future__ import annotations

import numpy as np
import pandas as pd


def _pearson(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.size < 8 or np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def spatial_domain_labels(shape, n_rows=2, n_cols=3):
    yy, xx = np.indices(shape)
    row = np.minimum(yy * n_rows // shape[0], n_rows - 1)
    col = np.minimum(xx * n_cols // shape[1], n_cols - 1)
    return row * n_cols + col


def enhancer_evidence_table(
    enhancer_maps,
    gene_map,
    tissue_mask,
    gene="gene",
    domain_labels=None,
    truth_classes=None,
    truth_domains=None,
    n_permutations=99,
    seed=0,
    core_threshold=0.65,
    regional_local_threshold=0.45,
    evidence_threshold=0.03,
    qvalue_threshold=0.10,
):
    """Quantify global versus localized enhancer-gene correspondence."""
    enhancer_maps = np.asarray(enhancer_maps)
    gene_map = np.asarray(gene_map)
    tissue_mask = np.asarray(tissue_mask, bool)
    if domain_labels is None:
        domain_labels = spatial_domain_labels(gene_map.shape)
    rows = []
    rng = np.random.default_rng(seed)
    domain_ids = sorted(np.unique(domain_labels[tissue_mask]))
    for i, enhancer in enumerate(enhancer_maps):
        global_corr = _pearson(enhancer[tissue_mask], gene_map[tissue_mask])
        local = []
        for domain_id in domain_ids:
            mask = tissue_mask & (domain_labels == domain_id)
            local.append((int(domain_id), _pearson(enhancer[mask], gene_map[mask]), int(mask.sum())))
        local_values = np.asarray([x[1] for x in local])
        best_index = int(np.argmax(local_values))
        best_domain, max_local, domain_size = local[best_index]
        median_other = float(np.median(np.delete(local_values, best_index))) if len(local) > 1 else 0.0
        local_advantage = max_local - global_corr
        regional_specificity = max_local - median_other
        evidence = (
            max(max_local, 0.0)
            * max(local_advantage, 0.0)
            * max(regional_specificity, 0.0)
        )
        null_maxima=[]
        for _ in range(n_permutations):
            shift_y=int(rng.integers(0,gene_map.shape[0]))
            shift_x=int(rng.integers(0,gene_map.shape[1]))
            shifted=np.roll(enhancer,(shift_y,shift_x),axis=(0,1))
            null_local=[]
            for domain_id in domain_ids:
                null_mask=tissue_mask & (domain_labels==domain_id)
                null_local.append(_pearson(shifted[null_mask],gene_map[null_mask]))
            null_maxima.append(max(null_local))
        permutation_pvalue=(1+sum(value>=max_local for value in null_maxima))/(n_permutations+1)
        if global_corr >= core_threshold:
            predicted_class = "core_concordant"
        elif max_local >= regional_local_threshold and evidence >= evidence_threshold:
            predicted_class = "regional_candidate"
        else:
            predicted_class = "unsupported"
        row = {
            "gene": gene,
            "enhancer_index": i,
            "global_concordance": global_corr,
            "max_regional_concordance": max_local,
            "best_domain": best_domain,
            "best_domain_size": domain_size,
            "local_advantage": local_advantage,
            "regional_specificity": regional_specificity,
            "regional_evidence_score": evidence,
            "permutation_pvalue": permutation_pvalue,
            "predicted_class": predicted_class,
        }
        if truth_classes is not None:
            row["truth_class"] = truth_classes[i]
        if truth_domains is not None:
            row["truth_domain"] = int(truth_domains[i])
        rows.append(row)
    result=pd.DataFrame(rows)
    order=np.argsort(result["permutation_pvalue"].to_numpy())
    ranked=result["permutation_pvalue"].to_numpy()[order]
    adjusted=np.minimum.accumulate((ranked*len(result)/np.arange(1,len(result)+1))[::-1])[::-1]
    qvalues=np.empty(len(result)); qvalues[order]=np.clip(adjusted,0,1)
    result["permutation_qvalue"]=qvalues
    regional=(result["global_concordance"]<core_threshold) & (result["max_regional_concordance"]>=regional_local_threshold) & (result["regional_evidence_score"]>=evidence_threshold) & (result["permutation_qvalue"]<=qvalue_threshold)
    result.loc[result["global_concordance"]>=core_threshold,"predicted_class"]="core_concordant"
    result.loc[(result["global_concordance"]<core_threshold) & ~regional,"predicted_class"]="unsupported"
    result.loc[regional,"predicted_class"]="regional_candidate"
    result["regional_evidence_score"]=result["regional_evidence_score"]*(1-result["permutation_qvalue"])
    return result


def gene_complexity_summary(evidence, gene="gene", top_k=5):
    regional = evidence[evidence["predicted_class"] == "regional_candidate"]
    top = regional.nlargest(top_k, "regional_evidence_score")
    counts = top["best_domain"].value_counts().to_numpy(dtype=float)
    if counts.size <= 1:
        domain_diversity = 0.0
    else:
        probabilities = counts / counts.sum()
        domain_diversity = float(-(probabilities * np.log(probabilities)).sum() / np.log(len(counts)))
    evidence_sum = float(top["regional_evidence_score"].sum())
    complexity = evidence_sum * np.log1p(len(regional)) * (1.0 + domain_diversity)
    return pd.DataFrame([{
        "gene": gene,
        "n_enhancers": len(evidence),
        "n_core": int((evidence["predicted_class"] == "core_concordant").sum()),
        "n_regional_candidates": len(regional),
        "n_unsupported": int((evidence["predicted_class"] == "unsupported").sum()),
        "n_supported_domains": int(regional["best_domain"].nunique()),
        "regional_evidence_topk_sum": evidence_sum,
        "domain_diversity": domain_diversity,
        "complexity_score": complexity,
    }])


def _column_correlations(matrix, target):
    matrix = np.asarray(matrix, dtype=float)
    target = np.asarray(target, dtype=float)
    matrix_centered = matrix - matrix.mean(axis=0, keepdims=True)
    target_centered = target - target.mean()
    numerator = (matrix_centered * target_centered[:, None]).sum(axis=0)
    denominator = np.sqrt((matrix_centered ** 2).sum(axis=0) * np.sum(target_centered ** 2))
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-12)


def enhancer_evidence_matrix(
    enhancer_values, gene_values, domain_labels, gene="gene", enhancer_names=None,
    core_threshold=0.35, regional_local_threshold=0.20, evidence_threshold=0.005,
):
    """Fast genome-wide screen on paired spots; significance is added in stage two."""
    enhancer_values = np.asarray(enhancer_values, dtype=float)
    gene_values = np.asarray(gene_values, dtype=float)
    domain_labels = np.asarray(domain_labels)
    global_corr = _column_correlations(enhancer_values, gene_values)
    domain_ids = np.unique(domain_labels)
    local = np.stack([_column_correlations(enhancer_values[domain_labels == domain],
        gene_values[domain_labels == domain]) for domain in domain_ids])
    best_position = np.argmax(local, axis=0)
    max_local = local[best_position, np.arange(local.shape[1])]
    best_domain = domain_ids[best_position]
    median_other = np.median(np.sort(local, axis=0)[:-1], axis=0)
    local_advantage = max_local - global_corr
    specificity = max_local - median_other
    score = np.maximum(max_local, 0) * np.maximum(local_advantage, 0) * np.maximum(specificity, 0)
    predicted = np.full(enhancer_values.shape[1], "unsupported", dtype=object)
    predicted[global_corr >= core_threshold] = "core_concordant"
    regional = ((global_corr < core_threshold) & (max_local >= regional_local_threshold)
                & (score >= evidence_threshold))
    predicted[regional] = "regional_candidate"
    names = enhancer_names if enhancer_names is not None else np.arange(enhancer_values.shape[1]).astype(str)
    return pd.DataFrame({"gene": gene, "enhancer": names,
        "enhancer_index": np.arange(enhancer_values.shape[1]),
        "global_concordance": global_corr, "max_regional_concordance": max_local,
        "best_domain": best_domain.astype(int), "local_advantage": local_advantage,
        "regional_specificity": specificity, "regional_evidence_score": score,
        "predicted_class": predicted, "screen_only": True})
