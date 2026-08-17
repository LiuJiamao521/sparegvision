from pathlib import Path
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import json
import re
import h5py
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from scipy.stats import rankdata, t as student_t
from sklearn.neighbors import NearestNeighbors

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sparegvision.gsps import moran_i

ROOT = Path(__file__).resolve().parents[1]
RNA_PATH = Path('/cluster3/labData/jiamao/SpaRegVision/weMERFISH/weMERFISH_combined_C_6s_E1_rescaled_z.h5ad')
ATAC_PATH = Path('/cluster3/labData/jiamao/SpaRegVision/weMERFISH/weMERFISH_spatial_ATAC_C_6s_E1.h5ad')
GTF_PATH = Path('/cluster3/labData/jiamao/Genome/zebrafish/Danio_rerio.GRCz11.113.gtf')
OUT_GENE_TSV = ROOT / 'results' / 'weMERFISH' / 'wemerfish_measured_gene_divergence_genomewide.tsv'
OUT_ENH_TSV = ROOT / 'results' / 'weMERFISH' / 'wemerfish_measured_enhancer_divergence_genomewide.tsv'
OUT_META = ROOT / 'results' / 'weMERFISH' / 'wemerfish_measured_gene_divergence_genomewide.metadata.json'

WINDOW_BP = 100_000
EXCLUDE_TSS_BP = 2_000
SIG_P = 0.01
MIN_ENHANCERS_FOR_RANK = 5
EPS = 1e-3
K_GRAPH = 6
K_NEIGHBORS = 8
N_ROUNDS = 3
MAJORITY_THRESHOLD = 5
GENE_SET = 'measured'


def decode_arr(arr):
    out = []
    for x in arr:
        if isinstance(x, bytes):
            out.append(x.decode())
        else:
            out.append(str(x))
    return out


def parse_gtf_all_genes(gtf_path: Path):
    gene_records = {}
    chrom_tss = {}
    pat = re.compile(r'gene_name "([^"]+)"')
    with gtf_path.open() as fh:
        for line in fh:
            if line.startswith('#'):
                continue
            fields = line.rstrip('\n').split('\t')
            if len(fields) < 9 or fields[2] != 'gene':
                continue
            chrom = f'chr{fields[0]}'
            start = int(fields[3])
            end = int(fields[4])
            strand = fields[6]
            attrs = fields[8]
            m = pat.search(attrs)
            if not m:
                continue
            gname = m.group(1)
            tss = end if strand == '-' else start
            chrom_tss.setdefault(chrom, []).append(tss)
            if gname not in gene_records:
                gene_records[gname] = {
                    'gene_name': gname,
                    'chrom': chrom,
                    'start': start,
                    'end': end,
                    'strand': strand,
                    'tss': tss,
                    'gtf_line': line.rstrip('\n'),
                }
    return gene_records, chrom_tss


def normalize01(x):
    x = np.asarray(x, dtype=float)
    xmin = float(np.nanmin(x))
    xmax = float(np.nanmax(x))
    return (x - xmin) / max(xmax - xmin, 1e-8)


def fit_two_state_mask(values_norm, n_iter=25):
    vals = np.asarray(values_norm, dtype=float).ravel()
    c0, c1 = np.quantile(vals, [0.25, 0.75])
    z = np.zeros(vals.shape, dtype=bool)
    for _ in range(n_iter):
        d0 = np.abs(vals - c0)
        d1 = np.abs(vals - c1)
        z = d1 < d0
        if np.any(~z):
            c0 = vals[~z].mean()
        if np.any(z):
            c1 = vals[z].mean()
    active_label = 1 if c1 > c0 else 0
    mask = z if active_label == 1 else ~z
    return mask.astype(np.uint8), float(min(c0, c1)), float(max(c0, c1))


def majority_regularize_knn(mask, coords, valid3d, k=8, n_rounds=3, threshold=5):
    out = np.zeros(mask.shape, dtype=np.uint8)
    out[:] = mask.astype(np.uint8)
    if valid3d.sum() == 0:
        return out
    vv = out[valid3d].copy()
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm='auto')
    nn.fit(coords[valid3d])
    neigh = nn.kneighbors(return_distance=False)
    for _ in range(n_rounds):
        votes = vv[neigh].sum(axis=1)
        vv = (votes >= threshold).astype(np.uint8)
    out[:] = 0
    out[valid3d] = vv
    return out


def per_panel_p99_normalize(values, percentile=99.0):
    arr = np.asarray(values, dtype=float).copy()
    pos = arr[np.isfinite(arr) & (arr > 0)]
    if pos.size == 0:
        return np.zeros_like(arr), 0.0
    p = float(np.percentile(pos, percentile))
    p = max(p, 1e-8)
    arr = np.clip(arr, 0, p) / p
    return arr, p


def pearson_on_ranks(gene_ranks: np.ndarray, peak_matrix: np.ndarray):
    gene_centered = gene_ranks - gene_ranks.mean()
    gene_ss = np.sum(gene_centered ** 2)
    peak_ranks = rankdata(peak_matrix, axis=0, method='average')
    peak_centered = peak_ranks - peak_ranks.mean(axis=0, keepdims=True)
    peak_ss = np.sum(peak_centered ** 2, axis=0)
    numer = np.sum(gene_centered[:, None] * peak_centered, axis=0)
    denom = np.sqrt(gene_ss * peak_ss)
    rho = np.divide(numer, denom, out=np.zeros_like(numer, dtype=np.float64), where=denom > 0)
    return rho


def finite_corr(x, y, method='pearson'):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan
    xv = x[m]
    yv = y[m]
    if np.std(xv) < 1e-8 or np.std(yv) < 1e-8:
        return np.nan
    if method == 'pearson':
        return float(np.corrcoef(xv, yv)[0, 1])
    xr = pd.Series(xv).rank().to_numpy(dtype=float)
    yr = pd.Series(yv).rank().to_numpy(dtype=float)
    return float(np.corrcoef(xr, yr)[0, 1])


def hotspot_component_stats(values_abs, coords, q=0.9, k=6):
    vals = np.asarray(values_abs, dtype=float)
    valid = np.isfinite(vals)
    vals = vals[valid]
    subcoords = np.asarray(coords, dtype=float)[valid]
    if vals.size == 0:
        return 0, 0.0, 0.0
    thr = float(np.nanquantile(vals, q))
    hot = vals >= thr
    if hot.sum() == 0:
        return 0, 0.0, thr
    if hot.sum() == 1:
        return 1, 1.0, thr
    n_hot = int(hot.sum())
    n_neighbors = max(1, min(k, n_hot - 1))
    nn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto')
    nn.fit(subcoords[hot])
    graph = nn.kneighbors_graph(mode='connectivity')
    graph = graph.maximum(graph.T)
    n_comp, labels = connected_components(graph, directed=False)
    sizes = np.bincount(labels)
    largest = int(sizes.max()) if sizes.size else 0
    return int(n_comp), float(largest / max(hot.sum(), 1)), thr


def positive(x):
    return float(max(float(x), 0.0))


def summarize_gene(scores):
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return {
            'median_difference_structure_score': np.nan,
            'mean_difference_structure_score': np.nan,
            'top3_mean_difference_structure_score': np.nan,
            'max_difference_structure_score': np.nan,
            'std_difference_structure_score': np.nan,
            'absolute_divergence_score': np.nan,
        }
    top3 = np.sort(scores)[-3:]
    top3_mean = float(np.mean(top3))
    med = float(np.median(scores))
    out = {
        'median_difference_structure_score': med,
        'mean_difference_structure_score': float(np.mean(scores)),
        'top3_mean_difference_structure_score': top3_mean,
        'max_difference_structure_score': float(np.max(scores)),
        'std_difference_structure_score': float(np.std(scores)),
    }
    out['absolute_divergence_score'] = float(0.6 * med + 0.4 * top3_mean)
    return out


def main():
    gene_records, chrom_tss = parse_gtf_all_genes(GTF_PATH)

    with h5py.File(RNA_PATH, 'r') as rna, h5py.File(ATAC_PATH, 'r') as atac:
        gene_names = decode_arr(rna['var']['_index'][:])
        gene_name_to_idx = {g: i for i, g in enumerate(gene_names)}
        coords3d = np.asarray(atac['obsm']['spatial_rescaled_z'][:], dtype=float)
        valid3d = np.isfinite(coords3d).all(axis=1)
        chroms = decode_arr(atac['var']['chrom'][:])
        starts = atac['var']['start'][:].astype(np.int64)
        ends = atac['var']['end'][:].astype(np.int64)
        intervals = decode_arr(atac['var']['_index'][:])
        centers = ((starts + ends) // 2).astype(np.int64)

        enhancer_rows = []
        gene_rows = []

        measured_genes = [g for g in gene_names if g in gene_records]
        total = len(measured_genes)
        for gi, gene in enumerate(measured_genes, start=1):
            if gi % 25 == 0 or gi == 1 or gi == total:
                print(f'[{gi}/{total}] {gene}', flush=True)
            grec = gene_records[gene]
            chrom = grec['chrom']
            tss = grec['tss']
            tss_list = chrom_tss.get(chrom, [])
            gene_idx = gene_name_to_idx[gene]
            gene_expr = np.asarray(rna['X'][:, gene_idx], dtype=np.float64)
            gene_expr = np.maximum(gene_expr, 0.0)

            # candidate peaks
            same_chrom = np.array([c == chrom for c in chroms], dtype=bool)
            within = np.abs(centers - tss) <= WINDOW_BP
            not_tss = np.ones_like(within, dtype=bool)
            if tss_list:
                for x in tss_list:
                    not_tss &= np.abs(centers - x) > EXCLUDE_TSS_BP
            candidate_mask = same_chrom & within & not_tss
            candidate_idx = np.flatnonzero(candidate_mask)
            if candidate_idx.size == 0:
                gene_rows.append({
                    'gene': gene, 'n_candidate_peaks_tested': 0, 'n_significant_positive_links': 0,
                    'latent_mask_fraction': np.nan, 'mask_low_center': np.nan, 'mask_high_center': np.nan,
                    'n_enhancers_retained': 0, 'rank_eligible': False, 'note': 'no_candidate_peaks'
                })
                continue

            peak_matrix = np.asarray(atac['X'][:, candidate_idx], dtype=np.float64)
            gene_ranks = rankdata(gene_expr, method='average')
            rho = pearson_on_ranks(gene_ranks, peak_matrix)
            n = gene_expr.shape[0]
            df = n - 2
            rho_clip = np.clip(rho, -0.999999999, 0.999999999)
            t_stat = rho_clip * np.sqrt(df / (1.0 - rho_clip ** 2))
            p_one_tailed = student_t.sf(t_stat, df)
            sig_mask = p_one_tailed < SIG_P
            sig_idx = np.flatnonzero(sig_mask)

            gene_norm = normalize01(gene_expr)
            raw_mask, c_low, c_high = fit_two_state_mask(gene_norm, n_iter=25)
            latent_mask = majority_regularize_knn(raw_mask, coords3d, valid3d, k=K_NEIGHBORS, n_rounds=N_ROUNDS, threshold=MAJORITY_THRESHOLD)
            control = gene_expr * latent_mask
            control_norm, control_p99 = per_panel_p99_normalize(control, percentile=99.0)
            active_valid = (latent_mask > 0) & valid3d & np.isfinite(control_norm)

            if sig_idx.size == 0:
                gene_rows.append({
                    'gene': gene,
                    'n_candidate_peaks_tested': int(candidate_idx.size),
                    'n_significant_positive_links': 0,
                    'latent_mask_fraction': float(latent_mask.mean()),
                    'mask_low_center': c_low,
                    'mask_high_center': c_high,
                    'control_p99': control_p99,
                    'n_enhancers_retained': 0,
                    'rank_eligible': False,
                    'note': 'no_significant_links',
                })
                continue

            sig_peak_matrix = np.log1p(np.maximum(peak_matrix[:, sig_idx], 0.0))
            sig_candidate_idx = candidate_idx[sig_idx]
            sig_rho = rho[sig_idx]
            sig_p = p_one_tailed[sig_idx]

            gene_scores = []
            retained = 0
            for j, peak_idx in enumerate(sig_candidate_idx):
                raw_vals = sig_peak_matrix[:, j]
                latent_vals = raw_vals * latent_mask
                enh_norm, enh_p99 = per_panel_p99_normalize(latent_vals, percentile=99.0)
                valid = active_valid & np.isfinite(enh_norm)
                if valid.sum() < 10:
                    continue
                x = control_norm[valid]
                y = enh_norm[valid]
                diff = y - x
                logfc = np.log2((y + EPS) / (x + EPS))
                if np.std(x) < 1e-8:
                    slope, intercept = 0.0, float(np.mean(y))
                else:
                    slope, intercept = np.polyfit(x, y, deg=1)
                resid = y - (intercept + slope * x)
                coords_valid = coords3d[valid]
                diff_abs_moran = float(moran_i(np.abs(diff), coords_valid, k=K_GRAPH))
                logfc_abs_moran = float(moran_i(np.abs(logfc), coords_valid, k=K_GRAPH))
                resid_abs_moran = float(moran_i(np.abs(resid), coords_valid, k=K_GRAPH))
                _, diff_hot_frac, diff_thr = hotspot_component_stats(np.abs(diff), coords_valid, q=0.9, k=K_GRAPH)
                _, logfc_hot_frac, logfc_thr = hotspot_component_stats(np.abs(logfc), coords_valid, q=0.9, k=K_GRAPH)
                _, resid_hot_frac, resid_thr = hotspot_component_stats(np.abs(resid), coords_valid, q=0.9, k=K_GRAPH)
                diff_structure_score = float(np.nanmean(np.abs(diff)) * (1.0 + positive(diff_abs_moran)) * diff_hot_frac)
                logfc_structure_score = float(np.nanmean(np.abs(logfc)) * (1.0 + positive(logfc_abs_moran)) * logfc_hot_frac)
                residual_structure_score = float(np.nanstd(resid) * (1.0 + positive(resid_abs_moran)) * resid_hot_frac)
                difference_structure_score = float(0.4 * diff_structure_score + 0.25 * logfc_structure_score + 0.35 * residual_structure_score)
                retained += 1
                gene_scores.append(difference_structure_score)
                enhancer_rows.append({
                    'gene': gene,
                    'panel_id': f'E{retained}',
                    'enhancer': f"{chroms[peak_idx]}:{starts[peak_idx]}-{ends[peak_idx]}",
                    'interval_id': intervals[peak_idx],
                    'initial_rho': float(sig_rho[j]),
                    'pvalue_one_tailed_positive': float(sig_p[j]),
                    'distance_to_tss': int(centers[peak_idx] - tss),
                    'abs_distance_to_tss': int(abs(centers[peak_idx] - tss)),
                    'latent_state_rho': finite_corr(x, y, method='pearson'),
                    'mean_abs_diff': float(np.nanmean(np.abs(diff))),
                    'mean_abs_logfc': float(np.nanmean(np.abs(logfc))),
                    'residual_std': float(np.nanstd(resid)),
                    'diff_structure_score': diff_structure_score,
                    'logfc_structure_score': logfc_structure_score,
                    'residual_structure_score': residual_structure_score,
                    'difference_structure_score': difference_structure_score,
                })

            gs = summarize_gene(gene_scores)
            gene_rows.append({
                'gene': gene,
                'n_candidate_peaks_tested': int(candidate_idx.size),
                'n_significant_positive_links': int(sig_idx.size),
                'latent_mask_fraction': float(latent_mask.mean()),
                'mask_low_center': c_low,
                'mask_high_center': c_high,
                'control_p99': control_p99,
                'n_enhancers_retained': int(retained),
                'rank_eligible': bool(retained >= MIN_ENHANCERS_FOR_RANK),
                'note': '',
                **gs,
            })

    enh_df = pd.DataFrame(enhancer_rows)
    gene_df = pd.DataFrame(gene_rows)
    gene_df = gene_df.sort_values(['absolute_divergence_score', 'median_difference_structure_score'], ascending=[False, False], na_position='last').reset_index(drop=True)
    eligible = gene_df['rank_eligible'].fillna(False).to_numpy(dtype=bool)
    gene_df['absolute_rank'] = np.nan
    gene_df.loc[eligible, 'absolute_rank'] = np.arange(1, eligible.sum() + 1)

    enh_df.to_csv(OUT_ENH_TSV, sep='\t', index=False)
    gene_df.to_csv(OUT_GENE_TSV, sep='\t', index=False)
    meta = {
        'gene_set': GENE_SET,
        'n_measured_genes_in_rna': int(len(gene_names)),
        'n_measured_genes_in_gtf': int(len(measured_genes)),
        'window_bp': WINDOW_BP,
        'exclude_tss_bp': EXCLUDE_TSS_BP,
        'significance_rule': 'one-tailed positive Spearman p < 0.01',
        'latent_state_rule': 'raw RNA -> min-max -> two-state fit -> 3D kNN majority regularization',
        'gene_rank_rule': 'absolute_divergence_score = 0.6 * median_difference_structure_score + 0.4 * top3_mean_difference_structure_score',
        'min_enhancers_for_rank': MIN_ENHANCERS_FOR_RANK,
        'out_gene_tsv': str(OUT_GENE_TSV),
        'out_enhancer_tsv': str(OUT_ENH_TSV),
    }
    OUT_META.write_text(json.dumps(meta, indent=2), encoding='utf-8')

    print(str(OUT_GENE_TSV))
    print(gene_df.loc[gene_df['rank_eligible']].head(20)[['absolute_rank','gene','absolute_divergence_score','n_enhancers_retained','latent_mask_fraction']].to_string(index=False))

if __name__ == '__main__':
    main()
