from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sparegvision.io import read_feature_matrix
from sparegvision.gsps import moran_i

ROOT = Path(__file__).resolve().parents[1]
RNA_PATH = Path('/cluster3/labData/jiamao/SpaRegVision/weMERFISH/weMERFISH_combined_C_6s_E1_rescaled_z.h5ad')
ATAC_PATH = Path('/cluster3/labData/jiamao/SpaRegVision/weMERFISH/weMERFISH_spatial_ATAC_C_6s_E1.h5ad')
LINK_PATH = ROOT / 'results' / 'weMERFISH' / 'hdlbpa_gene_peak_links_100kb_excludeTSS2kb.significant.tsv'
LATENT_PATH = ROOT / 'results' / 'weMERFISH' / 'hdlbpa_latent_state_3d_nefmlike_no_log1p.cells.tsv.gz'
OUT_TSV = ROOT / 'results' / 'weMERFISH' / 'hdlbpa_significant_peak_spatial_difference_scores.tsv'
OUT_PNG = ROOT / 'results' / 'weMERFISH' / 'hdlbpa_significant_peak_spatial_difference_scores_xy_raw_and_latent.png'

GENE = 'hdlbpa'
EPS = 1e-3
K_GRAPH = 6
POINT_SIZE = 0.2

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'],
    'pdf.fonttype': 42,
    'font.size': 7,
    'axes.spines.right': False,
    'axes.spines.top': False,
    'axes.linewidth': 0.8,
})


def normalize01(x):
    x = np.asarray(x, dtype=float)
    xmin = float(np.nanmin(x))
    xmax = float(np.nanmax(x))
    return (x - xmin) / max(xmax - xmin, 1e-8)


def per_panel_p99_normalize(values, percentile=99.0):
    arr = np.asarray(values, dtype=float).copy()
    pos = arr[np.isfinite(arr) & (arr > 0)]
    if pos.size == 0:
        return np.zeros_like(arr), 0.0
    p = float(np.percentile(pos, percentile))
    p = max(p, 1e-8)
    arr = np.clip(arr, 0, p) / p
    return arr, p


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
    nn = NearestNeighbors(n_neighbors=min(k + 1, hot.sum()), algorithm='auto')
    nn.fit(subcoords[hot])
    graph = nn.kneighbors_graph(mode='connectivity')
    graph = graph.maximum(graph.T)
    n_comp, labels = connected_components(graph, directed=False)
    sizes = np.bincount(labels)
    largest = int(sizes.max()) if sizes.size else 0
    return int(n_comp), float(largest / max(hot.sum(), 1)), thr


def positive(x):
    return float(max(float(x), 0.0))


def add_scatter(ax, xy, values, cmap, title, vmin=None, vmax=None):
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=values, cmap=cmap, vmin=vmin, vmax=vmax, s=POINT_SIZE, linewidths=0, rasterized=True)
    ax.set_title(title, fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    return sc


def main():
    links = pd.read_csv(LINK_PATH, sep='\t').copy()
    links = links.reset_index(drop=True)
    links['panel_id'] = [f'E{i+1}' for i in range(len(links))]

    latent = pd.read_csv(LATENT_PATH, sep='\t')
    latent = latent.reset_index(drop=True)
    latent_mask = latent['hdlbpa_latent_state_3d'].to_numpy(dtype=float)
    coords3d = latent[['spatial_x', 'spatial_y', 'spatial_z_rescaled']].to_numpy(dtype=float)
    coords_xy = coords3d[:, :2]
    valid3d = latent['valid_3d'].to_numpy(dtype=int).astype(bool) if 'valid_3d' in latent.columns else np.isfinite(coords3d).all(axis=1)

    rna = ad.read_h5ad(RNA_PATH, backed='r')
    atac = ad.read_h5ad(ATAC_PATH, backed='r')
    if rna.n_obs != atac.n_obs or latent.shape[0] != atac.n_obs:
        raise ValueError('RNA/ATAC/latent row counts differ')
    if 'cell_id' in rna.obs.columns and 'cell_id' in atac.obs.columns:
        if not np.array_equal(rna.obs['cell_id'].astype(str).to_numpy(), atac.obs['cell_id'].astype(str).to_numpy()):
            raise ValueError('RNA and ATAC cell_id order differs')

    expr = read_feature_matrix(rna, [GENE], obs_names=rna.obs_names, batch_size=1)[:, 0]
    expr = np.maximum(np.asarray(expr, dtype=float), 0.0)
    control = expr * latent_mask
    control_norm, control_p99 = per_panel_p99_normalize(control, percentile=99.0)

    enhancers = links['interval_id'].astype(str).tolist()
    raw_atac = read_feature_matrix(atac, enhancers, obs_names=atac.obs_names, batch_size=64)
    raw_atac = np.log1p(np.maximum(np.asarray(raw_atac, dtype=float), 0.0))

    active_valid = (latent_mask > 0) & valid3d & np.isfinite(control_norm)
    spot_coords = coords3d[active_valid]
    control_active = control_norm[active_valid]

    rows = []
    for i, link_row in links.iterrows():
        raw_vals = raw_atac[:, i]
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

        diff_hot_n, diff_hot_frac, diff_thr = hotspot_component_stats(np.abs(diff), coords_valid, q=0.9, k=K_GRAPH)
        logfc_hot_n, logfc_hot_frac, logfc_thr = hotspot_component_stats(np.abs(logfc), coords_valid, q=0.9, k=K_GRAPH)
        resid_hot_n, resid_hot_frac, resid_thr = hotspot_component_stats(np.abs(resid), coords_valid, q=0.9, k=K_GRAPH)

        diff_structure_score = float(np.nanmean(np.abs(diff)) * (1.0 + positive(diff_abs_moran)) * diff_hot_frac)
        logfc_structure_score = float(np.nanmean(np.abs(logfc)) * (1.0 + positive(logfc_abs_moran)) * logfc_hot_frac)
        residual_structure_score = float(np.nanstd(resid) * (1.0 + positive(resid_abs_moran)) * resid_hot_frac)
        difference_structure_score = float(0.4 * diff_structure_score + 0.25 * logfc_structure_score + 0.35 * residual_structure_score)

        rows.append({
            'gene': GENE,
            'panel_id': link_row['panel_id'],
            'enhancer': link_row['region'],
            'interval_id': link_row['interval_id'],
            'initial_rho': float(link_row['spearman_rho']),
            'distance_to_tss': float(link_row['distance_to_tss']),
            'abs_distance_to_tss': float(link_row['abs_distance_to_tss']),
            'pvalue_one_tailed_positive': float(link_row['pvalue_one_tailed_positive']),
            'peak_signal_sum': float(link_row['peak_signal_sum']),
            'control_p99': control_p99,
            'enhancer_p99': enh_p99,
            'latent_state_rho': finite_corr(x, y, method='pearson'),
            'latent_state_spearman': finite_corr(x, y, method='spearman'),
            'mean_abs_diff': float(np.nanmean(np.abs(diff))),
            'mean_diff': float(np.nanmean(diff)),
            'std_diff': float(np.nanstd(diff)),
            'diff_abs_moran_i': diff_abs_moran,
            'diff_hotspots_q90': diff_hot_n,
            'diff_largest_hotspot_fraction': diff_hot_frac,
            'diff_abs_threshold_q90': diff_thr,
            'mean_abs_logfc': float(np.nanmean(np.abs(logfc))),
            'mean_logfc': float(np.nanmean(logfc)),
            'std_logfc': float(np.nanstd(logfc)),
            'logfc_abs_moran_i': logfc_abs_moran,
            'logfc_hotspots_q90': logfc_hot_n,
            'logfc_largest_hotspot_fraction': logfc_hot_frac,
            'logfc_abs_threshold_q90': logfc_thr,
            'residual_slope': float(slope),
            'residual_intercept': float(intercept),
            'residual_mean': float(np.nanmean(resid)),
            'residual_std': float(np.nanstd(resid)),
            'residual_abs_moran_i': resid_abs_moran,
            'residual_hotspots_q90': resid_hot_n,
            'residual_largest_hotspot_fraction': resid_hot_frac,
            'residual_abs_threshold_q90': resid_thr,
            'diff_structure_score': diff_structure_score,
            'logfc_structure_score': logfc_structure_score,
            'residual_structure_score': residual_structure_score,
            'difference_structure_score': difference_structure_score,
        })

    out = pd.DataFrame(rows).sort_values(['difference_structure_score', 'initial_rho'], ascending=[False, False]).reset_index(drop=True)
    out.to_csv(OUT_TSV, sep='\t', index=False)

    # plot panels: gene raw + gene latent + per-enhancer raw/latent pair
    n = len(out)
    ncols = 6
    n_panels = 2 + 2 * n
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.0 * ncols, 2.25 * nrows))
    axes = np.asarray(axes).ravel()
    blue = mpl.cm.Blues
    pink = mpl.colors.LinearSegmentedColormap.from_list('pinkish', ['#EDEDED', '#FFF7F3', '#FDE0DD', '#FCC5C0', '#FA9FB5', '#F768A1', '#DD3497', '#AE017E'], N=256)

    gene_raw_show, _ = per_panel_p99_normalize(expr, percentile=99.0)
    control_show, _ = per_panel_p99_normalize(control, percentile=99.0)
    add_scatter(axes[0], coords_xy, gene_raw_show, blue, 'HDLBPA raw RNA')
    add_scatter(axes[1], coords_xy, control_show, blue, 'HDLBPA latent\nlatent mask × raw RNA')

    feature_map = {f: j for j, f in enumerate(enhancers)}
    for k, row in out.iterrows():
        raw_ax = axes[2 + 2 * k]
        lat_ax = axes[2 + 2 * k + 1]

        raw_vals = raw_atac[:, feature_map[row['interval_id']]]
        raw_show, _ = per_panel_p99_normalize(raw_vals, percentile=99.0)
        add_scatter(raw_ax, coords_xy, raw_show, pink, f"{row['panel_id']} raw\ninitρ={row['initial_rho']:.2f}")

        lat_vals = raw_vals * latent_mask
        lat_show, _ = per_panel_p99_normalize(lat_vals, percentile=99.0)
        add_scatter(lat_ax, coords_xy, lat_show, pink, f"{row['panel_id']} latent\nlatentρ={row['latent_state_rho']:.2f}, diffS={row['difference_structure_score']:.2f}")

    for ax in axes[n_panels:]:
        ax.axis('off')

    fig.suptitle('HDLBPA significant enhancers ranked by 3D latent-state spatial difference (x-y projection)', fontsize=10, y=0.995)
    fig.text(0.01, 0.01,
             'Latent state used the retained 3D no-log1p HDLBPA mask. Control = raw HDLBPA RNA × latent mask. '
             'For each enhancer, left = raw log1p ATAC, right = latent-selected log1p ATAC. '
             'Pairs are ordered by difference_structure_score.',
             ha='left', va='bottom', fontsize=6.5, color='#555555')
    fig.tight_layout(rect=[0, 0.03, 1, 0.985])
    fig.savefig(OUT_PNG, dpi=300, bbox_inches='tight')

    print(OUT_TSV)
    print(OUT_PNG)
    print(out[['panel_id','enhancer','difference_structure_score','initial_rho','latent_state_rho']].head(10).to_string(index=False))

if __name__ == '__main__':
    main()
