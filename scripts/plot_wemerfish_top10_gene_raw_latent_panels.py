from pathlib import Path
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
RNA_PATH = Path('/cluster3/labData/jiamao/SpaRegVision/weMERFISH/weMERFISH_combined_C_6s_E1_rescaled_z.h5ad')
ATAC_PATH = Path('/cluster3/labData/jiamao/SpaRegVision/weMERFISH/weMERFISH_spatial_ATAC_C_6s_E1.h5ad')
GENE_RANK_PATH = ROOT / 'results' / 'weMERFISH' / 'wemerfish_measured_gene_divergence_genomewide.tsv'
ENH_PATH = ROOT / 'results' / 'weMERFISH' / 'wemerfish_measured_enhancer_divergence_genomewide.tsv'
OUT_DIR = ROOT / 'results' / 'weMERFISH' / os.environ.get('OUT_DIR_NAME', 'top10_measured_gene_divergence_panels')
POINT_SIZE = 0.2
K_NEIGHBORS = 8
N_ROUNDS = 3
MAJORITY_THRESHOLD = 5
TOPN = 10
START_RANK = int(os.environ.get('START_RANK', '1'))

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'],
    'font.size': 7,
    'axes.spines.right': False,
    'axes.spines.top': False,
    'axes.linewidth': 0.8,
})


def decode_arr(arr):
    out = []
    for x in arr:
        out.append(x.decode() if isinstance(x, bytes) else str(x))
    return out


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
    out = mask.astype(np.uint8).copy()
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


def add_scatter(ax, xy, values, cmap, title):
    ax.scatter(xy[:, 0], xy[:, 1], c=values, cmap=cmap, s=POINT_SIZE, linewidths=0, rasterized=True)
    ax.set_title(title, fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gene_df = pd.read_csv(GENE_RANK_PATH, sep='\t')
    enh_df = pd.read_csv(ENH_PATH, sep='\t')
    ranked = gene_df[gene_df['rank_eligible'].fillna(False)].copy().reset_index(drop=True)
    top = ranked.iloc[START_RANK - 1: START_RANK - 1 + TOPN].copy()

    with h5py.File(RNA_PATH, 'r') as rna, h5py.File(ATAC_PATH, 'r') as atac:
        gene_names = decode_arr(rna['var']['_index'][:])
        gene_to_idx = {g:i for i, g in enumerate(gene_names)}
        atac_intervals = decode_arr(atac['var']['_index'][:])
        interval_to_idx = {x:i for i,x in enumerate(atac_intervals)}
        coords3d = np.asarray(atac['obsm']['spatial_rescaled_z'][:], dtype=float)
        coords_xy = coords3d[:, :2]
        valid3d = np.isfinite(coords3d).all(axis=1)

        blue = mpl.cm.Blues
        pink = mpl.colors.LinearSegmentedColormap.from_list('pinkish', ['#EDEDED', '#FFF7F3', '#FDE0DD', '#FCC5C0', '#FA9FB5', '#F768A1', '#DD3497', '#AE017E'], N=256)

        summary_rows = []
        for _, grow in top.iterrows():
            gene = grow['gene']
            rank = int(grow['absolute_rank'])
            gidx = gene_to_idx[gene]
            expr = np.asarray(rna['X'][:, gidx], dtype=float)
            expr = np.maximum(expr, 0.0)
            gene_norm = normalize01(expr)
            raw_mask, c_low, c_high = fit_two_state_mask(gene_norm, n_iter=25)
            latent_mask = majority_regularize_knn(raw_mask, coords3d, valid3d, k=K_NEIGHBORS, n_rounds=N_ROUNDS, threshold=MAJORITY_THRESHOLD)
            control = expr * latent_mask
            gene_raw_show, _ = per_panel_p99_normalize(expr, percentile=99.0)
            control_show, control_p99 = per_panel_p99_normalize(control, percentile=99.0)

            sub = enh_df[enh_df['gene'] == gene].copy().sort_values(['difference_structure_score','initial_rho'], ascending=[False, False]).reset_index(drop=True)
            intervals = sub['interval_id'].astype(str).tolist()
            cols = np.array([interval_to_idx[x] for x in intervals], dtype=int)
            order = np.argsort(cols)
            sorted_cols = cols[order]
            raw_sorted = np.asarray(atac['X'][:, sorted_cols], dtype=float)
            inv_order = np.argsort(order)
            raw_atac = raw_sorted[:, inv_order]
            raw_atac = np.log1p(np.maximum(raw_atac, 0.0))

            n = len(sub)
            ncols = 6
            n_panels = 2 + 2 * n
            nrows = int(np.ceil(n_panels / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(2.0 * ncols, 2.25 * nrows))
            axes = np.asarray(axes).ravel()

            add_scatter(axes[0], coords_xy, gene_raw_show, blue, f'{gene} raw RNA')
            add_scatter(axes[1], coords_xy, control_show, blue, f'{gene} latent\nlatent mask × raw RNA')
            for k, row in sub.iterrows():
                raw_ax = axes[2 + 2 * k]
                lat_ax = axes[2 + 2 * k + 1]
                raw_vals = raw_atac[:, k]
                raw_show, _ = per_panel_p99_normalize(raw_vals, percentile=99.0)
                add_scatter(raw_ax, coords_xy, raw_show, pink, f"{row['panel_id']} raw\ninitρ={row['initial_rho']:.2f}")
                lat_vals = raw_vals * latent_mask
                lat_show, _ = per_panel_p99_normalize(lat_vals, percentile=99.0)
                add_scatter(lat_ax, coords_xy, lat_show, pink, f"{row['panel_id']} latent\nlatentρ={row['latent_state_rho']:.2f}, diffS={row['difference_structure_score']:.2f}")

            for ax in axes[n_panels:]:
                ax.axis('off')

            fig.suptitle(f'#{rank} {gene} measured-gene divergence panels (x-y projection)', fontsize=10, y=0.995)
            fig.text(0.01, 0.01,
                     f'absolute_divergence_score={grow["absolute_divergence_score"]:.3f}; latent_mask_fraction={grow["latent_mask_fraction"]:.3%}; '
                     f'n_enhancers_retained={int(grow["n_enhancers_retained"])}. Left = raw log1p ATAC, right = latent-selected log1p ATAC.',
                     ha='left', va='bottom', fontsize=6.5, color='#555555')
            fig.tight_layout(rect=[0, 0.03, 1, 0.985])
            stem = OUT_DIR / f'{rank:02d}_{gene}_raw_latent_panels'
            fig.savefig(stem.with_suffix('.png'), dpi=300, bbox_inches='tight')
            plt.close(fig)

            summary_rows.append({
                'rank': rank,
                'gene': gene,
                'absolute_divergence_score': float(grow['absolute_divergence_score']),
                'latent_mask_fraction': float(grow['latent_mask_fraction']),
                'n_enhancers_retained': int(grow['n_enhancers_retained']),
                'mask_low_center': c_low,
                'mask_high_center': c_high,
                'control_p99': control_p99,
                'output_png': str(stem.with_suffix('.png')),
            })
            print(f'done {rank:02d} {gene}', flush=True)

    pd.DataFrame(summary_rows).to_csv(OUT_DIR / 'top10_summary.tsv', sep='\t', index=False)
    print(str(OUT_DIR / 'top10_summary.tsv'))

if __name__ == '__main__':
    main()
