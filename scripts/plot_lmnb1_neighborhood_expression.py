from pathlib import Path
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

RNA_PATH = Path('/cluster3/labData/jiamao/SpaRegVision/weMERFISH/weMERFISH_combined_C_6s_E1_rescaled_z.h5ad')
OUT_PNG = Path('/cluster2/huanglab/jiamao/Project/SpaRegVision/results/weMERFISH/lmnb1_neighborhood_raw_imputed_expression_xy.png')
OUT_TSV = Path('/cluster2/huanglab/jiamao/Project/SpaRegVision/results/weMERFISH/lmnb1_neighborhood_raw_imputed_expression_xy.summary.tsv')
GENES = ['lmnb1','phax','aldh7a1','C10H5orf63','megf10','apbb1','mir24-1','mir27d','dre-mir-23b']
POINT_SIZE = 0.2

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'],
    'font.size': 7,
    'axes.spines.right': False,
    'axes.spines.top': False,
    'axes.linewidth': 0.8,
})


def decode_arr(arr):
    return [x.decode() if isinstance(x, bytes) else str(x) for x in arr]


def per_panel_p99_normalize(values, percentile=99.0):
    arr = np.asarray(values, dtype=float).copy()
    pos = arr[np.isfinite(arr) & (arr > 0)]
    if pos.size == 0:
        return np.zeros_like(arr), 0.0
    p = float(np.percentile(pos, percentile))
    p = max(p, 1e-8)
    arr = np.clip(arr, 0, p) / p
    return arr, p


def add_panel(ax, xy, values, title, cmap):
    ax.scatter(xy[:, 0], xy[:, 1], c=values, cmap=cmap, s=POINT_SIZE, linewidths=0, rasterized=True)
    ax.set_title(title, fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')


def main():
    with h5py.File(RNA_PATH, 'r') as f:
        measured_names = decode_arr(f['var']['_index'][:])
        measured_map = {g:i for i,g in enumerate(measured_names)}
        imputed_names = decode_arr(f['uns']['imputed_gene_names'][:]) if 'imputed_gene_names' in f['uns'] else []
        imputed_map = {g:i for i,g in enumerate(imputed_names)}
        xy = np.asarray(f['obsm']['spatial_rescaled_z'][:, :2], dtype=float)

        rows = []
        panels = []
        for gene in GENES:
            if gene in measured_map:
                vals = np.asarray(f['X'][:, measured_map[gene]], dtype=float)
                source = 'measured raw RNA'
            elif gene in imputed_map:
                vals = np.asarray(f['obsm']['X_imputed'][:, imputed_map[gene]], dtype=float)
                source = 'imputed RNA'
            else:
                vals = np.zeros(xy.shape[0], dtype=float)
                source = 'missing'
            vals = np.maximum(vals, 0.0)
            show, p99 = per_panel_p99_normalize(vals, percentile=99.0)
            panels.append((gene, source, show))
            rows.append({'gene': gene, 'source': source, 'p99': p99, 'nonzero_fraction': float(np.mean(vals > 0)), 'signal_sum': float(np.sum(vals))})

    n = len(panels)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 4.2 * nrows))
    axes = np.asarray(axes).ravel()
    cmap = mpl.cm.Blues

    for ax, (gene, source, show) in zip(axes, panels):
        add_panel(ax, xy, show, f'{gene}\n{source}', cmap)
    for ax in axes[n:]:
        ax.axis('off')

    fig.suptitle('lmnb1 ±100 kb neighborhood expression maps (x-y projection)', fontsize=10, y=0.995)
    fig.text(0.01, 0.01,
             'Only lmnb1 is available in measured RNA. The other neighborhood genes are shown from the imputed RNA matrix. '
             'Each panel was independently clipped at p99 and rescaled to 0–1 for shape comparison.',
             ha='left', va='bottom', fontsize=6.5, color='#555555')
    fig.tight_layout(rect=[0, 0.03, 1, 0.985])
    fig.savefig(OUT_PNG, dpi=300, bbox_inches='tight')

    import pandas as pd
    pd.DataFrame(rows).to_csv(OUT_TSV, sep='\t', index=False)
    print(OUT_PNG)
    print(OUT_TSV)

if __name__ == '__main__':
    main()
