from __future__ import annotations

import json
import os
import re
import textwrap
import urllib.parse
import urllib.request
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[1]
CANON = Path('/cluster3/labData/jiamao/SpaRegVision/GW12')
RNA_PATH = CANON / 'rectified_h' / 'GW12_spatial_RNA_rectangle_H_2dwarp_linear_101x76.h5ad'
ATAC_PATH = CANON / 'rectified_h' / 'GW12_spatial_ATAC_rectangle_H_2dwarp_linear_101x76.h5ad'
GENE_TSV = ROOT / 'results' / 'weMERFISH' / 'GW12' / 'gw12_rectified_h_gene_divergence_absolute_maskgt2pct.tsv'
ENH_TSV = ROOT / 'results' / 'weMERFISH' / 'GW12' / 'gw12_rectified_h_enhancer_divergence.tsv'
OUTDIR = ROOT / 'results' / 'weMERFISH' / 'GW12' / 'top20_maskgt2pct_pdfs'
TOPN = 20
WINDOW_BP = 100_000
POINT_SIZE = 1.6
EPS = 1e-3
ENSEMBL = 'https://rest.ensembl.org'

RAW_CMAP = mpl.cm.Blues
PEAK_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    'peak_cmap',
    ['#EDEDED', '#FFF7F3', '#FDE0DD', '#FCC5C0', '#FA9FB5', '#F768A1', '#DD3497', '#AE017E'],
    N=256,
)
GENE_COLOR_TARGET = '#1F4E79'
GENE_COLOR_NEIGHBOR = '#666666'
ENH_COLOR = '#7F7F7F'

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'],
    'pdf.fonttype': 42,
    'font.size': 7,
    'axes.spines.right': False,
    'axes.spines.top': False,
    'axes.linewidth': 0.8,
})


def ensembl_json(path: str, params: dict | None = None):
    url = ENSEMBL + path
    if params:
        url += '?' + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={'Content-Type': 'application/json', 'Accept': 'application/json'})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def gene_lookup(gene: str):
    data = ensembl_json(f'/lookup/symbol/homo_sapiens/{gene}', {'expand': 0})
    return {
        'gene_name': data['display_name'],
        'chrom': f"chr{data['seq_region_name']}",
        'start': int(data['start']),
        'end': int(data['end']),
        'strand': '-' if int(data['strand']) < 0 else '+',
        'tss': int(data['end']) if int(data['strand']) < 0 else int(data['start']),
    }


def region_neighbors(chrom: str, start: int, end: int, target_gene: str):
    region = f"{chrom.replace('chr','')}:{start}-{end}"
    data = ensembl_json(f'/overlap/region/human/{region}', {'feature': 'gene'})
    rows = []
    seen = set()
    for x in data:
        name = x.get('external_name') or x.get('id')
        if not name or name in seen or name == target_gene:
            continue
        seen.add(name)
        strand = '-' if int(x['strand']) < 0 else '+'
        rows.append({
            'gene_name': name,
            'chrom': f"chr{x['seq_region_name']}",
            'start': int(x['start']),
            'end': int(x['end']),
            'strand': strand,
            'tss': int(x['end']) if strand == '-' else int(x['start']),
        })
    if not rows:
        return pd.DataFrame(columns=['gene_name','chrom','start','end','strand','tss'])
    return pd.DataFrame(rows).sort_values(['start','end','gene_name']).reset_index(drop=True)


def dense_to_grid(obs, values):
    grid = np.full((76, 101), np.nan, dtype=float)
    rows = obs['grid_row'].to_numpy(dtype=int)
    cols = obs['grid_col'].to_numpy(dtype=int)
    grid[rows, cols] = np.asarray(values, dtype=float)
    return grid


def normalize01(x):
    x = np.asarray(x, dtype=float)
    xmin = float(np.nanmin(x))
    xmax = float(np.nanmax(x))
    return (x - xmin) / max(xmax - xmin, 1e-8)


def fit_two_state_mask(gene_norm, n_iter=25):
    vals = gene_norm.ravel()
    c0, c1 = np.quantile(vals, [0.25, 0.75])
    for _ in range(n_iter):
        d0 = np.abs(vals - c0)
        d1 = np.abs(vals - c1)
        z = d1 < d0
        if np.any(~z):
            c0 = vals[~z].mean()
        if np.any(z):
            c1 = vals[z].mean()
    active_label = 1 if c1 > c0 else 0
    mask = z.reshape(gene_norm.shape) if active_label == 1 else (~z).reshape(gene_norm.shape)
    return mask.astype(np.uint8)


def majority_regularize(mask, n_rounds=3, threshold=5):
    out = mask.copy().astype(np.uint8)
    for _ in range(n_rounds):
        padded = np.pad(out, ((1, 1), (1, 1)), mode='edge')
        votes = np.zeros_like(out, dtype=int)
        for dy in range(3):
            for dx in range(3):
                votes += padded[dy:dy + out.shape[0], dx:dx + out.shape[1]]
        out = (votes >= threshold).astype(np.uint8)
    return out


def per_panel_p99_normalize(grid, percentile=99.0):
    arr = np.asarray(grid, dtype=float).copy()
    valid = np.isfinite(arr)
    pos = arr[valid & (arr > 0)]
    if pos.size == 0:
        out = np.zeros_like(arr, dtype=float)
        out[~valid] = np.nan
        return out, 0.0
    p = float(np.percentile(pos, percentile))
    p = max(p, 1e-8)
    arr[valid] = np.clip(arr[valid], 0, p) / p
    return arr, p


def format_panel_title(lines, wrap_width=10):
    out = []
    for line in lines:
        if line is None:
            continue
        s = str(line).strip()
        if not s:
            continue
        out.append(textwrap.fill(s, width=wrap_width, break_long_words=False, break_on_hyphens=False))
    return '\n'.join(out)


def add_panel_label(ax, title):
    ax.set_facecolor('none')
    ax.patch.set_alpha(0.0)
    ax.axis('off')
    ax.text(0.02, 0.02, title, transform=ax.transAxes, ha='left', va='bottom', fontsize=4.5, linespacing=1.0, clip_on=True)


def add_scatter(ax, xy, values, cmap):
    ax.set_facecolor('none')
    ax.patch.set_alpha(0.0)
    ax.scatter(xy[:, 0], xy[:, 1], c=values, cmap=cmap, s=POINT_SIZE, linewidths=0, rasterized=True)
    xmin = float(np.nanmin(xy[:, 0])); xmax = float(np.nanmax(xy[:, 0]))
    ymin = float(np.nanmin(xy[:, 1])); ymax = float(np.nanmax(xy[:, 1]))
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymax, ymin)
    ax.margins(x=0.0, y=0.0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect('equal', adjustable='box', anchor='N')
    for spine in ax.spines.values():
        spine.set_visible(False)


def add_score_row(ax, xvals, yvals, ylabel, xlim=None):
    x = np.asarray(xvals, dtype=float)
    y = np.asarray(yvals, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    ax.set_facecolor('none')
    ax.patch.set_alpha(0.0)
    if m.any():
        if xlim is not None:
            ax.set_xlim(*xlim)
        ymin = float(np.nanmin(y[m])); ymax = float(np.nanmax(y[m]))
        if abs(ymax - ymin) < 1e-12:
            ymin -= 0.5; ymax += 0.5
        else:
            pad_y = (ymax - ymin) * 0.12
            ymin -= pad_y; ymax += pad_y
        ax.set_ylim(ymin, ymax)
        base = ymin
        ax.vlines(x[m], base, y[m], color='#4D4D4D', linewidth=0.8, zorder=3)
        ax.scatter(x[m], y[m], s=22, color='#4D4D4D', marker='D', linewidths=0, zorder=4)
    elif xlim is not None:
        ax.set_xlim(*xlim)
    ax.set_ylabel(ylabel, fontsize=6.0, rotation=0, ha='right', va='center', labelpad=18)
    ax.tick_params(axis='x', bottom=False, labelbottom=False)
    ax.tick_params(axis='y', labelsize=5.2, length=2)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_linewidth(0.6)
    ax.spines['bottom'].set_linewidth(0.6)
    ax.spines['bottom'].set_color('#B0B0B0')
    ax.spines['left'].set_color('#B0B0B0')


def draw_locus_axis(ax, gene_record, neighbor_df, links_df):
    chrom = gene_record['chrom']
    xs = [int(gene_record['start']), int(gene_record['end'])]
    if len(neighbor_df):
        xs.extend(neighbor_df['start'].astype(int).tolist())
        xs.extend(neighbor_df['end'].astype(int).tolist())
    if len(links_df):
        xs.extend(links_df['peak_start'].astype(int).tolist())
        xs.extend(links_df['peak_end'].astype(int).tolist())
    xmin = int(min(xs) - 2500); xmax = int(max(xs) + 2500); span = xmax - xmin
    ax.set_xlim(xmin, xmax); ax.set_ylim(-1.0, 1.0); ax.set_yticks([])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(round(x)):,}"))
    ax.set_xlabel(f"{chrom} genomic position", labelpad=22)
    ax.set_title(f"{gene_record['gene_name']} locus and significant enhancers", fontsize=9, pad=8)
    ax.spines['left'].set_visible(False); ax.spines['right'].set_visible(False); ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_position(('data', 0.0)); ax.spines['bottom'].set_linewidth(1.0); ax.spines['bottom'].set_color('#444444')
    ax.tick_params(axis='x', bottom=True, labelbottom=True, length=3, pad=2)

    all_genes = pd.concat([pd.DataFrame([gene_record]), neighbor_df], ignore_index=True)
    gene_box_y = 0.28; gene_box_h = 0.18
    for _, row in all_genes.sort_values(['start','end','gene_name']).iterrows():
        color = GENE_COLOR_TARGET if row['gene_name'] == gene_record['gene_name'] else GENE_COLOR_NEIGHBOR
        start = int(row['start']); end = int(row['end']); tss = int(row['tss']); strand = row.get('strand', '+')
        gene_w = max((end - start) * 0.5, 1.0)
        gene_x = tss - gene_w if strand == '-' else tss
        ax.add_patch(mpl.patches.Rectangle((gene_x, gene_box_y), gene_w, gene_box_h, facecolor=color, edgecolor=color, linewidth=0.8, alpha=0.92))
        arrow_dx = max(span * 0.012, 400); y_arrow = gene_box_y + gene_box_h + 0.09
        if strand == '-':
            ax.annotate('', xy=(tss - arrow_dx, y_arrow), xytext=(tss, y_arrow), arrowprops=dict(arrowstyle='-|>', lw=0.9, color=color, shrinkA=0, shrinkB=0))
        else:
            ax.annotate('', xy=(tss + arrow_dx, y_arrow), xytext=(tss, y_arrow), arrowprops=dict(arrowstyle='-|>', lw=0.9, color=color, shrinkA=0, shrinkB=0))
        ax.vlines(tss, gene_box_y - 0.03, gene_box_y + gene_box_h + 0.03, color=color, linewidth=0.8)
        ax.text((start + end) / 2, gene_box_y + gene_box_h + 0.10, row['gene_name'], ha='center', va='bottom', fontsize=6.0, color=color)

    enh_y = -0.44; enh_h = 0.12
    centers = links_df['peak_center'].to_numpy(dtype=float)
    if len(centers) > 1:
        min_gap = float(np.min(np.diff(np.sort(centers))))
    else:
        min_gap = span * 0.02
    box_w = max(min(span * 0.0025, max(min_gap * 0.55, 80.0)), 35.0)
    for _, row in links_df.iterrows():
        xmid = float(row['peak_center'])
        ax.add_patch(mpl.patches.Rectangle((xmid - box_w / 2, enh_y), box_w, enh_h, facecolor=ENH_COLOR, edgecolor=ENH_COLOR, linewidth=0.7, alpha=0.95))
        ax.text(xmid, enh_y - 0.10, str(row['panel_id']), ha='center', va='top', fontsize=5.5, color='#333333')


def parse_region(region: str):
    m = re.match(r'^(chr[^:]+):(\d+)-(\d+)$', str(region))
    if not m:
        return None
    return m.group(1), int(m.group(2)), int(m.group(3))


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rank_df = pd.read_csv(GENE_TSV, sep='\t')
    rank_df = rank_df[rank_df['absolute_rank_maskgt2pct'].notna()].copy()
    rank_df = rank_df.sort_values('absolute_rank_maskgt2pct').head(TOPN)
    enh_df = pd.read_csv(ENH_TSV, sep='\t')

    rna = ad.read_h5ad(RNA_PATH, backed='r')
    atac = ad.read_h5ad(ATAC_PATH, backed='r')
    if not np.array_equal(rna.obs_names.to_numpy(), atac.obs_names.to_numpy()):
        raise ValueError('GW12 rectified_h RNA and ATAC obs order differs')
    xy = np.column_stack([rna.obs['grid_col'].to_numpy(dtype=float), rna.obs['grid_row'].to_numpy(dtype=float)])
    rna_index = pd.Index(rna.var_names.astype(str))
    atac_index = pd.Index(atac.var_names.astype(str))

    for _, grow in rank_df.iterrows():
        gene = str(grow['gene'])
        print(gene, flush=True)
        gene_record = gene_lookup(gene)
        neighbor_df = region_neighbors(gene_record['chrom'], max(1, gene_record['start'] - WINDOW_BP), gene_record['end'] + WINDOW_BP, gene)
        sub = enh_df[enh_df['gene'] == gene].copy()
        if sub.empty:
            continue
        starts = []; ends=[]; centers=[]
        for reg in sub['region']:
            parsed = parse_region(reg)
            if parsed is None:
                starts.append(np.nan); ends.append(np.nan); centers.append(np.nan)
            else:
                _, s, e = parsed
                starts.append(s); ends.append(e); centers.append((s+e)//2)
        sub['peak_start'] = starts; sub['peak_end'] = ends; sub['peak_center'] = centers
        sub = sub.sort_values(['peak_start','peak_end']).reset_index(drop=True)
        sub['panel_id'] = [f'E{i+1}' for i in range(len(sub))]

        expr_idx = rna_index.get_indexer([gene])[0]
        expr = rna.X[:, expr_idx]
        if hasattr(expr, 'toarray'):
            expr = expr.toarray().reshape(-1)
        else:
            expr = np.asarray(expr).reshape(-1)
        expr = np.log1p(np.maximum(expr, 0.0))
        expr_grid = dense_to_grid(rna.obs, expr)
        gene_norm = normalize01(expr_grid)
        latent_mask = majority_regularize(fit_two_state_mask(gene_norm, n_iter=25), n_rounds=3, threshold=5)

        feature_rows = []
        feature_rows.append({'kind': 'gene', 'label': gene, 'source': 'measured raw RNA', 'raw': expr_grid, 'latent': expr_grid * latent_mask, 'genomic_x': int(gene_record['tss'])})
        for _, nrow in neighbor_df.iterrows():
            ng = nrow['gene_name']
            idx = rna_index.get_indexer([ng])[0]
            if idx < 0:
                continue
            ne = rna.X[:, idx]
            if hasattr(ne, 'toarray'):
                ne = ne.toarray().reshape(-1)
            else:
                ne = np.asarray(ne).reshape(-1)
            ne = np.log1p(np.maximum(ne, 0.0))
            ne_grid = dense_to_grid(rna.obs, ne)
            feature_rows.append({'kind': 'neighbor', 'label': ng, 'source': 'measured raw RNA', 'raw': ne_grid, 'latent': ne_grid * latent_mask, 'genomic_x': int(nrow['tss'])})
        for _, row in sub.iterrows():
            idx = atac_index.get_indexer([str(row['region'])])[0]
            if idx < 0:
                continue
            vals = atac.X[:, idx]
            if hasattr(vals, 'toarray'):
                vals = vals.toarray().reshape(-1)
            else:
                vals = np.asarray(vals).reshape(-1)
            vals = np.log1p(np.maximum(vals, 0.0))
            raw_grid = dense_to_grid(atac.obs, vals)
            feature_rows.append({'kind': 'enhancer', 'label': row['panel_id'], 'raw': raw_grid, 'latent': raw_grid * latent_mask, 'diff_score': float(row['difference_structure_score']), 'genomic_x': int(row['peak_center'])})
        feature_rows = sorted(feature_rows, key=lambda d: (d['genomic_x'], 0 if d['kind'] != 'enhancer' else 1, d['label']))
        enhancer_col_pos = {item['label']: i for i, item in enumerate(feature_rows) if item['kind'] == 'enhancer'}
        score_x = np.array([enhancer_col_pos.get(pid, np.nan) for pid in sub['panel_id']], dtype=float)

        n_features = len(feature_rows)
        fig_h = 7.9
        fig_w = max(16.0, min(42.0, 2.15 * n_features + 1.6))
        fig = plt.figure(figsize=(fig_w, fig_h), facecolor='none')

        left = 0.035; right = 0.965
        ax_locus = fig.add_axes([left, 0.80, right-left, 0.14], facecolor='none')
        draw_locus_axis(ax_locus, gene_record, neighbor_df, sub)
        ax_d = fig.add_axes([left, 0.64, right-left, 0.075], facecolor='none')
        add_score_row(ax_d, score_x, sub['difference_structure_score'].to_numpy(dtype=float), ylabel='diffS', xlim=(-0.5, n_features-0.5))

        total_w = right - left
        col_gap = 0.004
        col_w = (total_w - col_gap * max(n_features - 1, 0)) / max(n_features, 1)
        raw_label_y = 0.47; raw_label_h = 0.035; raw_plot_y = 0.275; raw_plot_h = 0.195
        latent_label_y = 0.155; latent_label_h = 0.040; latent_plot_y = 0.015; latent_plot_h = 0.145

        for i, item in enumerate(feature_rows):
            x0 = left + i * (col_w + col_gap)
            ax_raw_label = fig.add_axes([x0, raw_label_y, col_w, raw_label_h], facecolor='none')
            ax_raw = fig.add_axes([x0, raw_plot_y, col_w, raw_plot_h], facecolor='none')
            ax_lat_label = fig.add_axes([x0, latent_label_y, col_w, latent_label_h], facecolor='none')
            ax_lat = fig.add_axes([x0, latent_plot_y, col_w, latent_plot_h], facecolor='none')
            raw_show, _ = per_panel_p99_normalize(item['raw'], percentile=99.0)
            lat_show, _ = per_panel_p99_normalize(item['latent'], percentile=99.0)
            if item['kind'] == 'enhancer':
                raw_title = format_panel_title([item['label']], wrap_width=9)
                lat_title = format_panel_title([item['label'], f"diffS={item['diff_score']:.2f}"], wrap_width=9)
                raw_cmap = PEAK_CMAP; lat_cmap = PEAK_CMAP
            else:
                raw_title = format_panel_title([item['label'], 'RNA'], wrap_width=10)
                lat_title = format_panel_title([item['label'], 'latent'], wrap_width=10)
                raw_cmap = RAW_CMAP; lat_cmap = RAW_CMAP
            add_panel_label(ax_raw_label, raw_title)
            add_scatter(ax_raw, xy, raw_show[rna.obs['grid_row'].to_numpy(dtype=int), rna.obs['grid_col'].to_numpy(dtype=int)], raw_cmap)
            add_panel_label(ax_lat_label, lat_title)
            add_scatter(ax_lat, xy, lat_show[rna.obs['grid_row'].to_numpy(dtype=int), rna.obs['grid_col'].to_numpy(dtype=int)], lat_cmap)

        fig.suptitle(f"{gene} GW12 rectified_h locus + spatial raw/latent view", fontsize=11, y=0.997)
        out = OUTDIR / f'{gene}_locus_spatial_longpdf.pdf'
        fig.savefig(out, bbox_inches='tight', transparent=True, facecolor='none')
        plt.close(fig)

    rna.file.close(); atac.file.close()


if __name__ == '__main__':
    main()
