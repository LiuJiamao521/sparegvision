from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import textwrap
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.patches import ConnectionPatch
from matplotlib.ticker import FuncFormatter
from scipy.sparse.csgraph import connected_components
from scipy.stats import rankdata, t as student_t
from sklearn.neighbors import NearestNeighbors

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sparegvision.gsps import moran_i


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RNA = Path("/cluster3/labData/jiamao/SpaRegVision/weMERFISH/weMERFISH_combined_C_6s_E1_rescaled_z.h5ad")
DEFAULT_ATAC = Path("/cluster3/labData/jiamao/SpaRegVision/weMERFISH/weMERFISH_spatial_ATAC_C_6s_E1.h5ad")
DEFAULT_GTF = Path("/cluster3/labData/jiamao/Genome/zebrafish/Danio_rerio.GRCz11.113.gtf")
DEFAULT_OUTDIR = ROOT / "results" / "weMERFISH"

EPS = 1e-3
SIG_P = 0.01
K_GRAPH = 6
K_NEIGHBORS = 8
N_ROUNDS = 3
MAJORITY_THRESHOLD = 5
POINT_SIZE = 0.2

RAW_CMAP = mpl.cm.Blues
PEAK_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "peak_cmap",
    ["#EDEDED", "#FFF7F3", "#FDE0DD", "#FCC5C0", "#FA9FB5", "#F768A1", "#DD3497", "#AE017E"],
    N=256,
)
GENE_COLOR_TARGET = "#1F4E79"
GENE_COLOR_NEIGHBOR = "#666666"
ENH_COLOR_BASE = "#888888"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
    }
)


def decode_arr(arr):
    return [x.decode() if isinstance(x, bytes) else str(x) for x in arr]


def parse_gtf_all_genes(gtf_path: Path):
    gene_records = []
    chrom_tss = {}
    pat = re.compile(r'gene_name "([^"]+)"')
    with gtf_path.open() as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] != "gene":
                continue
            m = pat.search(fields[8])
            if not m:
                continue
            gene = m.group(1)
            chrom = f"chr{fields[0]}"
            start = int(fields[3])
            end = int(fields[4])
            strand = fields[6]
            tss = end if strand == "-" else start
            rec = {
                "gene_name": gene,
                "chrom": chrom,
                "start": start,
                "end": end,
                "strand": strand,
                "tss": tss,
                "gtf_line": line.rstrip("\n"),
            }
            gene_records.append(rec)
            chrom_tss.setdefault(chrom, []).append(tss)
    gene_df = pd.DataFrame(gene_records).drop_duplicates(subset=["gene_name"], keep="first")
    return gene_df, chrom_tss


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
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto")
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
    peak_ranks = rankdata(peak_matrix, axis=0, method="average")
    peak_centered = peak_ranks - peak_ranks.mean(axis=0, keepdims=True)
    peak_ss = np.sum(peak_centered ** 2, axis=0)
    numer = np.sum(gene_centered[:, None] * peak_centered, axis=0)
    denom = np.sqrt(gene_ss * peak_ss)
    rho = np.divide(numer, denom, out=np.zeros_like(numer, dtype=np.float64), where=denom > 0)
    return rho


def finite_corr(x, y, method="pearson"):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan
    xv = x[m]
    yv = y[m]
    if np.std(xv) < 1e-8 or np.std(yv) < 1e-8:
        return np.nan
    if method == "pearson":
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
    nn = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto")
    nn.fit(subcoords[hot])
    graph = nn.kneighbors_graph(mode="connectivity")
    graph = graph.maximum(graph.T)
    n_comp, labels = connected_components(graph, directed=False)
    sizes = np.bincount(labels)
    largest = int(sizes.max()) if sizes.size else 0
    return int(n_comp), float(largest / max(hot.sum(), 1)), thr


def positive(x):
    return float(max(float(x), 0.0))


def read_gene_expr(f, gene_name, measured_map, imputed_map):
    if gene_name in measured_map:
        vals = np.asarray(f["X"][:, measured_map[gene_name]], dtype=float)
        return np.maximum(vals, 0.0), "measured raw RNA"
    if gene_name in imputed_map:
        vals = np.asarray(f["obsm"]["X_imputed"][:, imputed_map[gene_name]], dtype=float)
        return np.maximum(vals, 0.0), "imputed RNA"
    vals = np.zeros(f["X"].shape[0], dtype=float)
    return vals, "missing"


def format_panel_title(lines, wrap_width=10):
    out = []
    for line in lines:
        if line is None:
            continue
        s = str(line).strip()
        if not s:
            continue
        wrapped = textwrap.fill(s, width=wrap_width, break_long_words=False, break_on_hyphens=False)
        out.append(wrapped)
    return "\n".join(out)


def add_panel_label(ax, title):
    ax.set_zorder(3)
    ax.set_facecolor("none")
    ax.patch.set_alpha(0.0)
    ax.axis("off")
    ax.text(
        0.02,
        0.02,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=4.5,
        linespacing=1.0,
        clip_on=True,
    )


def add_scatter(ax, xy, values, cmap):
    ax.set_zorder(1)
    ax.set_facecolor("none")
    ax.patch.set_alpha(0.0)
    ax.scatter(xy[:, 0], xy[:, 1], c=values, cmap=cmap, s=POINT_SIZE, linewidths=0, rasterized=True)
    xmin = float(np.nanmin(xy[:, 0]))
    xmax = float(np.nanmax(xy[:, 0]))
    ymin = float(np.nanmin(xy[:, 1]))
    ymax = float(np.nanmax(xy[:, 1]))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymax, ymin)
    ax.margins(x=0.0, y=0.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box", anchor="N")
    for spine in ax.spines.values():
        spine.set_visible(False)


def add_score_row(ax, xvals, yvals, ylabel, invert=False, threshold=None, xlim=None):
    x = np.asarray(xvals, dtype=float)
    y = np.asarray(yvals, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    ax.set_facecolor("none")
    ax.patch.set_alpha(0.0)
    if m.any():
        if xlim is not None:
            ax.set_xlim(*xlim)
        else:
            xmin = float(np.nanmin(x[m]))
            xmax = float(np.nanmax(x[m]))
            pad = max((xmax - xmin) * 0.015, 1.0)
            ax.set_xlim(xmin - pad, xmax + pad)
        ymin = float(np.nanmin(y[m]))
        ymax = float(np.nanmax(y[m]))
        if abs(ymax - ymin) < 1e-12:
            ymin -= 0.5
            ymax += 0.5
        else:
            pad_y = (ymax - ymin) * 0.12
            ymin -= pad_y
            ymax += pad_y
        bottom = ymax if invert else ymin
        top = ymin if invert else ymax
        ax.set_ylim((bottom, top))
        base = 0.0 if (ymin <= 0.0 <= ymax) else (ymax if invert else ymin)
        ax.vlines(x[m], base, y[m], color="#4D4D4D", linewidth=0.8, zorder=3)
        ax.scatter(x[m], y[m], s=22, color="#4D4D4D", marker='D', linewidths=0, zorder=4)
    elif xlim is not None:
        ax.set_xlim(*xlim)
    ax.set_ylabel(ylabel, fontsize=6.0, rotation=0, ha="right", va="center", labelpad=18)
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
    ax.tick_params(axis="y", labelsize=5.2, length=2)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.spines["bottom"].set_color("#B0B0B0")
    ax.spines["left"].set_color("#B0B0B0")
    if threshold is not None:
        ax.axhline(threshold, color="#9E9E9E", linestyle="--", linewidth=0.6, zorder=1)


def draw_locus_axis(ax, gene_record, neighbor_df, links_df, color_metric="diff_rank"):
    chrom = gene_record["chrom"]
    xs = [int(gene_record["start"]), int(gene_record["end"])]
    if len(neighbor_df):
        xs.extend(neighbor_df["start"].astype(int).tolist())
        xs.extend(neighbor_df["end"].astype(int).tolist())
    if len(links_df):
        xs.extend(links_df["peak_start"].astype(int).tolist())
        xs.extend(links_df["peak_end"].astype(int).tolist())
    xmin = int(min(xs) - 2500)
    xmax = int(max(xs) + 2500)
    span = xmax - xmin
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-1.0, 1.0)
    ax.set_yticks([])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(round(x)):,}"))
    ax.set_xlabel(f"{chrom} genomic position (Mb)", labelpad=22)
    ax.set_title(f"{gene_record['gene_name']} locus and significant enhancers", fontsize=9, pad=8)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_position(("data", 0.0))
    ax.spines["bottom"].set_linewidth(1.0)
    ax.spines["bottom"].set_color("#444444")
    ax.tick_params(axis="x", bottom=True, labelbottom=True, length=3, pad=2)

    all_genes = pd.concat([pd.DataFrame([gene_record]), neighbor_df], ignore_index=True)
    gene_box_y = 0.28
    gene_box_h = 0.18
    for _, row in all_genes.sort_values(["start", "end", "gene_name"]).iterrows():
        color = GENE_COLOR_TARGET if row["gene_name"] == gene_record["gene_name"] else GENE_COLOR_NEIGHBOR
        start = int(row["start"])
        end = int(row["end"])
        tss = int(row["tss"])
        strand = row.get("strand", "+")
        gene_w = max((end - start) * 0.5, 1.0)
        if strand == "-":
            gene_x = tss - gene_w
        else:
            gene_x = tss
        rect = mpl.patches.Rectangle((gene_x, gene_box_y), gene_w, gene_box_h, facecolor=color, edgecolor=color, linewidth=0.8, alpha=0.92)
        ax.add_patch(rect)
        arrow_dx = max(span * 0.012, 400)
        y_arrow = gene_box_y + gene_box_h + 0.09
        if strand == "-":
            ax.annotate("", xy=(tss - arrow_dx, y_arrow), xytext=(tss, y_arrow), arrowprops=dict(arrowstyle='-|>', lw=0.9, color=color, shrinkA=0, shrinkB=0))
        else:
            ax.annotate("", xy=(tss + arrow_dx, y_arrow), xytext=(tss, y_arrow), arrowprops=dict(arrowstyle='-|>', lw=0.9, color=color, shrinkA=0, shrinkB=0))
        ax.vlines(tss, gene_box_y - 0.03, gene_box_y + gene_box_h + 0.03, color=color, linewidth=0.8)
        ax.text((start + end) / 2, gene_box_y + gene_box_h + 0.10, row["gene_name"], ha="center", va="bottom", fontsize=6.0, color=color)

    if len(links_df) == 0:
        return

    enh_y = -0.44
    enh_h = 0.12
    centers = links_df["peak_center"].to_numpy(dtype=float)
    if len(centers) > 1:
        min_gap = float(np.min(np.diff(np.sort(centers))))
    else:
        min_gap = span * 0.02
    box_w = max(min(span * 0.0025, max(min_gap * 0.55, 80.0)), 35.0)
    enh_color = "#7F7F7F"
    for _, row in links_df.iterrows():
        xmid = float(row["peak_center"])
        rect = mpl.patches.Rectangle((xmid - box_w / 2, enh_y), box_w, enh_h, facecolor=enh_color, edgecolor=enh_color, linewidth=0.7, alpha=0.95)
        ax.add_patch(rect)
        ax.text(xmid, enh_y - 0.10, str(row["panel_id"]), ha="center", va="top", fontsize=5.5, color="#333333")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gene", required=True)
    ap.add_argument("--rna-h5ad", default=str(DEFAULT_RNA))
    ap.add_argument("--atac-h5ad", default=str(DEFAULT_ATAC))
    ap.add_argument("--gtf", default=str(DEFAULT_GTF))
    ap.add_argument("--window-bp", type=int, default=100_000)
    ap.add_argument("--exclude-tss-bp", type=int, default=2_000)
    ap.add_argument("--out-prefix", default=None)
    ap.add_argument("--rank-by", choices=["pvalue", "rho", "diffscore"], default="pvalue")
    ap.add_argument("--max-peaks", type=int, default=None)
    args = ap.parse_args()

    gene = args.gene
    out_prefix = Path(args.out_prefix) if args.out_prefix else (DEFAULT_OUTDIR / f"{gene}_locus_spatial_longpdf")
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_pdf = out_prefix.with_suffix(".pdf")
    out_links = out_prefix.with_suffix(".links.tsv")
    out_summary = out_prefix.with_suffix(".summary.tsv")
    out_latent = out_prefix.with_suffix(".latent.tsv.gz")

    gene_df, chrom_tss = parse_gtf_all_genes(Path(args.gtf))
    if gene not in set(gene_df["gene_name"]):
        raise ValueError(f"Gene not found in GTF: {gene}")
    gene_record = gene_df.loc[gene_df["gene_name"] == gene].iloc[0].to_dict()

    with h5py.File(args.rna_h5ad, "r") as rna, h5py.File(args.atac_h5ad, "r") as atac:
        measured_names = decode_arr(rna["var"]["_index"][:])
        imputed_names = decode_arr(rna["uns"]["imputed_gene_names"][:]) if "imputed_gene_names" in rna["uns"] else []
        measured_map = {g: i for i, g in enumerate(measured_names)}
        imputed_map = {g: i for i, g in enumerate(imputed_names)}

        target_expr, target_source = read_gene_expr(rna, gene, measured_map, imputed_map)
        target_norm = normalize01(target_expr)
        raw_mask, c_low, c_high = fit_two_state_mask(target_norm, n_iter=25)

        coords3d = np.asarray(atac["obsm"]["spatial_rescaled_z"][:], dtype=float)
        coords_xy = coords3d[:, :2]
        valid3d = np.isfinite(coords3d).all(axis=1)
        latent_mask = majority_regularize_knn(raw_mask, coords3d, valid3d, k=K_NEIGHBORS, n_rounds=N_ROUNDS, threshold=MAJORITY_THRESHOLD)
        latent_active_frac = float(latent_mask.mean())

        latent_df = pd.DataFrame(
            {
                "spatial_x": coords3d[:, 0],
                "spatial_y": coords3d[:, 1],
                "spatial_z_rescaled": coords3d[:, 2],
                "valid_3d": valid3d.astype(int),
                "target_expr": target_expr,
                "target_raw_state": raw_mask.astype(int),
                "target_latent_state": latent_mask.astype(int),
            }
        )
        with gzip.open(out_latent, "wt") as fh:
            latent_df.to_csv(fh, sep="\t", index=False)

        chroms = decode_arr(atac["var"]["chrom"][:])
        starts = atac["var"]["start"][:].astype(np.int64)
        ends = atac["var"]["end"][:].astype(np.int64)
        intervals = decode_arr(atac["var"]["_index"][:])
        centers = ((starts + ends) // 2).astype(np.int64)
        chrom = gene_record["chrom"]
        tss = int(gene_record["tss"])
        tss_list = chrom_tss.get(chrom, [])

        candidate_idx = []
        candidate_rows = []
        for i, (pchrom, start, end, interval, center) in enumerate(zip(chroms, starts, ends, intervals, centers)):
            if pchrom != chrom:
                continue
            if abs(int(center) - tss) > args.window_bp:
                continue
            if any(abs(int(center) - x) <= args.exclude_tss_bp for x in tss_list):
                continue
            candidate_idx.append(i)
            candidate_rows.append(
                {
                    "gene": gene,
                    "target_tss": tss,
                    "region": f"{pchrom}:{int(start)}-{int(end)}",
                    "interval_id": interval,
                    "peak_chrom": pchrom,
                    "peak_start": int(start),
                    "peak_end": int(end),
                    "peak_center": int(center),
                    "distance_to_tss": int(center) - tss,
                    "abs_distance_to_tss": abs(int(center) - tss),
                }
            )
        if not candidate_idx:
            raise ValueError("No candidate peaks remained after filtering.")

        peak_matrix = np.asarray(atac["X"][:, candidate_idx], dtype=float)
        gene_ranks = rankdata(target_expr, method="average")
        rho = pearson_on_ranks(gene_ranks, peak_matrix)
        n = target_expr.shape[0]
        df = n - 2
        rho_clip = np.clip(rho, -0.999999999, 0.999999999)
        t_stat = rho_clip * np.sqrt(df / (1.0 - rho_clip**2))
        p_one_tailed = student_t.sf(t_stat, df)
        links = pd.DataFrame(candidate_rows)
        links["spearman_rho"] = rho
        links["pvalue_one_tailed_positive"] = p_one_tailed
        links["significant_positive_p_lt_0_01"] = links["pvalue_one_tailed_positive"] < SIG_P
        links["panel_id"] = [f"E{i+1}" for i in range(len(links))]

        sig = links.loc[links["significant_positive_p_lt_0_01"]].copy().sort_values(["peak_start", "peak_end"]).reset_index(drop=True)
        if args.max_peaks is not None:
            sig = sig.iloc[: args.max_peaks].copy()
        if sig.empty:
            raise ValueError("No significant positive peaks remained after filtering.")
        sig["panel_id"] = [f"E{i+1}" for i in range(len(sig))]

        interval_to_candidate_j = {intervals[idx]: j for j, idx in enumerate(candidate_idx)}
        active_valid = (latent_mask > 0) & valid3d
        control = target_expr * latent_mask
        control_norm, control_p99 = per_panel_p99_normalize(control, percentile=99.0)
        active_valid &= np.isfinite(control_norm)

        score_rows = []
        for _, row in sig.iterrows():
            cj = interval_to_candidate_j[row["interval_id"]]
            raw_vals = np.log1p(np.maximum(peak_matrix[:, cj], 0.0))
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
            _, diff_hot_frac, _ = hotspot_component_stats(np.abs(diff), coords_valid, q=0.9, k=K_GRAPH)
            _, logfc_hot_frac, _ = hotspot_component_stats(np.abs(logfc), coords_valid, q=0.9, k=K_GRAPH)
            _, resid_hot_frac, _ = hotspot_component_stats(np.abs(resid), coords_valid, q=0.9, k=K_GRAPH)
            diff_structure_score = float(np.nanmean(np.abs(diff)) * (1.0 + positive(diff_abs_moran)) * diff_hot_frac)
            logfc_structure_score = float(np.nanmean(np.abs(logfc)) * (1.0 + positive(logfc_abs_moran)) * logfc_hot_frac)
            residual_structure_score = float(np.nanstd(resid) * (1.0 + positive(resid_abs_moran)) * resid_hot_frac)
            difference_structure_score = float(0.4 * diff_structure_score + 0.25 * logfc_structure_score + 0.35 * residual_structure_score)
            score_rows.append(
                {
                    "interval_id": row["interval_id"],
                    "latent_state_rho": finite_corr(x, y, method="pearson"),
                    "difference_structure_score": difference_structure_score,
                    "enhancer_p99": enh_p99,
                }
            )

        score_df = pd.DataFrame(score_rows)
        sig = sig.merge(score_df, on="interval_id", how="left")
        sig["pvalue_rank"] = sig["pvalue_one_tailed_positive"].rank(method="dense", ascending=True).astype(int)
        sig["rho_rank"] = sig["spearman_rho"].rank(method="dense", ascending=False).astype(int)
        sig["diff_rank"] = sig["difference_structure_score"].rank(method="dense", ascending=False, na_option="bottom").astype(int)
        sig = sig.sort_values(["peak_start", "peak_end"]).reset_index(drop=True)
        sig["panel_id"] = [f"E{i+1}" for i in range(len(sig))]
        sig.to_csv(out_links, sep="\t", index=False)

        neighbor_df = gene_df.loc[
            (gene_df["chrom"] == chrom)
            & (gene_df["gene_name"] != gene)
            & (np.abs(gene_df["tss"].astype(int) - tss) <= args.window_bp)
        ].copy().sort_values(["start", "end"]).reset_index(drop=True)

        # layout
        feature_rows = []
        feature_rows.append({"kind": "gene", "label": gene, "source": target_source, "raw": target_expr, "latent": target_expr * latent_mask, "genomic_x": tss})
        for _, nrow in neighbor_df.iterrows():
            n_gene = nrow["gene_name"]
            n_expr, n_source = read_gene_expr(rna, n_gene, measured_map, imputed_map)
            feature_rows.append({"kind": "neighbor", "label": n_gene, "source": n_source, "raw": n_expr, "latent": n_expr * latent_mask, "genomic_x": int(nrow["tss"])})
        for _, prow in sig.iterrows():
            cj = interval_to_candidate_j[prow["interval_id"]]
            raw_vals = np.log1p(np.maximum(peak_matrix[:, cj], 0.0))
            feature_rows.append(
                {
                    "kind": "enhancer",
                    "label": prow["panel_id"],
                    "source": "significant peak",
                    "raw": raw_vals,
                    "latent": raw_vals * latent_mask,
                    "init_rho": float(prow["spearman_rho"]),
                    "latent_rho": float(prow["latent_state_rho"]) if pd.notna(prow["latent_state_rho"]) else np.nan,
                    "diff_score": float(prow["difference_structure_score"]) if pd.notna(prow["difference_structure_score"]) else np.nan,
                    "genomic_x": int(prow["peak_center"]),
                }
            )

        feature_rows = sorted(feature_rows, key=lambda d: (d["genomic_x"], 0 if d["kind"] != "enhancer" else 1, d["label"]))
        enhancer_col_pos = {item["label"]: i for i, item in enumerate(feature_rows) if item["kind"] == "enhancer"}

        n_features = len(feature_rows)
        fig_h = 7.90
        fig_w = max(16.0, min(42.0, 2.15 * n_features + 1.6))
        fig = plt.figure(figsize=(fig_w, fig_h), facecolor="none")

        left = 0.035
        right = 0.965

        ax_locus = fig.add_axes([left, 0.80, right - left, 0.14], facecolor="none")
        draw_locus_axis(
            ax_locus,
            gene_record=gene_record,
            neighbor_df=neighbor_df,
            links_df=sig,
            color_metric={"pvalue": "pvalue_rank", "rho": "rho_rank", "diffscore": "diff_rank"}[args.rank_by],
        )

        score_x = np.array([enhancer_col_pos.get(pid, np.nan) for pid in sig["panel_id"]], dtype=float)

        ax_d = fig.add_axes([left, 0.64, right - left, 0.075], facecolor="none")
        add_score_row(
            ax_d,
            score_x,
            sig["difference_structure_score"].to_numpy(dtype=float),
            ylabel="diffS",
            invert=False,
            threshold=None,
            xlim=(-0.5, n_features - 0.5),
        )

        right = 0.965
        total_w = right - left
        col_gap = 0.004
        col_w = (total_w - col_gap * max(n_features - 1, 0)) / max(n_features, 1)

        raw_label_y = 0.47
        raw_label_h = 0.035
        raw_plot_y = 0.275
        raw_plot_h = 0.195
        latent_label_y = 0.155
        latent_label_h = 0.040
        latent_plot_y = 0.015
        latent_plot_h = 0.145

        raw_axes = []
        for i, item in enumerate(feature_rows):
            x0 = left + i * (col_w + col_gap)
            ax_raw_label = fig.add_axes([x0, raw_label_y, col_w, raw_label_h], facecolor="none")
            ax_raw = fig.add_axes([x0, raw_plot_y, col_w, raw_plot_h], facecolor="none")
            ax_lat_label = fig.add_axes([x0, latent_label_y, col_w, latent_label_h], facecolor="none")
            ax_lat = fig.add_axes([x0, latent_plot_y, col_w, latent_plot_h], facecolor="none")
            raw_axes.append(ax_raw)
            raw_show, _ = per_panel_p99_normalize(item["raw"], percentile=99.0)
            lat_show, _ = per_panel_p99_normalize(item["latent"], percentile=99.0)
            if item["kind"] == "enhancer":
                raw_title = format_panel_title([item['label'], f"rawρ={item['init_rho']:.2f}"], wrap_width=9)
                lat_title = format_panel_title([item['label'], f"latentρ={item['latent_rho']:.2f}", f"diffS={item['diff_score']:.2f}"], wrap_width=9)
                raw_cmap = PEAK_CMAP
                lat_cmap = PEAK_CMAP
            elif item["kind"] == "gene":
                src = item["source"]
                if src == "measured raw RNA":
                    raw_title = format_panel_title([item['label'], 'measured', 'RNA'], wrap_width=10)
                elif src == "imputed RNA":
                    raw_title = format_panel_title([item['label'], 'imputed', 'RNA'], wrap_width=10)
                else:
                    raw_title = format_panel_title([item['label'], 'RNA'], wrap_width=10)
                lat_title = format_panel_title([item['label'], 'latent'], wrap_width=10)
                raw_cmap = RAW_CMAP
                lat_cmap = RAW_CMAP
            else:
                raw_title = format_panel_title([item['label'], 'RNA'], wrap_width=10)
                lat_title = format_panel_title([item['label'], 'latent'], wrap_width=10)
                raw_cmap = RAW_CMAP
                lat_cmap = RAW_CMAP
            add_panel_label(ax_raw_label, raw_title)
            add_scatter(ax_raw, coords_xy, raw_show, raw_cmap)
            ax_lat_label.set_zorder(6)
            ax_lat.set_zorder(5)
            add_panel_label(ax_lat_label, lat_title)
            add_scatter(ax_lat, coords_xy, lat_show, lat_cmap)

        fig.suptitle(f"{gene} weMERFISH locus + spatial raw/latent view", fontsize=11, y=0.997)
        fig.text(
            0.01,
            0.005,
            f"Window = ±{args.window_bp:,} bp around {gene} TSS; exclude any TSS ±{args.exclude_tss_bp:,} bp; "
            f"peak significance = one-tailed positive Spearman p < {SIG_P:.2g}; middle track shows diffS. "
            f"Target latent uses raw RNA -> min-max -> two-state fit -> 3D kNN majority regularization. "
            f"Spatial panels are independently p99-clipped and rescaled to 0–1 for shape comparison.",
            ha="left",
            va="bottom",
            fontsize=6.4,
            color="#555555",
        )
        fig.savefig(out_pdf, bbox_inches="tight", transparent=True, facecolor="none")
        plt.close(fig)

        summary = pd.DataFrame(
            [
                {
                    "gene": gene,
                    "gene_source": target_source,
                    "window_bp": args.window_bp,
                    "exclude_tss_bp": args.exclude_tss_bp,
                    "n_neighbor_genes": int(len(neighbor_df)),
                    "n_candidate_peaks": int(len(links)),
                    "n_significant_peaks": int(len(sig)),
                    "latent_mask_fraction": latent_active_frac,
                    "mask_low_center": c_low,
                    "mask_high_center": c_high,
                    "control_p99": control_p99,
                    "out_pdf": str(out_pdf),
                    "out_links_tsv": str(out_links),
                    "out_latent_tsv_gz": str(out_latent),
                }
            ]
        )
        summary.to_csv(out_summary, sep="\t", index=False)

    print(out_pdf)
    print(out_links)
    print(out_summary)


if __name__ == "__main__":
    main()
