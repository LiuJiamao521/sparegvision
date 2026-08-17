from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.stats import rankdata, t as student_t


def decode_arr(arr):
    out = []
    for x in arr:
        if isinstance(x, bytes):
            out.append(x.decode())
        else:
            out.append(str(x))
    return out


def parse_gtf_gene_and_tss(gtf_path: Path, gene_name: str):
    gene_record = None
    chrom_tss = {}
    pat = re.compile(r'gene_name "([^"]+)"')
    with gtf_path.open() as fh:
        for line in fh:
            if line.startswith('#'):
                continue
            fields = line.rstrip('\n').split('\t')
            if len(fields) < 9 or fields[2] != 'gene':
                continue
            chrom = fields[0]
            start = int(fields[3])
            end = int(fields[4])
            strand = fields[6]
            attrs = fields[8]
            m = pat.search(attrs)
            if not m:
                continue
            gname = m.group(1)
            tss = end if strand == '-' else start
            chrom_tss.setdefault(f'chr{chrom}', []).append(tss)
            if gname == gene_name:
                gene_record = {
                    'gene_name': gname,
                    'chrom': f'chr{chrom}',
                    'start': start,
                    'end': end,
                    'strand': strand,
                    'tss': tss,
                    'gtf_line': line.rstrip('\n'),
                }
    if gene_record is None:
        raise ValueError(f'Gene not found in GTF: {gene_name}')
    return gene_record, chrom_tss


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gene', required=True)
    ap.add_argument('--gtf', required=True)
    ap.add_argument('--rna-h5ad', required=True)
    ap.add_argument('--atac-h5ad', required=True)
    ap.add_argument('--window-bp', type=int, default=100_000)
    ap.add_argument('--exclude-tss-bp', type=int, default=2_000)
    ap.add_argument('--out-prefix', required=True)
    args = ap.parse_args()

    gene_record, chrom_tss = parse_gtf_gene_and_tss(Path(args.gtf), args.gene)

    with h5py.File(args.rna_h5ad, 'r') as rna:
        gene_names = decode_arr(rna['var']['_index'][:])
        imputed_names = decode_arr(rna['uns']['imputed_gene_names'][:]) if 'imputed_gene_names' in rna['uns'] else []
        if args.gene in gene_names:
            gene_source = 'measured_X'
            gene_idx = gene_names.index(args.gene)
            gene_expr = rna['X'][:, gene_idx].astype(np.float64)
        elif args.gene in imputed_names:
            gene_source = 'imputed_X_imputed'
            gene_idx = imputed_names.index(args.gene)
            gene_expr = rna['obsm']['X_imputed'][:, gene_idx].astype(np.float64)
        else:
            raise ValueError(f'Gene not found in measured or imputed matrices: {args.gene}')

    with h5py.File(args.atac_h5ad, 'r') as atac:
        chroms = decode_arr(atac['var']['chrom'][:])
        starts = atac['var']['start'][:].astype(np.int64)
        ends = atac['var']['end'][:].astype(np.int64)
        intervals = decode_arr(atac['var']['_index'][:])

        candidate_idx = []
        candidate_rows = []
        tss_list = chrom_tss.get(gene_record['chrom'], [])
        for i, (chrom, start, end, interval) in enumerate(zip(chroms, starts, ends, intervals)):
            if chrom != gene_record['chrom']:
                continue
            center = (int(start) + int(end)) // 2
            if abs(center - gene_record['tss']) > args.window_bp:
                continue
            if any(abs(center - x) <= args.exclude_tss_bp for x in tss_list):
                continue
            candidate_idx.append(i)
            candidate_rows.append({
                'target_gene': args.gene,
                'target_chrom': gene_record['chrom'],
                'target_start': gene_record['start'],
                'target_end': gene_record['end'],
                'target_strand': gene_record['strand'],
                'target_tss': gene_record['tss'],
                'gene_expr_source': gene_source,
                'region': f'{chrom}:{start}-{end}',
                'interval_id': interval,
                'peak_chrom': chrom,
                'peak_start': int(start),
                'peak_end': int(end),
                'peak_center': center,
                'distance_to_tss': center - gene_record['tss'],
                'abs_distance_to_tss': abs(center - gene_record['tss']),
            })

        if not candidate_idx:
            raise ValueError('No candidate peaks remained after filtering.')

        peak_matrix = atac['X'][:, candidate_idx].astype(np.float64)

    gene_ranks = rankdata(gene_expr, method='average')
    rho = pearson_on_ranks(gene_ranks, peak_matrix)
    n = gene_expr.shape[0]
    df = n - 2
    rho_clip = np.clip(rho, -0.999999999, 0.999999999)
    t_stat = rho_clip * np.sqrt(df / (1.0 - rho_clip ** 2))
    p_one_tailed = student_t.sf(t_stat, df)

    out_df = pd.DataFrame(candidate_rows)
    out_df['spearman_rho'] = rho
    out_df['t_stat'] = t_stat
    out_df['pvalue_one_tailed_positive'] = p_one_tailed
    out_df['significant_positive_p_lt_0_01'] = out_df['pvalue_one_tailed_positive'] < 0.01
    out_df['gene_expr_nonzero_cells'] = int(np.sum(gene_expr > 0))
    out_df['gene_expr_nonzero_fraction'] = float(np.mean(gene_expr > 0))
    out_df['peak_nonzero_cells'] = np.sum(peak_matrix > 0, axis=0).astype(int)
    out_df['peak_nonzero_fraction'] = np.mean(peak_matrix > 0, axis=0)
    out_df['peak_signal_sum'] = np.sum(peak_matrix, axis=0)
    out_df = out_df.sort_values(['pvalue_one_tailed_positive', 'spearman_rho', 'abs_distance_to_tss'], ascending=[True, False, True])

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    all_path = out_prefix.with_suffix('.all.tsv')
    sig_path = out_prefix.with_suffix('.significant.tsv')
    meta_path = out_prefix.with_suffix('.metadata.json')
    out_df.to_csv(all_path, sep='\t', index=False)
    out_df.loc[out_df['significant_positive_p_lt_0_01']].to_csv(sig_path, sep='\t', index=False)

    metadata = {
        'gene': args.gene,
        'gene_record': gene_record,
        'rna_h5ad': args.rna_h5ad,
        'atac_h5ad': args.atac_h5ad,
        'gene_expr_source': gene_source,
        'window_bp': args.window_bp,
        'exclude_tss_bp': args.exclude_tss_bp,
        'n_cells': int(n),
        'n_candidate_peaks_tested': int(out_df.shape[0]),
        'n_significant_positive_links_p_lt_0_01': int(out_df['significant_positive_p_lt_0_01'].sum()),
        'output_all': str(all_path),
        'output_significant': str(sig_path),
        'rule': 'candidate peaks within +/-100 kb of target TSS, excluding peaks whose centers fall within +/-2 kb of any annotated TSS on the same chromosome; Spearman correlation with one-tailed t-test p<0.01 for positive links',
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')

    top = out_df.head(10)[['region', 'distance_to_tss', 'spearman_rho', 'pvalue_one_tailed_positive', 'significant_positive_p_lt_0_01']]
    print(json.dumps(metadata, indent=2))
    print('\nTop hits:')
    print(top.to_string(index=False))


if __name__ == '__main__':
    main()
