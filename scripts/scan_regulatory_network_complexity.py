"""Genome-wide mask-aware gene--enhancer network complexity scan."""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import h5py
from scipy.ndimage import gaussian_filter, label, zoom

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sparegvision.config import load_config
from sparegvision.io import open_h5ad, paired_observations, read_feature_matrix


def main():
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / 'configs' / 'default.yaml')
    links_path = root / 'data' / 'GW12_region_to_gene_links_filtered.tsv'
    out = root / 'results' / 'gw12_regulatory_network_complexity'
    out.mkdir(parents=True, exist_ok=True)
    links = pd.read_csv(links_path, sep='\t')
    genes = links.target.drop_duplicates().tolist()
    regions = links.region.drop_duplicates().tolist()
    rna = open_h5ad(root / 'data' / 'GW12_spatial_RNA.h5ad')
    atac = open_h5ad(root / 'data' / 'GW12_spatial_ATAC.h5ad')
    obs, coords = paired_observations(rna, atac, cfg['spatial_crop'])
    canvas = cfg['spatial_canvas']; xmin, xmax, ymin, ymax = map(float, canvas)
    px = float(cfg['pixel_size']); nx = int(np.floor((xmax-xmin)/px+1e-8))+1; ny = int(np.floor((ymax-ymin)/px+1e-8))+1
    gx = np.floor((coords[:, 0]-xmin)/px+1e-8).astype(int); gy = np.floor((coords[:, 1]-ymin)/px+1e-8).astype(int)
    flat = gy*nx + gx; counts = np.bincount(flat, minlength=nx*ny).astype(float); mask = counts > 0
    shape = (ny, nx); mask2 = mask.reshape(shape)
    low_mask = zoom(mask2.astype(float), (19/ny, 25/nx), order=0) > .5

    def image(values):
        sums = np.bincount(flat, weights=values, minlength=nx*ny)
        base = (sums / np.maximum(counts, 1)).reshape(shape)
        v = base[mask2]
        if len(v) == 0 or np.allclose(v, v[0]):
            z = np.zeros(shape, dtype=np.float32); return z, 0.0
        q2, q98 = np.percentile(v, [2, 98]); z = np.clip((base-q2)/(q98-q2+1e-8), 0, 1).astype(np.float32); z[~mask2] = 0
        hist, _ = np.histogram(z[mask2], bins=32, range=(0, 1)); p = hist[hist > 0].astype(float); p /= max(p.sum(), 1)
        ent = float(-np.sum(p*np.log(p))/np.log(32))
        return z, ent

    def lowres_map(z):
        # Compact spatial image embedding; all nodes share this exact transform.
        return zoom(z, (19/ny, 25/nx), order=1).astype(np.float32)

    def lowres(z):
        return lowres_map(z).ravel()

    def pattern_similarity(a, b):
        aa = np.asarray(a).reshape(19, 25)[low_mask].astype(float); bb = np.asarray(b).reshape(19, 25)[low_mask].astype(float)
        if len(aa) < 3 or np.std(aa) == 0 or np.std(bb) == 0: return 0.0
        return float(np.corrcoef(aa, bb)[0, 1])

    # Gene image vectors and expression domains.
    gene_vec, gene_ent, gene_domains = {}, {}, {}
    obs_idx = rna.obs_names.get_indexer(obs)
    Xr = rna.X[obs_idx, :]
    if hasattr(Xr, 'to_memory'): Xr = Xr.to_memory()
    gene_idx = {str(n): i for i, n in enumerate(rna.var_names)}
    for start in range(0, len(genes), 256):
        names = genes[start:start+256]; cols = [gene_idx[g] for g in names]
        sub = Xr[:, cols]
        if hasattr(sub, 'toarray'): sub = sub.toarray()
        sub = np.asarray(sub, dtype=float)
        for j, g in enumerate(names):
            z, ent = image(sub[:, j]); zlow = lowres_map(z); gene_vec[g] = zlow.ravel(); gene_ent[g] = ent
            # Detect domains on the full-resolution 76x101 canvas before
            # downsampling, otherwise two nearby domains can merge.
            threshold = np.quantile(z[mask2], .8) if mask2.any() else 1
            active = mask2 & (z >= threshold); labs, n = label(active, structure=np.ones((3,3), int)); sizes = np.bincount(labs.ravel())[1:]
            keep = np.argsort(sizes)[::-1]; keep = [int(k+1) for k in keep if sizes[k] >= 25][:2]
            dom = np.full(zlow.shape, -1, dtype=np.int16)
            full_dom = np.full(shape, -1, dtype=np.int16)
            for d, k in enumerate(keep): full_dom[labs == k] = d
            # Majority assignment from full-resolution pixels to the lowres grid.
            for yy in range(19):
                for xx in range(25):
                    y0, y1 = int(yy*ny/19), max(int((yy+1)*ny/19), int(yy*ny/19)+1)
                    x0, x1 = int(xx*nx/25), max(int((xx+1)*nx/25), int(xx*nx/25)+1)
                    q = full_dom[y0:y1, x0:x1]; q = q[q >= 0]
                    if len(q): dom[yy, xx] = int(np.bincount(q).argmax())
            gene_domains[g] = dom
        if (start+256) % 2048 < 256: print('genes:', min(start+256, len(genes)), flush=True)
    rna.file.close()

    # Accessibility image vectors. The ATAC raw matrix is CSR; scan its rows
    # once and aggregate directly to the low-resolution canvas. This avoids
    # thousands of slow backed single-column reads.
    region_idx = {r: i for i, r in enumerate(regions)}
    low_n = 19 * 25
    vec_mm = np.memmap(out / 'enhancer_vectors.float32.dat', mode='w+', dtype='float32', shape=(len(regions), low_n))
    img_mm = np.memmap(out / 'enhancer_images.float32.dat', mode='w+', dtype='float32', shape=(len(regions), 19, 25))
    ent_arr = np.zeros(len(regions), dtype='float32')
    low_x = np.minimum((gx * 19 // nx).astype(int), 18)
    low_y = np.minimum((gy * 19 // ny).astype(int), 18)
    # Correct x dimension separately (19 rows x 25 columns).
    low_x = np.minimum((gx * 25 // nx).astype(int), 24)
    low_y = np.minimum((gy * 19 // ny).astype(int), 18)
    low_flat = low_y * 25 + low_x
    low_counts = np.bincount(low_flat, minlength=low_n).astype(float)
    sums = np.memmap(out / 'enhancer_lowres_sums.float32.dat', mode='w+', dtype='float32', shape=(len(regions), low_n))
    sums[:] = 0
    raw_var_names = list(atac.raw.var_names)
    raw_var_idx = {str(name): i for i, name in enumerate(raw_var_names)}
    col_to_region = np.full(len(raw_var_names), -1, dtype=np.int32)
    for i, region in enumerate(regions):
        col_to_region[raw_var_idx[region]] = i
    selected_rows = np.zeros(rna.n_obs, dtype=bool)
    selected_rows[obs_idx] = True
    raw_path = root / 'data' / 'GW12_spatial_ATAC.h5ad'
    with h5py.File(raw_path, 'r') as f:
        data_ds = f['raw/X/data']; indices_ds = f['raw/X/indices']; indptr = np.asarray(f['raw/X/indptr'])
        for row_start in range(0, rna.n_obs, 512):
            row_end = min(row_start + 512, rna.n_obs)
            if not selected_rows[row_start:row_end].any():
                continue
            a, b = int(indptr[row_start]), int(indptr[row_end])
            data = np.asarray(data_ds[a:b], dtype=np.float32)
            indices = np.asarray(indices_ds[a:b], dtype=np.int64)
            for row in range(row_start, row_end):
                if not selected_rows[row]:
                    continue
                lo = int(indptr[row] - a); hi = int(indptr[row+1] - a)
                rr = col_to_region[indices[lo:hi]]
                keep = rr >= 0
                if keep.any():
                    # Map the original observation row to its low-resolution cell.
                    pos = np.flatnonzero(obs_idx == row)
                    if len(pos):
                        np.add.at(sums, (rr[keep], np.full(keep.sum(), low_flat[pos[0]], dtype=np.int64)), data[lo:hi][keep])
            if row_end % 2048 == 0: print('ATAC rows:', row_end, flush=True)
    for i in range(len(regions)):
        zlow = (np.asarray(sums[i], dtype=float) / np.maximum(low_counts, 1)).reshape(19, 25)
        vals = zlow[low_mask]
        if len(vals) and not np.allclose(vals, vals[0]):
            q2, q98 = np.percentile(vals, [2, 98]); zlow = np.clip((zlow-q2)/(q98-q2+1e-8), 0, 1).astype(np.float32)
        zlow[~low_mask] = 0
        hist, _ = np.histogram(zlow[low_mask], bins=32, range=(0, 1)); p = hist[hist > 0].astype(float); p /= max(p.sum(), 1)
        ent_arr[i] = float(-np.sum(p*np.log(p))/np.log(32)) if len(p) else 0
        img_mm[i] = zlow; vec_mm[i] = zlow.ravel()
    del sums
    atac.file.close()

    rows = []
    for g, group in links.groupby('target', sort=False):
        es = group.region.tolist(); gv = gene_vec[g]; ei = [region_idx[e] for e in es]
        sims = np.array([(pattern_similarity(gv, vec_mm[i]) + 1) / 2 for i in ei])
        pair = []
        for i in range(len(es)):
            for j in range(i+1, len(es)):
                pair.append(1-(pattern_similarity(vec_mm[ei[i]], vec_mm[ei[j]]) + 1) / 2)
        dom = gene_domains[g]; specific = 0; shared = 0; domain_n = int(dom.max()+1)
        local_scores = []
        for e in es:
            if domain_n < 2: continue
            em = img_mm[region_idx[e]]
            means = [float(em[dom == d].mean()) if np.any(dom == d) else 0.0 for d in range(domain_n)]
            total = float(em[low_mask].mean()) + 1e-8
            enrich = np.clip(np.array(means)/total - 1, -1, 1)
            local_scores.append(float(enrich.max()-enrich.min()))
            pos = enrich > .05
            if pos.sum() == 1: specific += 1
            elif pos.sum() >= 2: shared += 1
        sims01 = np.clip((sims+1)/2, 0, 1)
        entropy_spread = float(np.std([ent_arr[i] for i in ei]))
        # Score is size-normalized and does not reward simply having more links.
        breadth = float(1-sims01.mean())
        pair_div = float(np.mean(pair)) if pair else 0.0
        branch_n = int(sum(s < np.quantile(sims, .5) for s in sims))
        rows.append({'gene': g, 'n_enhancers': len(es), 'mean_gene_enhancer_similarity': float(sims01.mean()),
                     'enhancer_breadth': breadth, 'enhancer_pairwise_diversity': pair_div,
                     'enhancer_entropy_sd': entropy_spread, 'n_expression_domains': domain_n,
                     'domain_specific_fraction': float(specific/len(es)), 'shared_fraction': float(shared/len(es)),
                     'mean_domain_switching': float(np.mean(local_scores)/2) if local_scores else 0.0,
                     'network_branch_count': branch_n})
    result = pd.DataFrame(rows)
    # Primary score: enhancer diversity + spatial switching, with weak weighting
    # of gene--enhancer dissimilarity. The score is not a gene-expression score.
    result['regulatory_complexity_score'] = (
        .35*result.enhancer_pairwise_diversity +
        .25*result.enhancer_breadth +
        .25*result.domain_specific_fraction +
        .15*result.mean_domain_switching.clip(lower=0)
    )
    result = result.sort_values('regulatory_complexity_score', ascending=False)
    result.to_csv(out / 'gene_regulatory_complexity.tsv', sep='\t', index=False)
    meta = {'n_genes': len(result), 'n_regions': len(regions), 'n_paired_spots': len(obs), 'canvas_shape': [ny, nx], 'score_definition': '0.35 pairwise diversity + 0.25 enhancer breadth + 0.25 domain-specific fraction + 0.15 domain switching', 'embedding': 'mask-aware low-resolution spatial raster vector; UMAP reserved for visualization'}
    (out/'metadata.json').write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2)); print(result.head(20).to_string(index=False))


if __name__ == '__main__': main()
