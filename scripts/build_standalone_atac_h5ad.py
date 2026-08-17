from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')

ROOT = Path('/cluster2/huanglab/jiamao/Project/SpaRegVision')
DATA_DIR = ROOT / 'data' / 'weMERFISH'
TEMPLATE_H5AD = DATA_DIR / 'weMERFISH_combined_C_6s_E1_rescaled_z.h5ad'
RAW_ROOT = DATA_DIR / 'atac_raw_6s'
CSV_DIR = RAW_ROOT / 'csv'
LOG_DIR = RAW_ROOT / 'logs'
READABLE_TXT = LOG_DIR / 'readable_intervals.txt'
OUT_H5AD = DATA_DIR / 'weMERFISH_spatial_ATAC_C_6s_E1.h5ad'
STATE_JSON = DATA_DIR / 'weMERFISH_spatial_ATAC_C_6s_E1.build_state.json'
TMP_H5AD = DATA_DIR / (OUT_H5AD.name + '.tmp')


def write_string_array(group: h5py.Group, key: str, values: list[str]) -> None:
    if key in group:
        del group[key]
    dt = h5py.string_dtype(encoding='utf-8')
    ds = group.create_dataset(key, data=np.asarray(values, dtype=object), dtype=dt)
    ds.attrs['encoding-type'] = 'string-array'
    ds.attrs['encoding-version'] = '0.2.0'


def write_array(group: h5py.Group, key: str, values: np.ndarray) -> None:
    if key in group:
        del group[key]
    ds = group.create_dataset(key, data=values)
    ds.attrs['encoding-type'] = 'array'
    ds.attrs['encoding-version'] = '0.2.0'


def write_scalar_string(group: h5py.Group, key: str, value: str) -> None:
    if key in group:
        del group[key]
    dt = h5py.string_dtype(encoding='utf-8')
    ds = group.create_dataset(key, data=value, dtype=dt)
    ds.attrs['encoding-type'] = 'string'
    ds.attrs['encoding-version'] = '0.2.0'


def parse_intervals(intervals: list[str]) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    chrom = []
    start = []
    end = []
    width = []
    for interval in intervals:
        c, s, e = interval.split('-')
        chrom.append(f'chr{c}')
        s_i = int(s)
        e_i = int(e)
        start.append(s_i)
        end.append(e_i)
        width.append(e_i - s_i)
    return chrom, np.asarray(start, dtype=np.int64), np.asarray(end, dtype=np.int64), np.asarray(width, dtype=np.int64)


def rebuild_var_group(f: h5py.File, intervals: list[str]) -> None:
    if 'var' in f:
        del f['var']
    var = f.create_group('var')
    var.attrs['_index'] = '_index'
    var.attrs['column-order'] = np.asarray(['chrom', 'start', 'end', 'width'], dtype=object)
    var.attrs['encoding-type'] = 'dataframe'
    var.attrs['encoding-version'] = '0.2.0'
    write_string_array(var, '_index', intervals)
    chrom, start, end, width = parse_intervals(intervals)
    write_string_array(var, 'chrom', chrom)
    write_array(var, 'start', start)
    write_array(var, 'end', end)
    write_array(var, 'width', width)


def load_intervals(limit: int | None) -> list[str]:
    vals = [x.strip() for x in READABLE_TXT.read_text(encoding='utf-8').splitlines() if x.strip()]
    if limit is not None:
        vals = vals[:limit]
    return vals


def init_output(n_obs: int, n_vars: int) -> None:
    import shutil
    if TMP_H5AD.exists():
        TMP_H5AD.unlink()
    shutil.copy2(TEMPLATE_H5AD, TMP_H5AD)
    with h5py.File(TMP_H5AD, 'a') as f:
        for key in ['X', 'layers', 'obsp', 'varm', 'varp']:
            if key in f:
                del f[key]
        X = f.create_dataset(
            'X',
            shape=(n_obs, n_vars),
            dtype=np.float32,
            chunks=(1024, 64),
            compression='gzip',
            compression_opts=4,
        )
        X.attrs['encoding-type'] = 'array'
        X.attrs['encoding-version'] = '0.2.0'
        # Initialize  with a syntactically valid placeholder interval.
        # The full interval table is written after matrix construction completes.
        rebuild_var_group(f, ['1-0-500'] * n_vars)
        if 'obsm' in f:
            for key in list(f['obsm'].keys()):
                if key not in {'spatial', 'spatial_rescaled_z'}:
                    del f['obsm'][key]
        if 'uns' in f:
            for key in list(f['uns'].keys()):
                del f['uns'][key]
        write_scalar_string(f['uns'], 'modality', 'ATAC_500bp_bins')
        write_scalar_string(f['uns'], 'source', 'Local CSV reconstruction from MERFISHEYES 6s ATAC exports')
        write_scalar_string(f['uns'], 'cell_order', 'Matches transcriptome obs order')


def save_state(state: dict) -> None:
    STATE_JSON.write_text(json.dumps(state, indent=2), encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--chunk-size', type=int, default=256)
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()

    intervals = load_intervals(args.limit)
    if not intervals:
        raise SystemExit('No readable intervals found. Run downloader first.')

    with h5py.File(TEMPLATE_H5AD, 'r') as template:
        n_obs = template['obs']['cid'].shape[0]

    init_output(n_obs, len(intervals))
    state = {
        'n_obs': n_obs,
        'n_vars': len(intervals),
        'chunk_size': args.chunk_size,
        'next_col': 0,
        'started_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'completed': False,
    }
    save_state(state)

    with h5py.File(TMP_H5AD, 'a') as f:
        X = f['X']
        for start in range(0, len(intervals), args.chunk_size):
            end = min(start + args.chunk_size, len(intervals))
            chunk = intervals[start:end]
            block = np.empty((n_obs, len(chunk)), dtype=np.float32)
            for idx, interval in enumerate(chunk):
                arr = [x for x in (CSV_DIR / f'{interval}.csv').read_text(encoding='utf-8').split(',') if x]
                if not arr or arr[0] != interval:
                    raise ValueError(f'Unexpected CSV header for {interval}')
                vec = np.asarray(arr[1:], dtype=np.float32)
                if vec.shape[0] != n_obs:
                    raise ValueError(f'{interval} length {vec.shape[0]} != {n_obs}')
                block[:, idx] = vec
            X[:, start:end] = block
            state['next_col'] = end
            state['last_updated_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
            save_state(state)
            print(f'written_cols\t{start}:{end}', flush=True)

        rebuild_var_group(f, intervals)
        write_string_array(f['uns'], 'all_atac_interval_names', intervals)

    TMP_H5AD.replace(OUT_H5AD)
    state['completed'] = True
    state['completed_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
    save_state(state)
    print(f'saved\t{OUT_H5AD}')


if __name__ == '__main__':
    main()
