from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import h5py
import numpy as np
import requests

REQUESTED_INDEX_URL = "https://single-cell-data-yinan.s3.us-west-2.amazonaws.com/6s/atac/genes.csv"
ATAC_BASE_URL = "https://single-cell-data-yinan.s3.us-west-2.amazonaws.com/6s/atac"
ROOT = Path("/cluster2/huanglab/jiamao/Project/SpaRegVision")
DATA_DIR = ROOT / "data" / "weMERFISH"
TEMPLATE_H5AD = DATA_DIR / "weMERFISH_combined_C_6s_E1_rescaled_z.h5ad"
OUT_DIR = DATA_DIR / "genomewide_atac_6s"
OUT_H5AD = OUT_DIR / "weMERFISH_combined_C_6s_E1_imputed_atac_rescaled_z.h5ad"
STATE_JSON = OUT_DIR / "reconstruction_state.json"
READABLE_TXT = OUT_DIR / "readable_intervals.txt"
BLOCKED_TXT = OUT_DIR / "blocked_intervals.txt"
ERROR_TXT = OUT_DIR / "error_intervals.txt"
TMP_OUT = OUT_DIR / (OUT_H5AD.name + ".tmp")


os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")


def disable_ssl_warnings() -> None:
    requests.packages.urllib3.disable_warnings()


def fetch_text(session: requests.Session, url: str, timeout: int = 120, retries: int = 5) -> str:
    last_exc = None
    for attempt in range(retries):
        try:
            response = session.get(url, verify=False, timeout=timeout)
            response.raise_for_status()
            return response.text
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            time.sleep(min(2**attempt, 20))
    raise last_exc  # type: ignore[misc]


def fetch_interval_vector(session: requests.Session, interval: str, n_obs: int, timeout: int = 60, retries: int = 5):
    url = f"{ATAC_BASE_URL}/{interval}.csv"
    last_exc = None
    for attempt in range(retries):
        try:
            response = session.get(url, verify=False, timeout=timeout)
            if response.status_code == 403:
                return "blocked", None
            response.raise_for_status()
            arr = [x for x in response.text.split(",") if x]
            if not arr or arr[0] != interval:
                raise ValueError(f"unexpected header for {interval}: {arr[:3]}")
            vec = np.asarray(arr[1:], dtype=np.float32)
            if vec.shape[0] != n_obs:
                raise ValueError(f"{interval} vector length {vec.shape[0]} != {n_obs}")
            return "ok", vec
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            time.sleep(min(2**attempt, 20))
    return "error", repr(last_exc)


def get_interval_index(session: requests.Session) -> list[str]:
    text = fetch_text(session, REQUESTED_INDEX_URL)
    return [x for x in text.split(",") if x and x != "genes"]


def write_string_array(group: h5py.Group, key: str, values: list[str]) -> None:
    if key in group:
        del group[key]
    dt = h5py.string_dtype(encoding="utf-8")
    ds = group.create_dataset(key, data=np.asarray(values, dtype=object), dtype=dt)
    ds.attrs["encoding-type"] = "string-array"
    ds.attrs["encoding-version"] = "0.2.0"


def write_scalar_string(group: h5py.Group, key: str, value: str) -> None:
    if key in group:
        del group[key]
    dt = h5py.string_dtype(encoding="utf-8")
    ds = group.create_dataset(key, data=value, dtype=dt)
    ds.attrs["encoding-type"] = "string"
    ds.attrs["encoding-version"] = "0.2.0"


def read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_lines(path: Path, values: list[str]) -> None:
    if not values:
        return
    with path.open("a", encoding="utf-8") as handle:
        for value in values:
            handle.write(value + "\n")


def initialize_output() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if OUT_H5AD.exists() or TMP_OUT.exists():
        return
    shutil.copy2(TEMPLATE_H5AD, TMP_OUT)
    with h5py.File(TMP_OUT, "a") as f:
        if "X_imputed" in f["obsm"]:
            del f["obsm"]["X_imputed"]
        f["obsm"].create_dataset(
            "X_imputed",
            shape=(f["obs"].attrs["column-order"].shape[0] * 0 + f["X"].shape[0], 0),
            maxshape=(f["X"].shape[0], None),
            dtype=np.float32,
            chunks=(min(f["X"].shape[0], 1024), 16),
            compression="gzip",
            compression_opts=4,
        )
        f["obsm"]["X_imputed"].attrs["encoding-type"] = "array"
        f["obsm"]["X_imputed"].attrs["encoding-version"] = "0.2.0"
        uns = f["uns"]
        if "imputed_gene_names" in uns:
            del uns["imputed_gene_names"]
        if "imputed_atac_region_names" in uns:
            del uns["imputed_atac_region_names"]
        write_scalar_string(uns, "imputed_modality", "ATAC_500bp_bins")
    STATE_JSON.write_text(
        json.dumps(
            {
                "template_h5ad": str(TEMPLATE_H5AD),
                "output_h5ad": str(OUT_H5AD),
                "next_interval_index": 0,
                "requested_intervals": None,
                "readable_intervals": 0,
                "blocked_intervals": 0,
                "error_intervals": 0,
                "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "completed": False,
            }, indent=2
        )
    )
    for path in [READABLE_TXT, BLOCKED_TXT, ERROR_TXT]:
        if not path.exists():
            path.write_text("", encoding="utf-8")


def load_state() -> dict:
    return json.loads(STATE_JSON.read_text(encoding="utf-8"))


def save_state(state: dict) -> None:
    STATE_JSON.write_text(json.dumps(state, indent=2), encoding="utf-8")


def finalize_output(state: dict) -> None:
    readable = read_lines(READABLE_TXT)
    blocked = read_lines(BLOCKED_TXT)
    errors = read_lines(ERROR_TXT)
    with h5py.File(TMP_OUT, "a") as f:
        uns = f["uns"]
        write_string_array(uns, "imputed_gene_names", readable)
        write_string_array(uns, "imputed_atac_region_names", readable)
        write_string_array(uns, "blocked_atac_interval_names", blocked)
        write_string_array(uns, "error_atac_interval_names", errors)
        write_scalar_string(uns, "imputed_modality", "ATAC_500bp_bins")
        write_scalar_string(uns, "imputed_source", "MERFISHEYES public S3 per-interval vectors")
        write_scalar_string(uns, "imputed_cell_order", "Matches template obs order")
    TMP_OUT.replace(OUT_H5AD)
    state["completed"] = True
    state["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_state(state)


def run(batch_size: int, max_batches: int | None) -> None:
    disable_ssl_warnings()
    initialize_output()
    state = load_state()
    session = requests.Session()
    intervals = get_interval_index(session)
    state["requested_intervals"] = len(intervals)
    save_state(state)

    with h5py.File(TMP_OUT, "a") as f:
        dset = f["obsm"]["X_imputed"]
        n_obs = dset.shape[0]
        batch_counter = 0
        while state["next_interval_index"] < len(intervals):
            start_idx = state["next_interval_index"]
            end_idx = min(start_idx + batch_size, len(intervals))
            chunk = intervals[start_idx:end_idx]
            ok_names: list[str] = []
            ok_vectors: list[np.ndarray] = []
            blocked: list[str] = []
            errors: list[str] = []

            for interval in chunk:
                status, payload = fetch_interval_vector(session, interval, n_obs)
                if status == "ok":
                    ok_names.append(interval)
                    ok_vectors.append(payload)  # type: ignore[arg-type]
                elif status == "blocked":
                    blocked.append(interval)
                else:
                    errors.append(f"{interval}\t{payload}")

            if ok_vectors:
                block = np.column_stack(ok_vectors)
                old_cols = dset.shape[1]
                dset.resize((n_obs, old_cols + block.shape[1]))
                dset[:, old_cols:old_cols + block.shape[1]] = block
                append_lines(READABLE_TXT, ok_names)
            append_lines(BLOCKED_TXT, blocked)
            append_lines(ERROR_TXT, errors)

            state["next_interval_index"] = end_idx
            state["readable_intervals"] = dset.shape[1]
            state["blocked_intervals"] = len(read_lines(BLOCKED_TXT))
            state["error_intervals"] = len(read_lines(ERROR_TXT))
            state["last_batch_end"] = end_idx
            state["last_updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            save_state(state)

            batch_counter += 1
            print(
                f"batch\t{batch_counter}\tintervals\t{start_idx}:{end_idx}\t"
                f"ok\t{len(ok_names)}\tblocked\t{len(blocked)}\terrors\t{len(errors)}\t"
                f"written_cols\t{dset.shape[1]}"
            )
            if max_batches is not None and batch_counter >= max_batches:
                return

    finalize_output(state)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-batches", type=int, default=None)
    args = parser.parse_args()
    run(batch_size=args.batch_size, max_batches=args.max_batches)


if __name__ == "__main__":
    main()
