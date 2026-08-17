from __future__ import annotations

import argparse
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REQUESTED_INDEX_URL = 'https://single-cell-data-yinan.s3.us-west-2.amazonaws.com/6s/atac/genes.csv'
ATAC_BASE_URL = 'https://single-cell-data-yinan.s3.us-west-2.amazonaws.com/6s/atac'
ROOT = Path('/cluster2/huanglab/jiamao/Project/SpaRegVision')
RAW_ROOT = ROOT / 'data' / 'weMERFISH' / 'atac_raw_6s'
CSV_DIR = RAW_ROOT / 'csv'
LOG_DIR = RAW_ROOT / 'logs'
STATE_JSON = LOG_DIR / 'download_state.json'
INDEX_TXT = LOG_DIR / 'all_intervals.txt'
READABLE_TXT = LOG_DIR / 'readable_intervals.txt'
BLOCKED_TXT = LOG_DIR / 'blocked_intervals.txt'
ERROR_TXT = LOG_DIR / 'error_intervals.txt'
DOWNLOAD_LOG = LOG_DIR / 'download.log'


def append_line(path: Path, text: str) -> None:
    with path.open('a', encoding='utf-8') as handle:
        handle.write(text + '\n')


def save_state(state: dict) -> None:
    STATE_JSON.write_text(json.dumps(state, indent=2), encoding='utf-8')


def load_state() -> dict:
    if not STATE_JSON.exists():
        return {}
    return json.loads(STATE_JSON.read_text(encoding='utf-8'))


def log(message: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    append_line(DOWNLOAD_LOG, line)


def ensure_layout() -> None:
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    for path in [READABLE_TXT, BLOCKED_TXT, ERROR_TXT, DOWNLOAD_LOG]:
        if not path.exists():
            path.write_text('', encoding='utf-8')


def existing_done() -> tuple[set[str], set[str], set[str]]:
    def read(path: Path) -> set[str]:
        if not path.exists():
            return set()
        return {line.strip().split('\t')[0] for line in path.read_text(encoding='utf-8').splitlines() if line.strip()}
    return read(READABLE_TXT), read(BLOCKED_TXT), read(ERROR_TXT)


def run_curl(args: list[str], timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout, check=False)


def fetch_index(timeout: int, retries: int) -> list[str]:
    last_err = None
    for attempt in range(retries):
        cp = run_curl(['curl', '-k', '-fL', '--connect-timeout', '20', '--max-time', str(timeout), REQUESTED_INDEX_URL], timeout=timeout + 10)
        if cp.returncode == 0:
            values = [x for x in cp.stdout.split(',') if x and x != 'genes']
            INDEX_TXT.write_text('\n'.join(values) + '\n', encoding='utf-8')
            return values
        last_err = cp.stderr.strip() or f'curl_exit_{cp.returncode}'
        time.sleep(min(2 ** attempt, 20))
    raise RuntimeError(f'failed to fetch index: {last_err}')


def single_attempt(interval: str, timeout: int) -> tuple[str, str | None]:
    target = CSV_DIR / f'{interval}.csv'
    url = f'{ATAC_BASE_URL}/{interval}.csv'
    cp = run_curl([
        'curl', '-k', '-fL', '--connect-timeout', '20', '--max-time', str(timeout),
        '--retry', '0', url, '-o', str(target)
    ], timeout=timeout + 10)
    if cp.returncode == 0 and target.exists() and target.stat().st_size > 0:
        return 'ok', None
    if target.exists() and target.stat().st_size == 0:
        target.unlink()
    stderr = (cp.stderr or '').strip()
    stdout = (cp.stdout or '').strip()
    msg = stderr or stdout or f'curl_exit_{cp.returncode}'
    if '403' in msg or 'The requested URL returned error: 403' in msg:
        return 'blocked', None
    return 'error', msg


def download_one(interval: str, timeout: int, retries: int, extra_error_retries: int) -> tuple[str, str, str | None]:
    target = CSV_DIR / f'{interval}.csv'
    if target.exists() and target.stat().st_size > 0:
        return interval, 'ok', 'cached'

    last_err = None
    total_attempts = retries + extra_error_retries
    for attempt in range(total_attempts):
        status, detail = single_attempt(interval, timeout)
        if status == 'ok':
            return interval, 'ok', None
        if status == 'blocked':
            return interval, 'blocked', None
        last_err = detail
        time.sleep(min(2 ** attempt, 30))
    return interval, 'error', last_err


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=50)
    parser.add_argument('--timeout', type=int, default=60)
    parser.add_argument('--retries', type=int, default=5)
    parser.add_argument('--extra-error-retries', type=int, default=8)
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()

    ensure_layout()
    state = load_state()
    log('fetching interval index')
    intervals = fetch_index(timeout=max(args.timeout, 120), retries=args.retries)
    readable_done, blocked_done, error_done = existing_done()
    done = readable_done | blocked_done
    pending = [x for x in intervals if x not in done]
    if args.limit is not None:
        pending = pending[: args.limit]

    state.update({
        'requested_intervals': len(intervals),
        'pending_intervals': len(pending),
        'workers': args.workers,
        'retries': args.retries,
        'extra_error_retries': args.extra_error_retries,
        'last_started_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'completed': False,
    })
    save_state(state)
    log(f'start workers={args.workers} pending={len(pending)} total={len(intervals)} retries={args.retries}+{args.extra_error_retries}')

    readable = len(readable_done)
    blocked = len(blocked_done)
    errors = 0
    processed = 0

    # Rebuild error log for this run only; transient failures should not permanently exclude intervals.
    ERROR_TXT.write_text('', encoding='utf-8')

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_one, interval, args.timeout, args.retries, args.extra_error_retries): interval
            for interval in pending
        }
        for future in as_completed(futures):
            interval, status, detail = future.result()
            processed += 1
            if status == 'ok':
                if detail != 'cached':
                    append_line(READABLE_TXT, interval)
                readable += 1
            elif status == 'blocked':
                append_line(BLOCKED_TXT, interval)
                blocked += 1
            else:
                append_line(ERROR_TXT, f'{interval}\t{detail}')
                errors += 1

            if processed % 50 == 0 or processed == len(pending):
                state.update({
                    'processed_this_run': processed,
                    'readable_intervals': readable,
                    'blocked_intervals': blocked,
                    'error_intervals': errors,
                    'last_updated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                })
                save_state(state)
                log(f'processed={processed}/{len(pending)} readable={readable} blocked={blocked} errors={errors}')

    state.update({
        'processed_this_run': processed,
        'readable_intervals': readable,
        'blocked_intervals': blocked,
        'error_intervals': errors,
        'last_updated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'completed': True,
    })
    save_state(state)
    log(f'finished readable={readable} blocked={blocked} errors={errors}')


if __name__ == '__main__':
    main()
