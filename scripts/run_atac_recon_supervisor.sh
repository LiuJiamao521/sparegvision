#!/usr/bin/env bash
set -u

ROOT=/cluster2/huanglab/jiamao/Project/SpaRegVision
SCRIPT="$ROOT/scripts/reconstruct_genomewide_atac_6s_standalone.py"
WORKDIR="$ROOT"
STATE="$ROOT/data/weMERFISH/genomewide_atac_6s_standalone/reconstruction_state.json"
LOG="$ROOT/data/weMERFISH/genomewide_atac_6s_standalone/supervisor.log"

WORKERS="${1:-20}"
BATCH_SIZE="${2:-256}"
SLEEP_SECS="${3:-5}"
TIMEOUT_SECS="${4:-1800}"

mkdir -p "$(dirname "$LOG")"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"
}

read_state_field() {
  python - <<'PY' "$STATE" "$1"
import json, sys
from pathlib import Path
p=Path(sys.argv[1])
key=sys.argv[2]
if not p.exists():
    print('')
else:
    data=json.loads(p.read_text())
    print(data.get(key, ''))
PY
}

log "supervisor start workers=$WORKERS batch_size=$BATCH_SIZE sleep=$SLEEP_SECS timeout=$TIMEOUT_SECS"

while true; do
  completed="$(read_state_field completed)"
  if [[ "$completed" == "True" || "$completed" == "true" ]]; then
    log "reconstruction completed; supervisor exiting"
    exit 0
  fi

  before_idx="$(read_state_field next_interval_index)"
  before_readable="$(read_state_field readable_intervals)"
  before_blocked="$(read_state_field blocked_intervals)"
  log "launch batch from next_interval_index=${before_idx:-NA} readable=${before_readable:-NA} blocked=${before_blocked:-NA}"

  if timeout "$TIMEOUT_SECS" env HDF5_USE_FILE_LOCKING=FALSE python "$SCRIPT" --batch-size "$BATCH_SIZE" --max-batches 1 --workers "$WORKERS" >> "$LOG" 2>&1; then
    after_idx="$(read_state_field next_interval_index)"
    after_readable="$(read_state_field readable_intervals)"
    after_blocked="$(read_state_field blocked_intervals)"
    log "batch finished next_interval_index=${after_idx:-NA} readable=${after_readable:-NA} blocked=${after_blocked:-NA}"
  else
    code=$?
    after_idx="$(read_state_field next_interval_index)"
    log "batch exited nonzero code=$code; current next_interval_index=${after_idx:-NA}; retrying after sleep"
  fi

  sleep "$SLEEP_SECS"
done
