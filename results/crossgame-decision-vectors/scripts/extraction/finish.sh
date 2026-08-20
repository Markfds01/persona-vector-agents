#!/bin/bash
# Land the already-generated trust rows, then finish ultimatum and the PD census.
set -u
R=/home/marco/dockmaster/state/worktrees/crossgame-vector
OUT=/home/marco/dockmaster/data/crossgame-vector
PY=$R/.venv/bin/python
echo "=== trust: extract ($(date -Is)) ==="
"$PY" "$R/scratch/extract_crossgame.py" --rows "$R/scratch/rows/trust.csv" \
    --out-dir "$OUT/acts/trust" --device 0 || { echo "FAILED extract trust"; exit 1; }
echo "=== trust: land ($(date -Is)) ==="
cp "$R/scratch/rows/trust.csv" "$OUT/acts/trust/rows.csv"
"$PY" "$R/scratch/land_family.py" --family trust --rows "$OUT/acts/trust/rows.csv" \
    --acts "$OUT/acts/trust" --out "$OUT" || { echo "FAILED land trust"; exit 1; }
echo "=== trust: landed ($(date -Is)) ==="
exec bash "$R/scratch/resume_run.sh" ultimatum:24 prisoners_dilemma:8
