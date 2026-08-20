#!/bin/bash
# Generate, extract and land the cross-game grid one game at a time.
#
# Per game: generate its 30 cells, capture activations over prompt+response, then
# write that game's vector and its manifest entry. Landing per game is the whole
# point - the GPU is shared with a tenant that cycles models without warning, so
# an eviction has to cost the game in flight and nothing else.
#
# A failure stops the run. It is never retried in a loop: a game that failed is a
# result to report, and the games already landed stay landed.
set -u
ROOT=/home/marco/dockmaster/state/worktrees/crossgame-vector
OUT=${OUT:-/home/marco/dockmaster/data/crossgame-vector}
PY=$ROOT/.venv/bin/python
SAMPLES=${SAMPLES:?}
BATCH=${BATCH:?}
SEED=${SEED:?}
DEVICE=${DEVICE:?}
FAMILIES=${FAMILIES:?}

mkdir -p "$ROOT/scratch/rows" "$OUT/acts"
for fam in $FAMILIES; do
  echo "=== $fam: generate ($(date -Is)) ==="
  "$PY" "$ROOT/scratch/gen_crossgame.py" --out "$ROOT/scratch/rows/$fam.csv" \
      --samples "$SAMPLES" --batch-size "$BATCH" --seed "$SEED" \
      --families "$fam" --device "$DEVICE" || { echo "FAILED generate $fam"; exit 1; }

  echo "=== $fam: extract ($(date -Is)) ==="
  "$PY" "$ROOT/scratch/extract_crossgame.py" --rows "$ROOT/scratch/rows/$fam.csv" \
      --out-dir "$OUT/acts/$fam" --device "$DEVICE" \
      || { echo "FAILED extract $fam"; exit 1; }

  echo "=== $fam: land ($(date -Is)) ==="
  cp "$ROOT/scratch/rows/$fam.csv" "$OUT/acts/$fam/rows.csv"
  "$PY" "$ROOT/scratch/land_family.py" --family "$fam" \
      --rows "$OUT/acts/$fam/rows.csv" --acts "$OUT/acts/$fam" --out "$OUT" \
      || { echo "FAILED land $fam"; exit 1; }
  echo "=== $fam: landed ($(date -Is)) ==="
done
echo "ALL DONE $(date -Is)"
