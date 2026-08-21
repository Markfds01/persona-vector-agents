#!/usr/bin/env bash
# Re-extract every activation in this directory's corpus with `audit.extract`.
#
# One extractor for the whole corpus: the six cross-game grids and the earlier
# Dictator-only grid, all from their COMMITTED rows CSVs, so the comparison
# against the archived activations is exact rather than distributional. Nothing
# is re-generated here — the rows are the ones already committed.
#
# One game per invocation, resumable: the device is shared with a tenant that
# cycles models without warning, so an eviction costs the shard in flight and
# `--resume` picks the game back up from the shards already on disk. Nothing
# about dtype, attention implementation, batch semantics or revision is ever
# relaxed to make a run fit — that would silently break comparability with every
# vector already built.
#
#   ACTS=<somewhere with ~8 GB> PY=<python with torch+transformers> \
#       bash results/crossgame-decision-vectors/scripts/run_extraction.sh
#
# DEVICE pins one card; RESUME=1 continues an interrupted run.
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"
DIR="$(dirname "$HERE")"
REPO="$(cd "$DIR/../.." && pwd)"

ACTS="${ACTS:?set ACTS to the directory the activation shards go in (~8 GB)}"
PY="${PY:-python3}"
DEVICE="${DEVICE:-0}"
RESUME_FLAG=""
[ "${RESUME:-0}" = "1" ] && RESUME_FLAG="--resume"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

XGRID="$HERE/extraction/crossgame_grid.py"
DGRID="$REPO/results/dictator-decision-vector/scripts/decision_grid.py"

cd "$REPO"
mkdir -p "$ACTS"
for fam in dictator trust ultimatum apology overfishing prisoners_dilemma; do
  echo "=== $fam $(date -u +%FT%TZ) ==="
  "$PY" -u -m audit.extract --rows "$DIR/extraction/$fam.csv" \
      --out-dir "$ACTS/$fam" --grid "$XGRID" --device "$DEVICE" $RESUME_FLAG
  cp "$DIR/extraction/$fam.csv" "$ACTS/$fam/rows.csv"
done

# The Dictator-only grid: its vector is what section 6 projects out, so it has to
# come off the same extractor as everything it is compared with.
echo "=== dictator_only $(date -u +%FT%TZ) ==="
"$PY" -u -m audit.extract \
    --rows "$REPO/results/dictator-decision-vector/extraction/grid_seed0.csv" \
    --out-dir "$ACTS/dictator_only" --grid "$DGRID" --device "$DEVICE" $RESUME_FLAG
cp "$REPO/results/dictator-decision-vector/extraction/grid_seed0.csv" \
   "$ACTS/dictator_only/rows.csv"
echo "=== EXTRACTION COMPLETE $(date -u +%FT%TZ) ==="
