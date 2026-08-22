#!/usr/bin/env bash
# The whole package, end to end: null vectors, the sweep, the analysis, the tables.
#
# Nothing here is situational. Every path that depends on the machine — the
# interpreter, the activation root, where the rows land — arrives from the
# environment, and everything else is derived from this file's own location, so
# the pipeline runs out of a checkout with no absolute path in it.
#
#   ACTS=<activation root from the vector study's run_extraction.sh> \
#   PY=<python with torch+transformers> SAMPLES=<n per point> \
#       bash results/crossgame-per-game-steering/scripts/run_steering.sh
#
# DEVICE pins one card (default 0). GAMES restricts the run. RESUME=1 continues an
# interrupted sweep, skipping every beta point whose CSV is already whole; that is
# the flag to use after an OOM, and it is safe because each point is generated
# independently of the others.
#
# Stage 1 is CPU-only and needs ACTS. Stages 2-4 do not touch it: once the null
# vectors are committed the sweep and the analysis run from the checkout alone.
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"
DIR="$(dirname "$HERE")"
REPO="$(cd "$DIR/../.." && pwd)"

PY="${PY:-python3}"
DEVICE="${DEVICE:-0}"
SAMPLES="${SAMPLES:?set SAMPLES to the number of generations per beta point}"
BATCH="${BATCH:-20}"
SEED="${SEED:-0}"
NULL_SEED="${NULL_SEED:-20260821}"
POLICY="${POLICY:-strict}"
GAMES="${GAMES:-dictator,trust,ultimatum,apology,overfishing,prisoners_dilemma}"
LOGS="${LOGS:-$DIR/provenance}"
RESUME_FLAG=""
[ "${RESUME:-0}" = "1" ] && RESUME_FLAG="--resume"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

cd "$REPO"
mkdir -p "$DIR/vectors" "$DIR/rows" "$DIR/analysis" "$LOGS"

# 1. the matched nulls. CPU only, and it refuses to write one unless it can first
#    rebuild the committed vector it is a null of.
if [ "${SKIP_NULLS:-0}" != "1" ]; then
  echo "=== null vectors $(date -u +%FT%TZ) ==="
  ACTS="${ACTS:?set ACTS to the activation root, or SKIP_NULLS=1 to use the committed nulls}"
  CUDA_VISIBLE_DEVICES="" "$PY" -u "$HERE/measurement/build_nulls.py" \
      --acts "$ACTS" --out-dir "$DIR/vectors" --policy "$POLICY" \
      --seed "$NULL_SEED" --report "$LOGS/null_vectors.json" \
      2>&1 | tee "$LOGS/null_vectors.log"
fi

# 2. the sweep itself. One process holds the weights for every game — the card is
#    shared and each load is a fresh wait for headroom — but each game lands its
#    own manifest entry the moment it finishes, and each beta point its own CSV.
echo "=== sweep $(date -u +%FT%TZ) ==="
"$PY" -u "$HERE/prompting/run_sweep.py" \
    --out-dir "$DIR/rows" --provenance "$LOGS/sweep_provenance.json" \
    --games "$GAMES" --policy "$POLICY" --samples "$SAMPLES" \
    --batch-size "$BATCH" --seed "$SEED" --null-seed "$NULL_SEED" \
    --device "$DEVICE" $RESUME_FLAG 2>&1 | tee -a "$LOGS/sweep.log"

# 3-4. everything downstream is CPU-only and reads the committed rows.
echo "=== analysis $(date -u +%FT%TZ) ==="
CUDA_VISIBLE_DEVICES="" "$PY" -u "$HERE/measurement/analyze_game.py" \
    --rows-root "$DIR/rows" --coefficients "$LOGS/coefficients.csv" \
    --policy "$POLICY" --games "$GAMES" --out "$DIR/analysis/steering.json" \
    2>&1 | tee "$LOGS/analysis.log"

echo "=== tables $(date -u +%FT%TZ) ==="
CUDA_VISIBLE_DEVICES="" "$PY" -u "$HERE/pooling/crossgame_tables.py" \
    --analysis "$DIR/analysis/steering.json" \
    --out-csv "$DIR/analysis/points.csv" 2>&1 | tee "$LOGS/tables.log"

echo "=== COMPLETE $(date -u +%FT%TZ) ==="
