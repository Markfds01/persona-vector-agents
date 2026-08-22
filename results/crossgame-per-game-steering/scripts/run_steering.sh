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
# POLICY picks the arm and OWNS EVERY OUTPUT PATH. strict writes rows/,
# analysis/steering.json, analysis/points.csv and provenance/; relaxed writes
# rows_relaxed/, analysis/steering_relaxed.json, analysis/points_relaxed.csv and
# provenance/relaxed/. A row's filename carries its game, arm and coefficient but
# NOT its policy, so the two arms must not share an output root — the second run
# would overwrite the first, and both rounds are committed.
#
# The relaxed arm as it was actually run — five games, not six. The Prisoner's
# Dilemma's relaxed vectors are element-wise identical to its strict ones (its
# answer space has two points and no middle), so its strict rows ARE its relaxed
# rows and regenerating them would only reproduce them:
#
#   POLICY=relaxed GAMES=ultimatum,overfishing,dictator,trust,apology \
#   ACTS=<...> PY=<...> SAMPLES=100 DEVICE=1 bash .../run_steering.sh
#
# Stage 1 still builds all six relaxed nulls; GAMES restricts the sweep only.
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

# Every output below is suffixed by the policy, so one arm can never land on the
# other's committed artifacts. An unknown policy stops here rather than inventing
# a third set of paths nothing else in the package reads.
case "$POLICY" in
  strict)  SUFFIX="";         LOG_SUBDIR="" ;;
  relaxed) SUFFIX="_relaxed"; LOG_SUBDIR="/relaxed" ;;
  *) echo "POLICY must be strict or relaxed, not '$POLICY'" >&2; exit 2 ;;
esac
PROVENANCE="$DIR/provenance"
# derived, never taken from the environment: LOGS holds sweep_provenance.json,
# coefficients.csv and both logs, so an override here defeats the guarantee above
LOGS="$PROVENANCE$LOG_SUBDIR"
ROWS="$DIR/rows$SUFFIX"
ANALYSIS_JSON="$DIR/analysis/steering$SUFFIX.json"
POINTS_CSV="$DIR/analysis/points$SUFFIX.csv"
# the null report stays in provenance/ under BOTH policies, suffixed rather than
# nested: build_vector_manifest.load_reports reads exactly these two names
NULL_REPORT="$PROVENANCE/null_vectors$SUFFIX.json"
NULL_LOG="$PROVENANCE/null_vectors$SUFFIX.log"

RESUME_FLAG=""
[ "${RESUME:-0}" = "1" ] && RESUME_FLAG="--resume"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

cd "$REPO"
mkdir -p "$DIR/vectors" "$ROWS" "$DIR/analysis" "$LOGS" "$PROVENANCE"
echo "=== policy $POLICY: rows -> $ROWS, analysis -> $ANALYSIS_JSON, logs -> $LOGS"

# 1. the matched nulls. CPU only, and it refuses to write one unless it can first
#    rebuild the committed vector it is a null of. All six, whatever GAMES says:
#    the sweep may skip a game whose vectors are identical to its strict pair, but
#    the manifest still describes every null.
if [ "${SKIP_NULLS:-0}" != "1" ]; then
  echo "=== null vectors $(date -u +%FT%TZ) ==="
  ACTS="${ACTS:?set ACTS to the activation root, or SKIP_NULLS=1 to use the committed nulls}"
  CUDA_VISIBLE_DEVICES="" "$PY" -u "$HERE/measurement/build_nulls.py" \
      --acts "$ACTS" --out-dir "$DIR/vectors" --policy "$POLICY" \
      --seed "$NULL_SEED" --report "$NULL_REPORT" \
      2>&1 | tee "$NULL_LOG"
fi

# 2. the sweep itself. One process holds the weights for every game — the card is
#    shared and each load is a fresh wait for headroom — but each game lands its
#    own manifest entry the moment it finishes, and each beta point its own CSV.
#    run_sweep.py writes coefficients.csv beside --provenance, so that lands under
#    this policy's log directory too.
echo "=== sweep $(date -u +%FT%TZ) ==="
"$PY" -u "$HERE/prompting/run_sweep.py" \
    --out-dir "$ROWS" --provenance "$LOGS/sweep_provenance.json" \
    --games "$GAMES" --policy "$POLICY" --samples "$SAMPLES" \
    --batch-size "$BATCH" --seed "$SEED" --null-seed "$NULL_SEED" \
    --device "$DEVICE" $RESUME_FLAG 2>&1 | tee -a "$LOGS/sweep.log"

# 3-4. everything downstream is CPU-only and reads the committed rows.
echo "=== analysis $(date -u +%FT%TZ) ==="
CUDA_VISIBLE_DEVICES="" "$PY" -u "$HERE/measurement/analyze_game.py" \
    --rows-root "$ROWS" --coefficients "$LOGS/coefficients.csv" \
    --policy "$POLICY" --games "$GAMES" --out "$ANALYSIS_JSON" \
    2>&1 | tee "$LOGS/analysis.log"

echo "=== tables $(date -u +%FT%TZ) ==="
CUDA_VISIBLE_DEVICES="" "$PY" -u "$HERE/pooling/crossgame_tables.py" \
    --analysis "$ANALYSIS_JSON" \
    --out-csv "$POINTS_CSV" 2>&1 | tee "$LOGS/tables.log"

echo "=== COMPLETE $(date -u +%FT%TZ) ==="
