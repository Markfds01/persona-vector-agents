#!/usr/bin/env bash
# Stage: generation. Owns rows/ and provenance/generation_*.{json,log}; nothing else.
#
#   PY=<python with torch+transformers> [DEVICE=<gpu>] [SETS=heldout,bridge] \
#   [ARMS=...] [KS=-3,3] [SAMPLES=3] [RESUME=1] bash results/general-question-transfer/scripts/run_generation.sh
#
# The defaults are the checkpoint-2 pilot (PREREG section 14): the Dictator strict decision
# vector, the trait vector and the Dictator's shuffled and orthogonal nulls at k in
# {-3, 0, +3}, 3 samples per question, held-out and bridge sets. DEVICE defaults to the A40
# with the most free memory. Rows are never overwritten by a resume that finds them whole;
# a short point is regenerated whole; a whole point generated with another vector is an error.
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"
DIR="$(dirname "$HERE")"
REPO="$(cd "$DIR/../.." && pwd)"
PY="${PY:-python3}"
SETS="${SETS:-heldout,bridge}"
ARMS="${ARMS:-decision/dictator,trait/altruism,shuffled_null/dictator,orthogonal_null/dictator}"
KS="${KS:--3,3}"
SAMPLES="${SAMPLES:-3}"
if [ -z "${DEVICE:-}" ]; then
  DEVICE="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -n -r | head -1 | cut -d, -f1)"
fi
RESUME_FLAG=""
[ "${RESUME:-0}" = "1" ] && RESUME_FLAG="--resume"
ROWS="$DIR/rows"
PROVENANCE="$DIR/provenance"
LOG="$PROVENANCE/generation_$(echo "$SETS" | tr ',' '_').log"
mkdir -p "$ROWS" "$PROVENANCE"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
cd "$REPO"
echo "=== generation $(date -u +%FT%TZ): sets=$SETS arms=$ARMS ks=$KS samples=$SAMPLES device=$DEVICE" | tee -a "$LOG"
"$PY" -u "$HERE/generation/run_generation.py" --rows-dir "$ROWS" --provenance-dir "$PROVENANCE" \
    --sets "$SETS" --arms "$ARMS" --ks="$KS" --samples "$SAMPLES" --device "$DEVICE" \
    $RESUME_FLAG 2>&1 | tee -a "$LOG"
echo "=== COMPLETE $(date -u +%FT%TZ)" | tee -a "$LOG"
