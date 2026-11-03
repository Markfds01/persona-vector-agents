#!/usr/bin/env bash
# The full run (PREREG D1 div8_main): 13,200 rows plus the 120-row unhooked_repeat control,
# into rows_full/ and provenance/generation_full/ (DEVIATIONS 2026-11-03). Four processes,
# two per A40; each runs its share of the points through run_generation.sh in order.
#
#   PY=~/venvs/pva-torch260-tf4523/bin/python bash scripts/run_full_generation.sh <part>
#
# <part> is 1..4. Every part is resumable (RESUME=1 is always on): a re-run skips whole points
# and regenerates short ones. Run all four, detached, e.g.
#   for p in 1 2 3 4; do setsid nohup bash scripts/run_full_generation.sh $p >/dev/null 2>&1 & done
set -eu
set -o pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DIR="$(dirname "$HERE")"
export PY="${PY:?set PY to the torch 2.6.0+cu124 / transformers 4.52.3 interpreter}"
export ROWS="$DIR/rows_full" PROVENANCE="$DIR/provenance/generation_full" SAMPLES=3 RESUME=1
export OMP_NUM_THREADS=8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p "$PROVENANCE"

DECISION="decision/dictator,decision/trust,decision/ultimatum,decision/apology,decision/overfishing,decision/prisoners_dilemma"
SECONDARY_DECISION="relaxed/ultimatum,relaxed/overfishing,dictator_only"
SHUFFLED="shuffled_null/dictator,shuffled_null/trust,shuffled_null/ultimatum,shuffled_null/apology,shuffled_null/overfishing,shuffled_null/prisoners_dilemma"
ORTHOGONAL="orthogonal_null/dictator,orthogonal_null/trust,orthogonal_null/ultimatum,orthogonal_null/apology,orthogonal_null/overfishing,orthogonal_null/prisoners_dilemma"
TRAIT="trait/altruism"

run() {  # sets arms ks controls(0|1)
  local nc=1
  [ "$4" = "1" ] && nc=0
  LOG="$PROVENANCE/generation_part${PART}.log" SETS="$1" ARMS="$2" KS="$3" NO_CONTROLS="$nc" \
    bash "$HERE/run_generation.sh"
}

PART="${1:?part 1..4}"
case "$PART" in
  1)  # GPU 0: held-out controls (baseline, unhooked, unhooked_repeat), the 9 decision arms, trait +-1/+-5
      export DEVICE=0
      run heldout "$DECISION,$SECONDARY_DECISION" "-3,3" 1
      run heldout "$TRAIT" "-5,-1,1,5" 0 ;;
  2)  # GPU 0: held-out nulls and the trait vector at +-3
      export DEVICE=0
      run heldout "$SHUFFLED,$ORTHOGONAL,$TRAIT" "-3,3" 0 ;;
  3)  # GPU 1: in-distribution (baseline, 6 decision vectors + trait), bridge baseline + 9 decision arms
      export DEVICE=1
      run in_distribution "$DECISION,$TRAIT" "-3,3" 1
      run bridge "$DECISION,$SECONDARY_DECISION" "-3,3" 1 ;;
  4)  # GPU 1: bridge nulls and the trait vector at +-3
      export DEVICE=1
      run bridge "$SHUFFLED,$ORTHOGONAL,$TRAIT" "-3,3" 0 ;;
  *) echo "part must be 1..4" >&2; exit 2 ;;
esac
echo "=== PART $PART COMPLETE $(date -u +%FT%TZ)" | tee -a "$PROVENANCE/generation_part${PART}.log"
