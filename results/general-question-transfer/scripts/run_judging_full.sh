#!/usr/bin/env bash
# Full run, primary judge, one or more rubrics (DEVIATIONS 2026-11-17), each followed by its 5%
# determinism re-send (PREREG 9.2, seed 20261026). Reads rows_full/, writes judgments/full/ and
# provenance/judging_full_primary_<kind>.log. Resumable: a re-launch sends only the calls without a
# finished row, and the re-send skips payloads already re-sent. Freeze check first, every launch.
#
#   PRIMARY_URL=<llama-server endpoint> KINDS="coherence pairwise" bash scripts/run_judging_full.sh
#   (kinds: coherence pairwise fairness generosity; trait was run by run_judging_full_trait.sh)
set -eu
set -o pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DIR="$(dirname "$HERE")"
REPO="$(cd "$DIR/../.." && pwd)"
PY="${PY:-python3}"
export PYTHONDONTWRITEBYTECODE=1
cd "$REPO"
for kind in ${KINDS:?set KINDS, e.g. "coherence pairwise"}; do
  LOG="$DIR/provenance/judging_full_primary_${kind}.log"
  python3 "$HERE/provenance/freeze.py" --check | tee -a "$LOG"
  for phase in judge resend; do
    echo "=== judging $(date -u +%FT%TZ): judge=primary phase=$phase kinds=$kind (full run)" | tee -a "$LOG"
    PIMIENTOROJO_BASE_URL="${PRIMARY_URL:?set PRIMARY_URL}" "$PY" -u "$HERE/judging/run_judging.py" \
        --judge primary --full --phase "$phase" --rows-dir "$DIR/rows_full" --out-dir "$DIR/judgments/full" \
        --kinds "$kind" 2>&1 | tee -a "$LOG"
  done
  echo "=== COMPLETE $kind $(date -u +%FT%TZ)" | tee -a "$LOG"
done
