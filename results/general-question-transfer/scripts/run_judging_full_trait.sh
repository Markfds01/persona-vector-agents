#!/usr/bin/env bash
# Full run, primary judge, altruism trait pass only (DEVIATIONS 2026-11-04), then its 5% determinism
# re-send. Reads rows_full/, writes judgments/full/ and provenance/judging_full_primary.log.
# Resumable: a re-launch sends only the calls without a finished row.
#
#   PRIMARY_URL=<llama-server endpoint, tunnel to PimientoRojo> bash scripts/run_judging_full_trait.sh
set -eu
set -o pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DIR="$(dirname "$HERE")"
REPO="$(cd "$DIR/../.." && pwd)"
PY="${PY:-python3}"
LOG="$DIR/provenance/judging_full_primary.log"
cd "$REPO"
python3 "$HERE/provenance/freeze.py" --check | tee -a "$LOG"
for phase in judge resend; do
  echo "=== judging $(date -u +%FT%TZ): judge=primary phase=$phase kinds=trait (full run)" | tee -a "$LOG"
  PIMIENTOROJO_BASE_URL="${PRIMARY_URL:?set PRIMARY_URL}" "$PY" -u "$HERE/judging/run_judging.py" \
      --judge primary --phase "$phase" --rows-dir "$DIR/rows_full" --out-dir "$DIR/judgments/full" \
      --kinds trait 2>&1 | tee -a "$LOG"
done
echo "=== COMPLETE $(date -u +%FT%TZ)" | tee -a "$LOG"
