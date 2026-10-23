#!/usr/bin/env bash
# Stage: judging. Reads rows/, owns judgments/ and provenance/judging_*.log; nothing else.
#
#   PRIMARY_URL=<llama-server endpoint> SECONDARY_URL=<gpt-oss endpoint> \
#   [PY=python3] [JUDGES="secondary primary"] [KINDS_PRIMARY="trait,coherence pairwise"] \
#       bash results/general-question-transfer/scripts/run_judging.sh
#
# The endpoints are the judges' (PimientoRojo, through the SSH tunnel); they come from the
# environment and go no further: run_judging.py reads them as PIMIENTOROJO_BASE_URL and
# writes them nowhere. If an endpoint is down the run stops with an error and writes nothing.
#
# Order, by default: the secondary judge's whole pass first, in one sitting (PREREG D7: the gpt-oss
# prompt carries the day it runs), then its determinism re-send; then the primary judge's trait and
# coherence calls, its pairwise calls, and its determinism re-send. Each call is resumable: a re-run
# sends only the calls that have no finished row.
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"
DIR="$(dirname "$HERE")"
REPO="$(cd "$DIR/../.." && pwd)"
PY="${PY:-python3}"
JUDGES="${JUDGES:-secondary primary}"
KINDS_PRIMARY="${KINDS_PRIMARY:-trait,coherence pairwise}"
KINDS_SECONDARY="${KINDS_SECONDARY:-trait,coherence,pairwise}"
ROWS="$DIR/rows"
OUT="$DIR/judgments"
PROVENANCE="$DIR/provenance"
mkdir -p "$PROVENANCE"
cd "$REPO"
run() {  # judge phase kinds
  local judge="$1" phase="$2" kinds="$3" url
  if [ "$judge" = primary ]; then url="${PRIMARY_URL:?set PRIMARY_URL}"; else url="${SECONDARY_URL:?set SECONDARY_URL}"; fi
  echo "=== judging $(date -u +%FT%TZ): judge=$judge phase=$phase kinds=$kinds" | tee -a "$PROVENANCE/judging_$judge.log"
  PIMIENTOROJO_BASE_URL="$url" "$PY" -u "$HERE/judging/run_judging.py" --judge "$judge" --phase "$phase" \
      --rows-dir "$ROWS" --out-dir "$OUT" --kinds "$kinds" 2>&1 | tee -a "$PROVENANCE/judging_$judge.log"
}
for judge in $JUDGES; do
  if [ "$judge" = secondary ]; then
    run secondary judge "$KINDS_SECONDARY"
    run secondary resend "$KINDS_SECONDARY"
  else
    for kinds in $KINDS_PRIMARY; do run primary judge "$kinds"; done
    run primary resend "trait,coherence,pairwise"
  fi
done
echo "=== COMPLETE $(date -u +%FT%TZ)" | tee -a "$PROVENANCE/judging_stage.log"
