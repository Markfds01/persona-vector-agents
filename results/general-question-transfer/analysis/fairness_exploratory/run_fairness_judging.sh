#!/usr/bin/env bash
# EXPLORATORY (docs/fairness_exploratory_prereg.md section 2): judges the existing held-out
# pilot answers with the fairness rubric. Primary judge first (all 9 held-out points, 1,080
# rows), then the secondary judge on its pre-registered 30-of-120 subset per point. Each call
# is resumable: a re-run sends only calls with no finished row. Needs the two endpoints:
#   PRIMARY_URL=<llama-server, Llama-3.3-70B> SECONDARY_URL=<llama-server, gpt-oss-20b> \
#       bash run_fairness_judging.sh
set -eu
set -o pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
GQT="$(cd "$HERE/../.." && pwd)"
REPO="$(cd "$GQT/../.." && pwd)"
PY="${PY:-python3}"
LOG="$HERE/judging_fairness.log"
cd "$REPO"
run() {  # judge url
  echo "=== fairness judging $(date -u +%FT%TZ): judge=$1" | tee -a "$LOG"
  PIMIENTOROJO_BASE_URL="$2" "$PY" -u "$GQT/scripts/judging/run_judging.py" --judge "$1" --phase judge \
      --rows-dir "$GQT/rows" --out-dir "$GQT/judgments" --kinds fairness 2>&1 | tee -a "$LOG"
}
run primary "${PRIMARY_URL:?set PRIMARY_URL}"
run secondary "${SECONDARY_URL:?set SECONDARY_URL}"
echo "=== COMPLETE $(date -u +%FT%TZ)" | tee -a "$LOG"
