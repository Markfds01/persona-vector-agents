#!/usr/bin/env bash
# Stage: judge validation and the human calibration export (checkpoint 2).
# Reads rows/ and judgments/; owns human/, provenance/human_sample_key.json, analysis/pilot_validation.json
# and analysis/pilot_report.md. CPU only, no network, no judge calls.
#
#   [PY=python3] bash results/general-question-transfer/scripts/run_validation.sh
#
# human/ holds the blind sheet, the grader page and the judge-reveal files; send the labellers the page
# (and sheet) only. The id-to-row key (provenance/human_sample_key.json) and the reveal files stay with the
# operator until both labellers are done.
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"
DIR="$(dirname "$HERE")"
REPO="$(cd "$DIR/../.." && pwd)"
PY="${PY:-python3}"
cd "$REPO"
mkdir -p "$DIR/analysis" "$DIR/human"
echo "=== calibration export $(date -u +%FT%TZ)"
"$PY" "$HERE/human/export_calibration.py" --rows-dir "$DIR/rows" --human-dir "$DIR/human" \
    --key "$DIR/provenance/human_sample_key.json" --judgments-dir "$DIR/judgments"
echo "=== validation numbers $(date -u +%FT%TZ)"
"$PY" "$HERE/validation/pilot_validation.py" --rows-dir "$DIR/rows" --judgments-dir "$DIR/judgments" \
    --provenance-dir "$DIR/provenance" --out "$DIR/analysis/pilot_validation.json"
echo "=== report $(date -u +%FT%TZ)"
"$PY" "$HERE/validation/write_report.py" --validation "$DIR/analysis/pilot_validation.json" \
    --human-key "$DIR/provenance/human_sample_key.json" --out "$DIR/analysis/pilot_report.md"
echo "=== COMPLETE $(date -u +%FT%TZ)"
