#!/usr/bin/env bash
# Everything downstream of the activations, in order. CPU only, no model loaded.
#
# Sequential on purpose: every stage is CPU-bound and already runs its own thread
# pool, so overlapping them only oversubscribes the cores. Nothing writes into the
# checkout; the two stages write into $WORK/crossgame and $WORK/pooled, which is
# also what keeps their two `analysis.json` files apart.
#
#   ACTS=<activation root from run_extraction.sh> WORK=<scratch dir> \
#   PY=<python with torch, transformers and safetensors> \
#       bash results/crossgame-decision-vectors/scripts/run_analysis.sh
#
# SNAPSHOT points at a local Qwen2.5-7B-Instruct snapshot; the two decode stages
# read the embedding matrix off its safetensors shard and never run the model.
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"
DIR="$(dirname "$HERE")"
REPO="$(cd "$DIR/../.." && pwd)"

ACTS="${ACTS:?set ACTS to the activation root run_extraction.sh wrote}"
WORK="${WORK:?set WORK to the directory this stage writes into}"
PY="${PY:-python3}"
SNAPSHOT="${SNAPSHOT:-$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28}"
export DM_THREADS="${DM_THREADS:-12}"
# the stages that do not cap their own thread pool take the cap from here
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$DM_THREADS}"
export CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export DM_ACTS="$ACTS" DM_POOLED_OUT="$WORK/pooled" DM_SNAPSHOT="$SNAPSHOT"

XG="$WORK/crossgame"
PL="$WORK/pooled"
mkdir -p "$WORK/logs" "$XG" "$PL/vectors"
FAMILIES="dictator trust ultimatum apology overfishing prisoners_dilemma"

# 1. per game: its vector, its pole census and its manifest entry, landed one at a
#    time so a stopped run leaves an interpretable directory rather than nothing.
for fam in $FAMILIES; do
  echo "[$(date -u +%T)] land $fam"
  "$PY" -u "$HERE/measurement/land_family.py" --family "$fam" \
      --rows "$ACTS/$fam/rows.csv" --acts "$ACTS/$fam" --out "$XG" \
      > "$WORK/logs/land_$fam.log" 2>&1
done

# 2. the six-game battery: agreement matrix and its label nulls, leave-one-game-out,
#    split-half, the pole census, and the comparisons against the two existing vectors
echo "[$(date -u +%T)] analyze"
"$PY" -u "$HERE/measurement/analyze_crossgame.py" --manifest "$XG/manifest.json" \
    --out "$XG" > "$WORK/logs/analysis.log" 2>&1

# 3. the pooling stage: the Dictator-only vector this corpus is compared against,
#    then the nine weightings and everything measured on them
echo "[$(date -u +%T)] dictator vector"
"$PY" -u "$HERE/pooling/dictator_vector.py" > "$WORK/logs/dictator_vector.log" 2>&1
for stage in build_vectors evaluate addendum reliability; do
  echo "[$(date -u +%T)] $stage"
  "$PY" -u "$HERE/pooling/$stage.py" > "$WORK/logs/$stage.log" 2>&1
done

# 4. the layer-0 token decode. Two calls: the six-game vectors as section 3 reports
#    them, then every pooled scheme and per-game vector of sections 3 and 5.
echo "[$(date -u +%T)] token decode"
"$PY" -u "$HERE/measurement/decode_layer0.py" --top 20 --snapshot "$SNAPSHOT" \
    --out "$XG/crossgame_layer0_tokens.json" --vectors \
    "$XG/decision_pooled_response_avg_diff_cellbalanced_strict.pt" \
    "$XG/decision_overfishing_response_avg_diff_cellbalanced_strict.pt" \
    "$XG/decision_prisoners_dilemma_response_avg_diff_cellbalanced_strict.pt" \
    "$PL/vectors/decision_dictator_only_response_avg_diff_cellbalanced.pt" \
    "$REPO/persona_vectors/Qwen2.5-7B-Instruct/altruism_response_avg_diff.pt" \
    > "$WORK/logs/decode_crossgame.log" 2>&1

POOLED_VECTORS=""
for policy in strict relaxed; do
  for scheme in cell_balanced game_equal_unit game_equal_raw game_precision_unit \
                game_precision_raw family_balanced_unit family_balanced_raw \
                non_dollar_unit non_dollar_raw; do
    POOLED_VECTORS="$POOLED_VECTORS $PL/vectors/decision_pooled_${scheme}_response_avg_diff_${policy}.pt"
  done
  for fam in $FAMILIES; do
    POOLED_VECTORS="$POOLED_VECTORS $PL/vectors/decision_${fam}_response_avg_diff_cellbalanced_${policy}.pt"
  done
done
"$PY" -u "$HERE/measurement/decode_layer0.py" --top 20 --snapshot "$SNAPSHOT" \
    --out "$PL/layer0_tokens.json" --vectors $POOLED_VECTORS \
    "$PL/vectors/decision_dictator_only_response_avg_diff_cellbalanced.pt" \
    "$REPO/persona_vectors/Qwen2.5-7B-Instruct/altruism_response_avg_diff.pt" \
    > "$WORK/logs/decode_pooled.log" 2>&1

echo "[$(date -u +%T)] digit share"
"$PY" -u "$HERE/pooling/digit_share.py" > "$WORK/logs/digit_share.log" 2>&1
echo "[$(date -u +%T)] ANALYSIS COMPLETE"
