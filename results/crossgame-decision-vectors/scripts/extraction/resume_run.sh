#!/bin/bash
# Finish the remaining games after the trust generation was OOM-killed mid-run.
#
# What actually failed: the other tenant on device 0 grew to 27.8 GiB while this
# run held 16.6 GiB, leaving 70 MiB free, and a forward pass inside generation
# died. The existing retry loop only guards the model LOAD, so it did not help.
#
# The unit of retry here is a WHOLE GAME, not a batch. `audit.generate.run` seeds
# each batch as seed + batch_index, so batch composition determines the draws:
# resuming a half-finished game from its CSV would silently generate the rest
# under a different composition than its seed implies. Restarting the game costs
# ~25 minutes and keeps each game's sampling exactly what its seed says it is.
#
# Batch size stays 32, the same value the three landed games used, so nothing
# about the sampling differs across games. Batch size was never the problem: the
# peak above the weights is ~1.4 GiB, against a 6 GiB swing from the other tenant.
# The protection that matters is refusing to start until the card has room.
set -u
R=/home/marco/dockmaster/state/worktrees/crossgame-vector
OUT=/home/marco/dockmaster/data/crossgame-vector
PY=$R/.venv/bin/python
SEED=0 DEVICE=0
# The other tenant swings between ~21.6 and ~27.4 GiB on device 0. At its peak
# that leaves ~17 GiB against a 15.2 GiB model, and a forward pass inside
# generation dies - which is exactly how the first trust attempt was lost. The
# ladder walks the batch down until one size survives instead of waiting for a
# tenant that may never leave.
#
# Batch size changes WHICH random draws happen (batches are seeded
# seed + batch_index) but not the distribution being drawn from: a game generated
# at batch 8 is still 24 samples per cell at temperature 1.0 from the same model,
# simply not byte-identical to what batch 32 would have produced. Every row
# records its own batch_size, so each game's provenance is on the row itself.
BATCH_LADDER=${BATCH_LADDER:-"32 16 8 4"}
NEED_MIB=${NEED_MIB:-16600}     # weights ~15.2 GiB + a working set + headroom

free_mib () { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$DEVICE" | tr -d ' '; }

wait_for_room () {
  local waited=0
  while [ "$(free_mib)" -lt "$NEED_MIB" ]; do
    if [ $((waited % 300)) -eq 0 ]; then
      echo "waiting for device $DEVICE: $(free_mib) MiB free, need $NEED_MIB (${waited}s)"
    fi
    sleep 20; waited=$((waited + 20))
    if [ "$waited" -ge 3600 ]; then echo "GAVE UP waiting for room after 1h"; return 1; fi
  done
  echo "device $DEVICE has $(free_mib) MiB free; starting"
}

for spec in "$@"; do
  fam=${spec%%:*}; samples=${spec##*:}
  ok=no
  for BATCH in $BATCH_LADDER; do
    echo "=== $fam: generate at batch $BATCH ($(date -Is)) ==="
    wait_for_room || break
    if "$PY" "$R/scratch/gen_crossgame.py" --out "$R/scratch/rows/$fam.csv" \
        --samples "$samples" --batch-size "$BATCH" --seed "$SEED" \
        --families "$fam" --device "$DEVICE"; then
      ok=yes; echo "$fam generated at batch $BATCH"; break
    fi
    echo "generate $fam at batch $BATCH failed; stepping the batch down"
  done
  if [ "$ok" != yes ]; then echo "FAILED generate $fam at every batch size"; exit 1; fi

  echo "=== $fam: extract ($(date -Is)) ==="
  wait_for_room || exit 1
  "$PY" "$R/scratch/extract_crossgame.py" --rows "$R/scratch/rows/$fam.csv" \
      --out-dir "$OUT/acts/$fam" --device "$DEVICE" || { echo "FAILED extract $fam"; exit 1; }

  echo "=== $fam: land ($(date -Is)) ==="
  cp "$R/scratch/rows/$fam.csv" "$OUT/acts/$fam/rows.csv"
  "$PY" "$R/scratch/land_family.py" --family "$fam" --rows "$OUT/acts/$fam/rows.csv" \
      --acts "$OUT/acts/$fam" --out "$OUT" || { echo "FAILED land $fam"; exit 1; }
  echo "=== $fam: landed ($(date -Is)) ==="
done
echo "RESUME RUN COMPLETE $(date -Is)"
