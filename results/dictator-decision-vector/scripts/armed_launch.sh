#!/bin/bash
# Single-shot opportunistic launch: wait for device 0 to have room, then run the
# orthogonal-null arm at +-10.51 ONCE. It never retries - if the run dies (the
# tenant's ollama daemon cycles models on its own schedule and can evict us
# mid-arm), this exits and reports, leaving whatever coefficients landed on disk.
LOG=/home/marco/dockmaster/data/steer-decision/orthogonal_null_pm1.log
deadline=$(( $(date +%s) + 3600 ))
while [ "$(date +%s)" -lt "$deadline" ]; do
  free0=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0)
  if [ "$free0" -ge 20000 ]; then
    echo "LAUNCHING: device 0 free=${free0}MiB at $(date -u +%H:%M:%SZ)" | tee -a "$LOG"
    cd /home/marco/dockmaster/state/worktrees/steer-decision
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python scratch/run_arm.py \
      --vector /home/marco/dockmaster/data/steer-decision/decision_shuffled_orthogonalised_seed20260819.pt \
      --their-vector /home/marco/dockmaster/state/worktrees/steer-decision/persona_vectors/Qwen2.5-7B-Instruct/altruism_response_avg_diff.pt \
      --out-dir /home/marco/dockmaster/data/steer-decision/rows/orthogonal-null \
      --arm orthogonal-null --ks 1,-1 \
      --manifest /home/marco/dockmaster/data/steer-decision/coefficients.csv >> "$LOG" 2>&1
    echo "RUN_EXIT=$?" | tee -a "$LOG"
    exit 0
  fi
  sleep 15
done
echo "NEVER_LAUNCHED: device 0 stayed under 20GB for 60 min" | tee -a "$LOG"
exit 1
