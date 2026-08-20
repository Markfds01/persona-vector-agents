#!/bin/bash
# wait for the running null-extension to finish, then run the orthogonalised arm
while kill -0 3789429 2>/dev/null; do sleep 20; done
cd /home/marco/dockmaster/state/worktrees/steer-decision
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scratch/run_arm.py \
  --vector /home/marco/dockmaster/data/steer-decision/decision_shuffled_orthogonalised_seed20260819.pt \
  --their-vector /home/marco/dockmaster/state/worktrees/steer-decision/persona_vectors/Qwen2.5-7B-Instruct/altruism_response_avg_diff.pt \
  --out-dir /home/marco/dockmaster/data/steer-decision/rows/orthogonal-null \
  --arm orthogonal-null --ks 5,-5 \
  --manifest /home/marco/dockmaster/data/steer-decision/coefficients.csv
