#!/bin/bash
set -u
R=/home/marco/dockmaster/state/worktrees/crossgame-vector
OUT=/home/marco/dockmaster/data/crossgame-vector
PY=$R/.venv/bin/python
bash "$R/scratch/resume_run.sh" prisoners_dilemma:8
