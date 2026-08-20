#!/bin/bash
# The full cross-game run, ordered so that an eviction costs the least important game.
#
# Sample counts differ per game, and the reason is the calibration pass, not taste:
#   overfishing        48 - its self-interested pole ran at 2/48 in calibration, and
#                           it is the only remaining game whose answer surface is not
#                           dollars, so it needs the most samples to fill that pole
#   the amount games   24 - calibration filled both their poles comfortably
#   prisoners_dilemma   8 - census only. It defected 143/143 across 12 distinct
#                           payoff matrices, so no vector can be built from it; these
#                           rows exist to report the corner over the whole grid
#                           rather than over the calibration subset.
set -u
R=/home/marco/dockmaster/state/worktrees/crossgame-vector
BATCH=32 SEED=0 DEVICE=0
SAMPLES=48 FAMILIES="overfishing"                                 BATCH=$BATCH SEED=$SEED DEVICE=$DEVICE bash "$R/scratch/run_all.sh" || exit 1
SAMPLES=24 FAMILIES="dictator apology trust ultimatum"            BATCH=$BATCH SEED=$SEED DEVICE=$DEVICE bash "$R/scratch/run_all.sh" || exit 1
SAMPLES=8  FAMILIES="prisoners_dilemma"                           BATCH=$BATCH SEED=$SEED DEVICE=$DEVICE bash "$R/scratch/run_all.sh" || exit 1
echo "MAIN RUN COMPLETE $(date -Is)"
