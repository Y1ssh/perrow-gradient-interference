#!/usr/bin/env bash
# run_all_norm_support.sh — the full norm/support confirmation sweep.
#
# Runs the deciding measurement across seeds and both optimizers so the
# mechanism (support divergence vs cancellation) is established at n=3 per
# optimizer, not n=1. ~2-3 min per run on an H100 => ~15-18 min for 6 runs.
#
# From the repo root:
#     bash measurement/run_all_norm_support.sh
#
# Outputs one JSON per run under results/norm_support/, then prints a summary.
set -u
cd "$(dirname "$0")/.."   # repo root

SEEDS=(42 123 456)
OPTS=(muon adamw)

for opt in "${OPTS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    echo ""
    echo "########## optimizer=$opt seed=$seed ##########"
    python measurement/run_norm_support.py --optimizer "$opt" --seed "$seed"
  done
done

# Optional: 350M single-seed check (uncomment if you want the scale point).
# Needs model/gpt2_medium.py wired into the driver; leave off unless asked.
# python measurement/run_norm_support.py --optimizer muon --seed 42 --scale 350m

echo ""
echo "########## AGGREGATE ##########"
python analysis/aggregate_norm_support.py --results results/norm_support --out analysis
echo ""
echo "Done. Table: analysis/norm_support_table.md"
