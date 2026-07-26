#!/usr/bin/env bash
# Round-2 Item-2 mixture test: MTP-weight sweep + checkpoints + T1/T2/T3 eval.
# Each run saves results/checkpoints/model_scale{S}_seed42.pt. ~45-65 min/run.
# Scales chosen so every predicted gap clears the ~0.05-nat MDE; 0.0=CE-only anchor, 1.0=shared-MTP anchor.
set -e
cd "$(dirname "$0")/.."
for SCALE in 0.0 1.0 0.25 0.5; do
  echo "=== phase_e mtp_scale=$SCALE seed=42 ==="
  python3 experiments/phase_e_mtp_weight_sweep.py --mtp_scale $SCALE --seed 42
done
echo
echo "=== T1: zero-training KL prediction from the CE-only checkpoint (the real hypothesis test) ==="
# expandable_segments avoids the fragmentation OOM seen on the 80GB card; k-scan reads against the per-scale band.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 analysis/estimate_kl.py \
  --ce_ckpt   results/checkpoints/model_scale0.0_seed42.pt \
  --mtp_ckpt  results/checkpoints/model_scale1.0_seed42.pt \
  --topk_list 256 1024 2048 2>&1 | tee kl_scan.log
echo
echo "=== Ceilings + control (re-measures CE-vs-L1 control + CE-vs-CE ceiling on the step-1000 snapshots) ==="
python3 analysis/measure_ceilings.py \
  --ce_ckpt  results/checkpoints/model_scale0.0_seed42_step1000.pt \
  --mtp_ckpt results/checkpoints/model_scale1.0_seed42_step1000.pt 2>&1 | tee ceilings_rerun.log
echo
echo "=== Sweep verdict table (compares measured gaps to estimate_kl's predicted curve) ==="
python3 analysis/analyze_mtp_sweep.py
