#!/usr/bin/env bash
# =============================================================================
# run_multiseed_baselines.sh
# Multi-seed supervised baselines  (reviewer round: memo #9 / reviewer #5)
# -----------------------------------------------------------------------------
# Trains each supervised model under several random seeds with ALL OTHER
# hyperparameters held fixed, evaluates every seed on the identical risk-enriched
# test set, tags each seed's output, and then aggregates mean +/- SD.
#
# The teacher-gated framework has no trainable parameters and is fully
# deterministic, so it is NOT reseeded -- that is the point of the comparison.
#
# Prereqly: the data splits (test.h5, norm_stats.json) and the Stage-2 siamese
# checkpoint must already exist, exactly as for the single-seed runs.
#
# Run:   bash run_multiseed_baselines.sh
# Then:  python aggregate_multiseed.py
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

CKPT=checkpoints
MS=logs/multiseed
mkdir -p "$MS"

# Edit these lists if you want more/fewer seeds (>=3 required, 5 recommended).
SYNTH_SEEDS=(0 1 2 3 4)
MITBIH_SEEDS=(7 1 2 3 4)

echo "############################################################"
echo "# CACHET synthetic-pair model : ${#SYNTH_SEEDS[@]} seeds"
echo "############################################################"
for S in "${SYNTH_SEEDS[@]}"; do
  echo "----- synthetic seed ${S} : train -----"
  python train_stage3_synthetic.py --seed "${S}"
  # The synthetic trainer reliably writes _last.pt every run (a _best.pt is only
  # written when val-loss improves and may be absent), so tag from _last.pt.
  if [ -f "${CKPT}/denoiser_synth_best.pt" ]; then
    cp "${CKPT}/denoiser_synth_best.pt" "${CKPT}/denoiser_synth_seed${S}.pt"
  else
    cp "${CKPT}/denoiser_synth_last.pt" "${CKPT}/denoiser_synth_seed${S}.pt"
  fi

  echo "----- synthetic seed ${S} : evaluate on test -----"
  python evaluate_stage3.py --split test --mode model --risk_sample \
         --checkpoint "${CKPT}/denoiser_synth_seed${S}.pt" \
         | tee "${MS}/synth_seed${S}.log"
  # main() writes logs/stage3_eval_test_model.json -- tag it by seed.
  cp logs/stage3_eval_test_model.json "${MS}/synth_seed${S}.json"
done

echo "############################################################"
echo "# MIT-BIH/NSTDB transfer model : ${#MITBIH_SEEDS[@]} seeds"
echo "############################################################"
for S in "${MITBIH_SEEDS[@]}"; do
  echo "----- mitbih seed ${S} : train -----"
  python train_mitbih_denoiser.py --seed "${S}"
  cp "${CKPT}/mitbih_em_denoiser_best.pt" "${CKPT}/mitbih_em_seed${S}.pt"

  echo "----- mitbih seed ${S} : evaluate transfer on test -----"
  python evaluate_stage3.py --split test --mode ecg_denoiser --risk_sample \
         --checkpoint "${CKPT}/mitbih_em_seed${S}.pt" \
         | tee "${MS}/mitbih_seed${S}.log"
  # main() writes logs/stage3_eval_test_ecg_denoiser.json -- tag it by seed.
  cp logs/stage3_eval_test_ecg_denoiser.json "${MS}/mitbih_seed${S}.json"
done

echo
echo "All seeds done. Per-seed metrics are in ${MS}/."
echo "Now run:  python aggregate_multiseed.py"
