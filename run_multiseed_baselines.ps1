# =============================================================================
# run_multiseed_baselines.ps1   (Windows PowerShell version)
# Multi-seed supervised baselines  (reviewer round: memo #9 / reviewer #5)
#
# Trains each supervised model under several seeds with all other hyperparameters
# fixed, evaluates each seed on the identical risk-enriched test set, tags each
# seed's output, then aggregates mean +/- SD. The teacher-gated framework is
# deterministic and is NOT reseeded.
#
# Run from the project folder in your torch_env:
#   .\run_multiseed_baselines.ps1
# Then:
#   python aggregate_multiseed.py
# =============================================================================
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$CKPT = "checkpoints"
$MS   = "logs\multiseed"
New-Item -ItemType Directory -Force -Path $MS | Out-Null

# Edit these if you want more/fewer seeds (>=3 required, 5 recommended).
$SYNTH_SEEDS  = @(0,1,2,3,4)
$MITBIH_SEEDS = @(7,1,2,3,4)

Write-Host "############################################################"
Write-Host "# CACHET synthetic-pair model : $($SYNTH_SEEDS.Count) seeds"
Write-Host "############################################################"
foreach ($S in $SYNTH_SEEDS) {
    Write-Host "----- synthetic seed $S : train -----"
    python train_stage3_synthetic.py --seed $S
    # _last.pt is written every run; _best.pt only when val-loss improves.
    if (Test-Path "$CKPT\denoiser_synth_best.pt") {
        Copy-Item "$CKPT\denoiser_synth_best.pt" "$CKPT\denoiser_synth_seed$S.pt" -Force
    } else {
        Copy-Item "$CKPT\denoiser_synth_last.pt" "$CKPT\denoiser_synth_seed$S.pt" -Force
    }
    Write-Host "----- synthetic seed $S : evaluate on test -----"
    python evaluate_stage3.py --split test --mode model --risk_sample --checkpoint "$CKPT\denoiser_synth_seed$S.pt"
    Copy-Item "logs\stage3_eval_test_model.json" "$MS\synth_seed$S.json" -Force
}

Write-Host "############################################################"
Write-Host "# MIT-BIH/NSTDB transfer model : $($MITBIH_SEEDS.Count) seeds"
Write-Host "############################################################"
foreach ($S in $MITBIH_SEEDS) {
    Write-Host "----- mitbih seed $S : train -----"
    python train_mitbih_denoiser.py --seed $S
    Copy-Item "$CKPT\mitbih_em_denoiser_best.pt" "$CKPT\mitbih_em_seed$S.pt" -Force
    Write-Host "----- mitbih seed $S : evaluate transfer on test -----"
    python evaluate_stage3.py --split test --mode ecg_denoiser --risk_sample --checkpoint "$CKPT\mitbih_em_seed$S.pt"
    Copy-Item "logs\stage3_eval_test_ecg_denoiser.json" "$MS\mitbih_seed$S.json" -Force
}

Write-Host ""
Write-Host "All seeds done. Per-seed metrics are in $MS\."
Write-Host "Now run:  python aggregate_multiseed.py"
