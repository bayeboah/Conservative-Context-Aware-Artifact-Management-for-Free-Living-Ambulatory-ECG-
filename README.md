# Conservative Context-Aware Artifact Management for Free-Living Ambulatory ECG

Official implementation and reproducibility artifacts for **Conservative Context-Aware Artifact Management for Free-Living Ambulatory ECG**.

## Overview

This repository implements and evaluates a deterministic, context-gated artifact-management framework (`teacher_gated`) on free-living ambulatory ECG from **CACHET-CADB**, alongside two supervised comparators:
1. A context-informed Cross-Attention 1D U-Net (`ResidualArtifactRemover`) trained on CACHET-derived synthetic pairs (`train_stage3_synthetic.py`).
2. An ECG-only 1D U-Net (`ECGUNet1D`) trained on MIT-BIH Arrhythmia + NSTDB electrode-motion noise pairs (`build_mitbih_em.py`, `train_mitbih_denoiser.py`) and transferred to CACHET-CADB without fine-tuning.

## Repository Contents

| Path | Description |
| :--- | :--- |
| `dataset.py`, `emg_detector.py`, `compute_norm.py` | CACHET-CADB chunked streaming preprocessing, EMG index computation, and training-only normalization |
| `split_config.py`, `build_context_stratified_split.py` | Participant-disjoint train/validation/test split search and validation |
| `output/context_stratified_split.json` | Authoritative participant-disjoint split manifest (16 train / 4 val / 4 test) |
| `output/norm_stats.json` | Training-partition ECG and IMU normalization statistics |
| `siamese_alignment.py`, `train_stage2.py`, `evaluate_stage2.py` | Stage 2 ECG–IMU contrastive/contextual encoder |
| `denoiser.py` | Deterministic context-gated framework (`generate_pseudo_target`, `gated_teacher_denoise`) and `ResidualArtifactRemover` |
| `train_stage3_synthetic.py` | Training script for the CACHET-derived supervised comparator |
| `build_mitbih_em.py`, `train_mitbih_denoiser.py` | MIT-BIH/NSTDB corpus builder and ECG-only transfer comparator training |
| `evaluate_stage3.py` | Deterministic 25,600-window risk-enriched evaluation (seed 13), paired participant-cluster bootstrap statistics, gating decomposition, and ablations |
| `audit_qrs_template_counts.py`, `per_patient_effects.py` | QRS-template eligibility audit and per-participant effect breakdowns |
| `logs/` | Authoritative per-stratum evaluation JSON/CSV outputs and environment snapshots (`environment_pip_freeze.txt`, `environment_summary.txt`) |
| `checkpoints/SHA256SUMS.txt` | SHA-256 verification checksums for all trained `.pt` checkpoints |
| `requirements.txt` | Documented compatible Python package-version ranges |

## Setup

```bash
pip install -r requirements.txt
```

Set dataset paths in `config.json` to point to your local copies of CACHET-CADB and MIT-BIH/NSTDB (both are publicly available from their original repositories). See [FINAL_PIPELINE_COMMANDS.md](FINAL_PIPELINE_COMMANDS.md) and [PIPELINE_SUMMARY.md](PIPELINE_SUMMARY.md) for the exact evaluation commands.
