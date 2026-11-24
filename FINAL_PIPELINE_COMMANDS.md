# Final Pipeline Commands

Run these commands from the repository root after configuring paths in `config.json` (or via environment variables) and installing dependencies with `pip install -r requirements.txt`.

## 1. Split Verification and Artifact Profile

```bash
python split_config.py
python artifact_profile.py --top 12
```

Key outputs:
- `output/context_stratified_split.json`
- `logs/artifact_profile_patient.csv`
- `logs/artifact_profile_patient_day.csv`
- `logs/artifact_profile_summary.json`

## 2. Official Stage 3 Evaluation (Context-Gated Framework)

```bash
python evaluate_stage3.py --split val --mode teacher_gated --risk_sample
python evaluate_stage3.py --split test --mode teacher_gated --risk_sample
```

Key outputs:
- `logs/stage3_eval_val_teacher_gated.json`
- `logs/stage3_eval_test_teacher_gated.json`

## 3. Supervised Comparators & Paired Statistics (Tables 3–6)

```bash
# CACHET-derived synthetic comparator evaluation
python evaluate_stage3.py --split test --mode model --checkpoint checkpoints/denoiser_synth_best.pt --risk_sample

# MIT-BIH/NSTDB cross-dataset transfer comparator evaluation
python evaluate_stage3.py --split test --mode ecg_denoiser --checkpoint checkpoints/mitbih_em_denoiser_best.pt --risk_sample

# Paired participant-cluster bootstrap statistics (10,000 iterations, seed 123)
python evaluate_stage3.py --split test --paired_stats --checkpoint checkpoints/denoiser_synth_best.pt --bootstrap_iters 10000 --bootstrap_seed 123
python evaluate_stage3.py --split test --paired_stats_transfer --checkpoint checkpoints/mitbih_em_denoiser_best.pt --bootstrap_iters 10000 --bootstrap_seed 123

# Gating decomposition analyses (Table 6)
python evaluate_stage3.py --split test --gated_baselines --checkpoint checkpoints/mitbih_em_denoiser_best.pt --bootstrap_iters 10000 --bootstrap_seed 123
python evaluate_stage3.py --split test --synthetic_gated_baseline --synth_checkpoint checkpoints/denoiser_synth_best.pt --bootstrap_iters 10000 --bootstrap_seed 123
```

## 4. Ablations & Sensitivity Analyses

```bash
python evaluate_stage3.py --split test --ablation --risk_sample
python evaluate_stage3.py --split test --frequency_ablation --risk_sample
python evaluate_stage3.py --split test --gate_sensitivity --risk_sample
python audit_qrs_template_counts.py --split test --risk_sample
```

Key outputs:
- `logs/stage3_ablation_test.json` / `.csv`
- `logs/stage3_frequency_ablation_test.json` / `.csv`
- `logs/stage3_gate_sensitivity_test.json` / `.csv`
- `logs/qrs_template_audit_test_risk.json`
