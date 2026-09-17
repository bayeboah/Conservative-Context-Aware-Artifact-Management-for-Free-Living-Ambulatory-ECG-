# Final Pipeline Commands

Run these from:

```powershell
cd C:\Users\sam\OneDrive\Desktop\ECG_Denoising
```

## 1. Verify Dataset

```powershell
python verify_dataset.py
```

## 2. Rebuild Artifact Profile

```powershell
python artifact_profile.py --top 12
```

Important outputs:

```text
logs/artifact_profile_patient.csv
logs/artifact_profile_patient_day.csv
logs/artifact_profile_summary.json
```

## 3. Official Stage 3 Evaluation

```powershell
python evaluate_stage3.py --split val --mode teacher_gated --risk_sample
python evaluate_stage3.py --split test --mode teacher_gated --risk_sample
```

Important outputs:

```text
logs/stage3_eval_val_teacher_gated.json
logs/stage3_eval_test_teacher_gated.json
logs/stage3_metrics_val_teacher_gated.png
logs/stage3_metrics_test_teacher_gated.png
logs/stage3_before_after_val_teacher_gated.png
logs/stage3_before_after_test_teacher_gated.png
```

## 4. Ablation

```powershell
python evaluate_stage3.py --split test --ablation --risk_sample
```

Important outputs:

```text
logs/stage3_ablation_test.json
logs/stage3_ablation_test.csv
```

## 5. Exploratory Synthetic Model

Only run this as an ablation, not as the official Stage 3 method.

```powershell
python evaluate_stage3.py --split test --mode model --checkpoint checkpoints/denoiser_synth_best.pt --risk_sample
```

## Official Interpretation

Use `teacher_gated` as the final Stage 3 result.

Report success using:

```text
low-risk identity preservation
high-risk artifact-band reduction
context-selective reduction
QRS correlation / morphology preservation
```

Treat legacy global `delta_snr` as informational only.
