# ECG Denoising Pipeline Summary

## Current Objective

The project is now framed as **context-gated artifact management** for CACHET-CADB, not blind supervised ECG denoising.

CACHET-CADB is an ambulatory ECG dataset with rich context and annotation files, but it does not provide true clean/noisy ECG pairs. Because of that, the final Stage 3 method should not claim to reconstruct a clean ECG from an unknown noisy input. The defensible objective is:

```text
Preserve low-risk ECG windows unchanged.
Apply conservative artifact suppression only when context indicates risk.
Preserve QRS morphology.
Evaluate by artifact-risk strata, not only global SNR.
```

## Pipeline

```text
Raw CACHET-CADB signal + annotation/context folders
        |
        v
dataset.py
        |
        v
train.h5 / val.h5 / test.h5
        |
        v
artifact_profile.py
        |
        v
patient/day/window artifact-risk profile
        |
        v
Stage 3 official method: teacher_gated
        |
        v
risk-stratified evaluation
```

## What Each Stage Does

### Preprocessing

`dataset.py` reads the raw signal files plus each patient's annotation/context folders and stores the useful fields in HDF5:

- ECG
- accelerometer and gyroscope
- movement acceleration intensity / MAI
- activity class
- body position
- step count
- rhythm annotations
- noise annotations
- R-peak reliability
- EMG flag and EMG ratio

### Artifact Profiling

`artifact_profile.py` combines HDF5 context fields with publication-level priors from CACHET-CADB Table 8.

It produces:

- `logs/artifact_profile_patient_day.csv`
- `logs/artifact_profile_patient.csv`
- `logs/artifact_profile_summary.json`

The profile identifies artifact-risk regions using:

- high MAI
- very high MAI
- activity/body-position transitions
- sudden MAI jumps
- noise annotations
- R-peak unreliability
- EMG flag / EMG ratio
- publication high-noise patients

### Stage 3 Official Method

The official Stage 3 method is `teacher_gated`.

```python
pseudo = generate_pseudo_target(ecg)
gate = motion_gate(mai, noise)
output = (1 - gate) * ecg + gate * pseudo
```

Interpretation:

- low-risk gate near 0: preserve ECG
- high-risk gate near 1: move toward conservative pseudo-target
- pseudo-target only suppresses defensible artifact bands
- QRS morphology should remain almost unchanged

### Synthetic Stage 3

`train_stage3_synthetic.py` is exploratory.

It creates synthetic pairs:

```text
clean-ish ECG target + artifact residual from risky donor = synthetic noisy ECG
```

The synthetic model learned the synthetic validation task, but the real risk-sampled evaluation showed poor transfer:

- high-risk artifact bands increased on real windows
- QRS preservation was worse than `teacher_gated`

Therefore synthetic Stage 3 is an ablation result, not the official method.

## Final Evaluation Commands

Run the official validation and test evaluations:

```powershell
python artifact_profile.py --top 12
python evaluate_stage3.py --split val --mode teacher_gated --risk_sample
python evaluate_stage3.py --split test --mode teacher_gated --risk_sample
```

Run the ablation table:

```powershell
python evaluate_stage3.py --split test --ablation --risk_sample
```

Evaluate the exploratory synthetic model only as an ablation:

```powershell
python evaluate_stage3.py --split test --mode model --checkpoint checkpoints/denoiser_synth_best.pt --risk_sample
```

## Success Criteria

Primary success criteria:

- low-risk identity preserved: low-MAI residual RMS near 0
- high-MAI artifact reduction greater than low-MAI artifact reduction
- positive artifact reduction in risk strata
- QRS correlation above 0.99
- no claim of true clean ECG reconstruction

Secondary/informational:

- legacy global `delta_snr`
- QRS/motion SNR
- percentage of windows improved

## Current Conclusion

The conservative `teacher_gated` Stage 3 passes as the official method:

```text
low-risk ECG remains unchanged
high-risk windows show target artifact-band reduction
QRS morphology is preserved
```

The learned synthetic model is retained as an ablation demonstrating that low synthetic validation loss does not guarantee safe transfer to real ambulatory ECG.
