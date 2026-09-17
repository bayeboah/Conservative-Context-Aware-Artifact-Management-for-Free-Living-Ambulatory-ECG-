"""
recalibrate_emg.py  (v2 — patient-specific thresholds)
-------------------------------------------------------
Computes a per-patient EMG threshold rather than a single global value.

WHY PATIENT-SPECIFIC THRESHOLDS
--------------------------------
The EMG ratio is:
    r = Σ_{f∈[20,40Hz]} P(f) / Σ_{f∈[5,15Hz]} P(f)

The denominator (QRS power in 5-15 Hz) varies dramatically across patients:
  - Low-voltage ECG (small BMI, thick subcutaneous fat) → small denominator
    → ratio inflated → patient flagged as "high EMG" even at rest
  - High-voltage ECG (lean build, thin chest wall) → large denominator
    → ratio deflated → real MA windows may go unflagged

A global threshold set at the 90th percentile of a mixed-patient distribution
applies the wrong standard to each individual. Patient-specific thresholds
calibrate on each patient's own resting windows, giving fair comparison.

METHOD
------
For each training patient:
  1. Select lying/sitting/sleeping windows with MAI < 0.10g
     (cleanest, minimal MA — these define that patient's baseline ratio)
  2. Compute 90th percentile of emg_ratio across those windows
  3. Store as that patient's threshold in config.json

At inference (update_emg_flags.py):
  - Windows from patient P get flagged using threshold[P]
  - Val/test patients: use mean of training thresholds as fallback

Usage:
    python recalibrate_emg.py
"""

import json
import numpy as np
import h5py
from pathlib import Path
from collections import defaultdict

try:
    from config import OUTPUT_ROOT
    CFG_PATH = Path(__file__).parent / 'config.json'
except ImportError:
    OUTPUT_ROOT = Path(r"C:\Users\sam\Desktop\ECG_Denoising\output")
    CFG_PATH    = Path(r"C:\Users\sam\Desktop\ECG_Denoising\config.json")

TRAIN_H5         = OUTPUT_ROOT / "train.h5"
EMG_CALIB_PCT    = 90.0
CLEAN_ACTIVITIES = {1, 2, 7}    # Lying, Sitting/Standing, Sleeping
CLEAN_MAI_MAX    = 0.10         # g

print("="*60)
print("  EMG Threshold Recalibration — Per-Patient")
print("="*60)
print()
print(f"  Source    : {TRAIN_H5}")
print(f"  Method    : {EMG_CALIB_PCT:.0f}th percentile per patient")
print(f"  Clean     : ActivityClass ∈ {{1,2,7}} AND MAI < {CLEAN_MAI_MAX}g")
print()

with h5py.File(TRAIN_H5, 'r') as f:
    n          = int(f.attrs['n_windows'])
    emg_ratio  = f['emg_ratio'][:]
    activity   = f['activity_class'][:]
    mao        = f['movement_accel'][:]
    loss_w     = f['loss_weight'][:]
    pids_raw   = [f['patient_id'][i].decode() for i in range(n)]

pids = np.array(pids_raw)
patients = sorted(set(pids))

print(f"  Training patients: {len(patients)}")
print()

# ── Provenance guard ──────────────────────────────────────────────────────────
# Per-patient thresholds must be calibrated ONLY on training patients. If train.h5
# still carries any patient that the current split assigns to val/test, calibrating
# here would derive a held-out patient's threshold from their own data — exactly the
# leak that occurred when this script was not re-run after a split change. Fail loudly.
_MANIFEST = OUTPUT_ROOT / "context_stratified_split.json"
if _MANIFEST.exists():
    with open(_MANIFEST) as _mf:
        _splits = json.load(_mf).get("splits", {})
    _heldout = set(_splits.get("val", [])) | set(_splits.get("test", []))
    _leaked = sorted(set(patients) & _heldout)
    if _leaked:
        raise SystemExit(
            "ABORT: EMG calibration would leak held-out data.\n"
            f"  train.h5 contains patients the current split assigns to val/test\n"
            f"  ({_MANIFEST.name}): {_leaked}\n"
            "  Rebuild train.h5 from the current split before recalibrating."
        )
    print(f"  Provenance guard OK: {len(patients)} calibration patients, "
          f"none in val/test.")
else:
    print(f"  WARNING: split manifest not found at {_MANIFEST};\n"
          "  cannot verify calibration provenance — proceed with caution.")
print()

per_patient = {}
fallback_thresholds = []

print(f"  {'Patient':<12}  {'Clean wins':>10}  {'p50':>8}  {'p90 (thresh)':>12}")
print(f"  {'─'*12}  {'─'*10}  {'─'*8}  {'─'*12}")

for pid in patients:
    pat_mask = pids == pid
    clean_mask = (
        pat_mask &
        np.isin(activity, list(CLEAN_ACTIVITIES)) &
        (mao < CLEAN_MAI_MAX) &
        (loss_w > 0.1)
    )
    n_clean = int(clean_mask.sum())

    if n_clean < 20:
        # Too few clean windows — use global 90th pct of all this patient's windows
        fallback = float(np.percentile(emg_ratio[pat_mask], EMG_CALIB_PCT))
        per_patient[pid] = fallback
        print(f"  {pid:<12}  {n_clean:>10}  {'n/a':>8}  {fallback:>12.6f}  ← fallback")
    else:
        ratios   = emg_ratio[clean_mask]
        p50      = float(np.percentile(ratios, 50))
        thresh   = float(np.percentile(ratios, EMG_CALIB_PCT))
        per_patient[pid] = thresh
        fallback_thresholds.append(thresh)
        print(f"  {pid:<12}  {n_clean:>10}  {p50:>8.4f}  {thresh:>12.6f}")

# Global fallback for val/test patients
global_fallback = float(np.mean(fallback_thresholds)) if fallback_thresholds else 0.394
print()
print(f"  Global fallback (val/test patients): {global_fallback:.6f}")
print(f"  (mean of training patient thresholds)")
print()

# Write to config.json
with open(CFG_PATH) as f:
    cfg = json.load(f)

cfg['emg_threshold']            = global_fallback          # kept for backward compat
cfg['emg_threshold_per_patient'] = per_patient             # new per-patient dict
cfg['emg_threshold_fallback']   = global_fallback
# Provenance stamp: records exactly which patients this calibration was derived
# from, so update_emg_flags.py can verify the config is not stale before applying.
cfg['emg_threshold_calibrated_on'] = sorted(patients)

with open(CFG_PATH, 'w') as f:
    json.dump(cfg, f, indent=4)

print(f"  Written to {CFG_PATH}:")
print(f"    emg_threshold              : {global_fallback:.6f}")
print(f"    emg_threshold_per_patient  : dict of {len(per_patient)} patients")
print()
print("  Next: run update_emg_flags.py to patch HDF5 files")
