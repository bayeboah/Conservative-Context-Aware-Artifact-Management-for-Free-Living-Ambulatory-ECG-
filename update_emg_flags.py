"""
update_emg_flags.py  (v2 — per-patient thresholds)
---------------------------------------------------
Patches emg_flagged and loss_weight in all HDF5 files using
per-patient thresholds from config.json.

Val/test patients not in the training threshold dict use the
global fallback (mean of training patient thresholds).

Usage:
    python recalibrate_emg.py   # first
    python update_emg_flags.py  # then
"""

import json
import numpy as np
import h5py
from pathlib import Path

try:
    from config import OUTPUT_ROOT
    CFG_PATH = Path(__file__).parent / 'config.json'
    LOSS_WEIGHT_CLEAN = 1.0
    LOSS_WEIGHT_EMG   = 0.4
    LOSS_WEIGHT_NOISE = 0.1
except ImportError:
    OUTPUT_ROOT       = Path(r"C:\Users\sam\Desktop\ECG_Denoising\output")
    CFG_PATH          = Path(r"C:\Users\sam\Desktop\ECG_Denoising\config.json")
    LOSS_WEIGHT_CLEAN = 1.0
    LOSS_WEIGHT_EMG   = 0.4
    LOSS_WEIGHT_NOISE = 0.1

HIGH_EMG_RISK = {3, 4, 5}   # Walking, Running, Cycling

with open(CFG_PATH) as f:
    cfg = json.load(f)

per_patient = cfg.get('emg_threshold_per_patient', {})
fallback    = cfg.get('emg_threshold_fallback',
                      cfg.get('emg_threshold', 0.394))

print("="*60)
print("  Update EMG Flags — Per-Patient Thresholds")
print("="*60)
print(f"  Per-patient thresholds : {len(per_patient)} patients in config")
print(f"  Fallback threshold     : {fallback:.6f}")
print()

# ── Provenance guard ──────────────────────────────────────────────────────────
# This is where a stale calibration reaches the data: any val/test patient that
# still has a personal threshold in per_patient would be flagged using a threshold
# derived from their own held-out recordings. Refuse to apply such a config.
_MANIFEST = OUTPUT_ROOT / "context_stratified_split.json"
if _MANIFEST.exists():
    with open(_MANIFEST) as _mf:
        _splits = json.load(_mf).get("splits", {})
    _heldout = set(_splits.get("val", [])) | set(_splits.get("test", []))
    _leaked = sorted(set(per_patient) & _heldout)
    if _leaked:
        raise SystemExit(
            "ABORT: config carries per-patient EMG thresholds for held-out patients.\n"
            f"  Leaked (val/test with a personal threshold): {_leaked}\n"
            "  The calibration is stale relative to the current split.\n"
            "  Re-run recalibrate_emg.py before applying flags."
        )
    _calib = cfg.get('emg_threshold_calibrated_on')
    if _calib is not None and set(_calib) != set(_splits.get("train", [])):
        print("  WARNING: calibration patient set does not match the manifest "
              "train set;\n  thresholds may be stale. Consider re-running "
              "recalibrate_emg.py.")
    else:
        print(f"  Provenance guard OK: no held-out patient has a personal threshold.")
else:
    print(f"  WARNING: split manifest not found at {_MANIFEST};\n"
          "  cannot verify threshold provenance — proceed with caution.")
print()

for split in ['train', 'val', 'test']:
    h5_path = OUTPUT_ROOT / f'{split}.h5'
    if not h5_path.exists():
        print(f"  {split}.h5 not found — skipping")
        continue

    print(f"  Patching {split}.h5...")
    with h5py.File(h5_path, 'r+') as f:
        n          = int(f.attrs['n_windows'])
        emg_ratio  = f['emg_ratio'][:]
        activity   = f['activity_class'][:]
        noise_ann  = f['noise_annotated'][:]
        pids       = np.array([f['patient_id'][i].decode() for i in range(n)])

        # Build per-window threshold array
        thresholds = np.array([
            per_patient.get(p, fallback) for p in pids
        ], dtype=np.float32)

        # Recompute emg_flagged using per-patient threshold
        new_flagged = (
            (emg_ratio > thresholds) &
            np.isin(activity, list(HIGH_EMG_RISK))
        ).astype(np.int8)

        # Recompute loss_weight
        new_loss = np.where(
            noise_ann.astype(bool),
            LOSS_WEIGHT_NOISE,
            np.where(new_flagged.astype(bool), LOSS_WEIGHT_EMG, LOSS_WEIGHT_CLEAN)
        ).astype(np.float32)

        f['emg_flagged'][:]  = new_flagged
        f['loss_weight'][:]  = new_loss
        f.attrs['emg_thresh'] = fallback   # global reference value

        n_flagged = int(new_flagged.sum())
        flag_pct  = 100 * n_flagged / n

        # Per-patient flag summary
        unique_pats = sorted(set(pids))
        print(f"    windows       : {n:,}")
        print(f"    emg_flagged   : {n_flagged:,} ({flag_pct:.1f}%)")
        print(f"    {'Patient':<12}  {'Threshold':>10}  {'Flagged':>8}  {'%':>6}")
        for pid in unique_pats:
            pmask   = pids == pid
            thresh  = per_patient.get(pid, fallback)
            pflag   = int(new_flagged[pmask].sum())
            ppct    = 100 * pflag / pmask.sum() if pmask.sum() > 0 else 0
            print(f"    {pid:<12}  {thresh:>10.6f}  {pflag:>8,}  {ppct:>5.1f}%")
        print()

print("Done. Run verify_dataset.py to confirm new flag rates.")
