"""
config.py
---------
Central configuration loader for the CACHET-CADB denoising pipeline.

Reads config.json from the same directory as this file and exposes
all settings as typed module-level constants. Every other script in
the pipeline imports from here — paths and parameters are never
hardcoded or passed as command-line arguments elsewhere.

Usage in any other script:
    from config import DATA_ROOT, EMG_THRESHOLD, ECG_FS, ...

To change a setting: edit config.json only. No script needs to change.

emg_threshold note:
    Set to null in config.json until you have run calibration.py.
    calibration.py writes the calibrated value back into config.json
    automatically. Importing EMG_THRESHOLD before calibration raises
    a clear error rather than silently using a wrong value.
"""

import json
from pathlib import Path
from typing import Tuple


# ── Load raw JSON ─────────────────────────────────────────────────────────────
_CFG_PATH = Path(__file__).parent / "config.json"

if not _CFG_PATH.exists():
    raise FileNotFoundError(
        f"config.json not found at {_CFG_PATH}.\n"
        "Copy config.json to the same directory as config.py."
    )

with open(_CFG_PATH) as _f:
    _cfg = json.load(_f)


# ── Helper ────────────────────────────────────────────────────────────────────
def _require(key: str):
    """Return config value, raising clearly if it is null or missing."""
    val = _cfg.get(key)
    if val is None:
        raise ValueError(
            f"config.json: '{key}' is null or missing.\n"
            f"  → Run calibration.py first to populate this value, "
            f"or set it manually in config.json."
        )
    return val


# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_ROOT      = Path(_require("data_root"))
OUTPUT_ROOT    = Path(_require("output_root"))
CHECKPOINT_DIR = Path(_require("checkpoint_dir"))
LOG_DIR        = Path(_require("log_dir"))


# ── EMG threshold ─────────────────────────────────────────────────────────────
# Returns None if not yet calibrated — callers that need the value
# (dataset.py, calibration.py) call _require() themselves or use
# get_emg_threshold() which gives a clear error message.
_EMG_THRESHOLD_RAW = _cfg.get("emg_threshold")


def get_emg_threshold() -> float:
    """
    Return the calibrated EMG threshold.
    Raises ValueError if calibration.py has not been run yet.
    """
    if _EMG_THRESHOLD_RAW is None:
        raise ValueError(
            "EMG threshold has not been calibrated yet.\n"
            "  → Run:  python calibration.py\n"
            "  → This writes the threshold into config.json automatically."
        )
    return float(_EMG_THRESHOLD_RAW)


EMG_THRESHOLD: float = _EMG_THRESHOLD_RAW   # None until calibrated


# ── Signal parameters ─────────────────────────────────────────────────────────
ECG_FS:          int   = int(_require("ecg_fs"))
IMU_FS_RAW:      int   = int(_require("imu_fs_raw"))
WINDOW_SEC:      int   = int(_require("window_sec"))
RANDOM_SEED:     int   = int(_require("random_seed"))

WINDOW_SAMPLES:  int   = ECG_FS * WINDOW_SEC          # 10240
IMU_SAMPLES_RAW: int   = IMU_FS_RAW * WINDOW_SEC      # 640


# ── Preprocessing parameters ──────────────────────────────────────────────────
BANDPASS_LOW:    float = float(_require("bandpass_low"))
BANDPASS_HIGH:   float = float(_require("bandpass_high"))
BANDPASS_ORDER:  int   = int(_require("bandpass_order"))

HR_AGREE_TOL:    float = float(_require("hr_agree_tol_bpm"))
RR_MIN_MS:       float = float(_require("rr_min_ms"))
RR_MAX_MS:       float = float(_require("rr_max_ms"))


# ── EMG detection parameters ──────────────────────────────────────────────────
EMG_QRS_BAND:       Tuple[float, float] = tuple(_require("emg_qrs_band"))
EMG_EMG_BAND:       Tuple[float, float] = tuple(_require("emg_emg_band"))
EMG_CALIBRATION_PCT: float              = float(_require("emg_calibration_pct"))


# ── Loss weights ──────────────────────────────────────────────────────────────
LOSS_WEIGHT_CLEAN: float = float(_require("loss_weight_clean"))
LOSS_WEIGHT_EMG:   float = float(_require("loss_weight_emg"))
LOSS_WEIGHT_NOISE: float = float(_require("loss_weight_noise"))


# ── Training hyperparameters ──────────────────────────────────────────────────
BATCH_SIZE:          int   = int(_require("batch_size"))
NUM_WORKERS:         int   = int(_require("num_workers"))
LEARNING_RATE:       float = float(_require("learning_rate"))
MAX_EPOCHS:          int   = int(_require("max_epochs"))
EARLY_STOP_PATIENCE: int   = int(_require("early_stop_patience"))

STAGE2_LAMBDA_AUX:       float = float(_require("stage2_lambda_aux"))
STAGE2_LAMBDA_CONT:      float = float(_require("stage2_lambda_cont"))
STAGE3_MAX_RESIDUAL:     float = float(_require("stage3_max_residual"))
STAGE3_LAMBDA_IDENTITY:  float = float(_require("stage3_lambda_identity"))
STAGE3_LAMBDA_RESIDUAL:  float = float(_require("stage3_lambda_residual"))
STAGE3_LAMBDA_SMOOTH:    float = float(_require("stage3_lambda_smooth"))
STAGE3_LAMBDA_QRS:       float = float(_require("stage3_lambda_qrs"))
STAGE3_LAMBDA_MOTION:    float = float(_require("stage3_lambda_motion"))


# ── Directory creation helper ─────────────────────────────────────────────────
def ensure_output_dirs() -> None:
    """
    Create output, checkpoint, and log directories if they do not exist.
    Call once at the start of any training or evaluation script.
    """
    for d in [OUTPUT_ROOT, CHECKPOINT_DIR, LOG_DIR]:
        d.mkdir(parents=True, exist_ok=True)


# ── Config update helper (used by calibration.py) ────────────────────────────
def write_emg_threshold(threshold: float) -> None:
    """
    Write the calibrated EMG threshold back into config.json.
    Called automatically by calibration.py after calibration completes.
    """
    with open(_CFG_PATH) as f:
        current = json.load(f)
    current["emg_threshold"] = round(float(threshold), 8)
    with open(_CFG_PATH, "w") as f:
        json.dump(current, f, indent=4)
    print(f"config.json updated: emg_threshold = {threshold:.8f}")


# ── Self-check ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("── config.py self-check ─────────────────────────────")
    print(f"  DATA_ROOT      : {DATA_ROOT}")
    print(f"  OUTPUT_ROOT    : {OUTPUT_ROOT}")
    print(f"  CHECKPOINT_DIR : {CHECKPOINT_DIR}")
    print(f"  LOG_DIR        : {LOG_DIR}")
    print()
    print(f"  ECG_FS         : {ECG_FS} Hz")
    print(f"  IMU_FS_RAW     : {IMU_FS_RAW} Hz")
    print(f"  WINDOW_SEC     : {WINDOW_SEC} s")
    print(f"  WINDOW_SAMPLES : {WINDOW_SAMPLES}")
    print(f"  RANDOM_SEED    : {RANDOM_SEED}")
    print()
    print(f"  BANDPASS       : {BANDPASS_LOW}–{BANDPASS_HIGH} Hz  order={BANDPASS_ORDER}")
    print(f"  HR_AGREE_TOL   : ±{HR_AGREE_TOL} bpm")
    print(f"  RR bounds      : {RR_MIN_MS}–{RR_MAX_MS} ms")
    print()
    print(f"  EMG_QRS_BAND   : {EMG_QRS_BAND} Hz")
    print(f"  EMG_EMG_BAND   : {EMG_EMG_BAND} Hz")
    print(f"  EMG_CALIB_PCT  : {EMG_CALIBRATION_PCT}th percentile")
    print()
    print(f"  EMG_THRESHOLD  : {EMG_THRESHOLD}  "
          f"{'(not yet calibrated — run calibration.py)' if EMG_THRESHOLD is None else '(calibrated)'}")
    print()
    print(f"  LOSS weights   : clean={LOSS_WEIGHT_CLEAN}  "
          f"emg={LOSS_WEIGHT_EMG}  noise={LOSS_WEIGHT_NOISE}")
    print()
    print(f"  BATCH_SIZE     : {BATCH_SIZE}")
    print(f"  LEARNING_RATE  : {LEARNING_RATE}")
    print(f"  MAX_EPOCHS     : {MAX_EPOCHS}")
    print(f"  EARLY_STOP     : {EARLY_STOP_PATIENCE} epochs")
    print()
    print(f"  STAGE2_LAMBDA_AUX      : {STAGE2_LAMBDA_AUX}")
    print(f"  STAGE2_LAMBDA_CONT     : {STAGE2_LAMBDA_CONT}")
    print(f"  STAGE3_MAX_RESIDUAL    : {STAGE3_MAX_RESIDUAL}")
    print(f"  STAGE3_LAMBDA_IDENTITY : {STAGE3_LAMBDA_IDENTITY}")
    print(f"  STAGE3_LAMBDA_RESIDUAL : {STAGE3_LAMBDA_RESIDUAL}")
    print(f"  STAGE3_LAMBDA_SMOOTH   : {STAGE3_LAMBDA_SMOOTH}")
    print(f"  STAGE3_LAMBDA_QRS      : {STAGE3_LAMBDA_QRS}")
    print(f"  STAGE3_LAMBDA_MOTION   : {STAGE3_LAMBDA_MOTION}")
    print()
    print("All settings loaded successfully.")
