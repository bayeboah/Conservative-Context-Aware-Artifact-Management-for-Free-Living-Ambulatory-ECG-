"""
artifact_profile.py
-------------------
Profile CACHET-CADB artifact risk from the preprocessed HDF5 splits.

This script combines publication-level priors from Table 8 with the per-window
context already extracted from annotation.csv/context.xlsx:

  - movement acceleration intensity (MAI)
  - activity and body position
  - noise annotations
  - R-peak reliability
  - EMG flags / EMG ratio
  - activity/body-position transitions

Outputs:
  logs/artifact_profile_patient_day.csv
  logs/artifact_profile_patient.csv
  logs/artifact_profile_summary.json

Usage:
  python artifact_profile.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import h5py
import numpy as np
import pandas as pd

from config import LOG_DIR, OUTPUT_ROOT, ensure_output_dirs
from split_config import PATIENT_REGISTRY


PAPER_NOISY_SIGNAL_PCT = {
    "P1": 7.48,
    "P2": 6.47,
    "P3": 8.40,
    "P4": 5.80,
    "P5": 16.27,
    "P6": 3.26,
    "P7": 12.43,
    "P8": 9.90,
    "P9": 29.61,
    "P10": 31.72,
    "P11": 12.72,
    "P12": 6.88,
    "P13": 2.65,
    "P14": 29.90,
    "P15": 28.05,
    "P16": 6.29,
    "P17": 6.02,
    "P18": 3.23,
    "P19": 3.59,
    "P21": 10.34,
    "P23": 7.24,
    "PNSR-1": 0.49,
    "PNSR-3": 27.14,
    "PNSR-4": 5.63,
}

ACTIVITY_LABELS = {
    0: "Unknown",
    1: "Lying",
    2: "Sitting/Standing",
    3: "Walking",
    4: "Running",
    5: "Cycling",
    6: "Driving",
    7: "Sleeping",
}

HIGH_MOTION_ACTIVITIES = {3, 4, 5}


def _decode_patient_ids(arr: np.ndarray) -> np.ndarray:
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in arr])


def _pct(mask: np.ndarray) -> float:
    if len(mask) == 0:
        return 0.0
    return 100.0 * float(np.mean(mask))


def _mode_label(values: np.ndarray, labels: Dict[int, str]) -> str:
    if len(values) == 0:
        return "Unknown"
    vals, counts = np.unique(values.astype(np.int64), return_counts=True)
    code = int(vals[int(np.argmax(counts))])
    return labels.get(code, str(code))


def _transition_pct(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    return 100.0 * float(np.mean(np.diff(values.astype(np.int64)) != 0))


def _mai_jump_pct(mai: np.ndarray, threshold: float) -> float:
    if len(mai) < 2:
        return 0.0
    return 100.0 * float(np.mean(np.abs(np.diff(mai.astype(np.float32))) >= threshold))


def _risk_score(row: Dict[str, float]) -> float:
    """Publication-informed, interpretable score for ranking risky days."""
    paper = row["paper_noisy_signal_pct"] / 100.0
    return float(
        0.30 * row["high_mai_pct"] / 100.0
        + 0.15 * row["very_high_mai_pct"] / 100.0
        + 0.15 * row["activity_transition_pct"] / 100.0
        + 0.10 * row["body_transition_pct"] / 100.0
        + 0.10 * row["mai_jump_pct"] / 100.0
        + 0.10 * row["emg_flag_pct"] / 100.0
        + 0.05 * row["noise_annotated_pct"] / 100.0
        + 0.10 * row["rpeak_unreliable_pct"] / 100.0
        + 0.15 * paper
    )


def _summarise_group(
    split: str,
    patient: str,
    day: int,
    idx: np.ndarray,
    arrays: Dict[str, np.ndarray],
) -> Dict[str, float | int | str]:
    activity = arrays["activity"][idx]
    body = arrays["body"][idx]
    mai = arrays["mai"][idx]
    emg_flag = arrays["emg_flag"][idx].astype(bool)
    emg_ratio = arrays["emg_ratio"][idx]
    noise = arrays["noise"][idx].astype(bool)
    reliable = arrays["reliable"][idx].astype(bool)

    row: Dict[str, float | int | str] = {
        "split": split,
        "patient": patient,
        "day": int(day),
        "site": PATIENT_REGISTRY.get(patient, {}).get("site", ""),
        "has_af_history": bool(PATIENT_REGISTRY.get(patient, {}).get("has_af", False)),
        "has_af_history_inferred": bool(PATIENT_REGISTRY.get(patient, {}).get("has_af", False)),
        "has_af_history_label_status": "inferred_not_source_verified",
        "nsr_only": bool(PATIENT_REGISTRY.get(patient, {}).get("nsr_only", False)),
        "n_windows": int(len(idx)),
        "hours": float(len(idx) * 10.0 / 3600.0),
        "paper_noisy_signal_pct": float(PAPER_NOISY_SIGNAL_PCT.get(patient, np.nan)),
        "mai_mean": float(np.mean(mai)) if len(mai) else 0.0,
        "mai_p90": float(np.percentile(mai, 90)) if len(mai) else 0.0,
        "mai_p95": float(np.percentile(mai, 95)) if len(mai) else 0.0,
        "high_mai_pct": _pct(mai >= 0.08),
        "very_high_mai_pct": _pct(mai >= 0.15),
        "high_motion_activity_pct": _pct(np.isin(activity, list(HIGH_MOTION_ACTIVITIES))),
        "dominant_activity": _mode_label(activity, ACTIVITY_LABELS),
        "activity_transition_pct": _transition_pct(activity),
        "body_transition_pct": _transition_pct(body),
        "mai_jump_pct": _mai_jump_pct(mai, threshold=0.05),
        "noise_annotated_pct": _pct(noise),
        "rpeak_reliable_pct": _pct(reliable),
        "rpeak_unreliable_pct": _pct(~reliable),
        "emg_flag_pct": _pct(emg_flag),
        "emg_ratio_mean": float(np.mean(emg_ratio)) if len(emg_ratio) else 0.0,
        "emg_ratio_p90": float(np.percentile(emg_ratio, 90)) if len(emg_ratio) else 0.0,
    }
    row["artifact_risk_score"] = _risk_score(row)  # type: ignore[arg-type]
    return row


def _load_split_metadata(h5_path: Path) -> Dict[str, np.ndarray]:
    with h5py.File(h5_path, "r") as f:
        return {
            "patient": _decode_patient_ids(f["patient_id"][:]),
            "day": f["day"][:].astype(np.int32),
            "window_idx": f["window_idx"][:].astype(np.int32),
            "activity": f["activity_class"][:].astype(np.int32),
            "body": f["body_position"][:].astype(np.int32),
            "mai": f["movement_accel"][:].astype(np.float32),
            "emg_flag": f["emg_flagged"][:].astype(np.int8),
            "emg_ratio": f["emg_ratio"][:].astype(np.float32),
            "noise": f["noise_annotated"][:].astype(np.int8),
            "reliable": f["rpeaks_reliable"][:].astype(np.int8),
        }


def collect_patient_day_profiles(splits: Iterable[str]) -> pd.DataFrame:
    rows: List[Dict[str, float | int | str]] = []
    for split in splits:
        h5_path = OUTPUT_ROOT / f"{split}.h5"
        if not h5_path.exists():
            raise FileNotFoundError(f"Missing {h5_path}; build dataset first.")

        arrays = _load_split_metadata(h5_path)
        order = np.lexsort((arrays["window_idx"], arrays["day"], arrays["patient"]))
        patient_sorted = arrays["patient"][order]
        day_sorted = arrays["day"][order]

        start = 0
        while start < len(order):
            patient = patient_sorted[start]
            day = int(day_sorted[start])
            end = start + 1
            while (
                end < len(order)
                and patient_sorted[end] == patient
                and int(day_sorted[end]) == day
            ):
                end += 1
            idx = order[start:end]
            rows.append(_summarise_group(split, patient, day, idx, arrays))
            start = end

    return pd.DataFrame(rows).sort_values(
        ["artifact_risk_score", "high_mai_pct", "emg_ratio_p90"],
        ascending=[False, False, False],
    )


def collect_patient_profiles(day_df: pd.DataFrame) -> pd.DataFrame:
    weighted_rows = []
    for (split, patient), g in day_df.groupby(["split", "patient"], sort=False):
        weights = g["n_windows"].to_numpy(dtype=np.float64)
        weights = weights / max(weights.sum(), 1.0)
        row = {
            "split": split,
            "patient": patient,
            "site": g["site"].iloc[0],
            "has_af_history": bool(g["has_af_history"].iloc[0]),
            "has_af_history_inferred": bool(g["has_af_history_inferred"].iloc[0]),
            "has_af_history_label_status": str(g["has_af_history_label_status"].iloc[0]),
            "nsr_only": bool(g["nsr_only"].iloc[0]),
            "n_days": int(len(g)),
            "n_windows": int(g["n_windows"].sum()),
            "hours": float(g["hours"].sum()),
            "paper_noisy_signal_pct": float(g["paper_noisy_signal_pct"].iloc[0]),
        }
        for col in (
            "artifact_risk_score",
            "mai_mean",
            "mai_p90",
            "mai_p95",
            "high_mai_pct",
            "very_high_mai_pct",
            "high_motion_activity_pct",
            "activity_transition_pct",
            "body_transition_pct",
            "mai_jump_pct",
            "noise_annotated_pct",
            "rpeak_reliable_pct",
            "rpeak_unreliable_pct",
            "emg_flag_pct",
            "emg_ratio_mean",
            "emg_ratio_p90",
        ):
            row[col] = float(np.sum(g[col].to_numpy(dtype=np.float64) * weights))
        row["top_risk_day"] = int(g.sort_values("artifact_risk_score", ascending=False)["day"].iloc[0])
        weighted_rows.append(row)

    return pd.DataFrame(weighted_rows).sort_values(
        ["artifact_risk_score", "paper_noisy_signal_pct"],
        ascending=[False, False],
    )


def write_summary(day_df: pd.DataFrame, patient_df: pd.DataFrame, out_dir: Path) -> Path:
    summary = {
        "source": "CACHET-CADB context + Table 8 publication priors",
        "interpretation": {
            "high_mai_pct": "movement_accel >= 0.08g",
            "very_high_mai_pct": "movement_accel >= 0.15g",
            "mai_jump_pct": "|delta MAI between adjacent 10s windows| >= 0.05g",
            "artifact_risk_score": "weighted ranking score, not a clinical label",
        },
        "top_patient_days": day_df.head(20).to_dict(orient="records"),
        "top_patients": patient_df.head(12).to_dict(orient="records"),
        "recommended_stage3_strata": [
            "low risk: MAI < 0.02g, no noise annotation, R-peaks reliable",
            "high motion: MAI >= 0.08g",
            "very high motion: MAI >= 0.15g",
            "transition: activity/body changed or MAI jump >= 0.05g",
            "quality risk: noise_annotated, rpeak_unreliable, or high EMG ratio",
        ],
    }
    path = out_dir / "artifact_profile_summary.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile CACHET-CADB artifact risk.")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        choices=["train", "val", "test"],
        help="HDF5 splits to profile.",
    )
    parser.add_argument("--top", type=int, default=12, help="Rows to print.")
    args = parser.parse_args()

    ensure_output_dirs()
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    day_df = collect_patient_day_profiles(args.splits)
    patient_df = collect_patient_profiles(day_df)

    day_path = LOG_DIR / "artifact_profile_patient_day.csv"
    patient_path = LOG_DIR / "artifact_profile_patient.csv"
    day_df.to_csv(day_path, index=False)
    patient_df.to_csv(patient_path, index=False)
    summary_path = write_summary(day_df, patient_df, LOG_DIR)

    print("\nArtifact profile complete")
    print(f"  Patient-day CSV : {day_path}")
    print(f"  Patient CSV     : {patient_path}")
    print(f"  Summary JSON    : {summary_path}")

    cols = [
        "split",
        "patient",
        "day",
        "artifact_risk_score",
        "high_mai_pct",
        "very_high_mai_pct",
        "activity_transition_pct",
        "mai_jump_pct",
        "rpeak_unreliable_pct",
        "emg_flag_pct",
        "paper_noisy_signal_pct",
    ]
    print("\nTop risky patient-days")
    print(day_df[cols].head(args.top).to_string(index=False))

    pcols = [
        "split",
        "patient",
        "artifact_risk_score",
        "top_risk_day",
        "high_mai_pct",
        "rpeak_unreliable_pct",
        "emg_flag_pct",
        "paper_noisy_signal_pct",
    ]
    print("\nTop risky patients")
    print(patient_df[pcols].head(args.top).to_string(index=False))


if __name__ == "__main__":
    main()
