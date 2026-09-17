"""
Audit QRS-template availability for Stage 3 evaluation samples.

This script mirrors the input-side part of evaluate_stage3.qrs_template_corr:
R-peaks are detected on the input ECG, peaks too close to the window boundary
are removed, and beat templates with near-zero variance are considered invalid.

It is intended to explain why QRS-correlation rows can have a smaller n than
the full Stage 3 risk-sampled evaluation set.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np

from config import LOG_DIR, OUTPUT_ROOT
from dataset import load_norm_stats
from evaluate_stage3 import (
    ECG_FS,
    build_risk_sample_indices,
    chunk_loader,
    detect_rpeaks,
)


def _cause_for_input_template(sig: np.ndarray, half_win_ms: int) -> tuple[str, int, int]:
    hw = int(half_win_ms / 1000 * ECG_FS)
    peaks_raw = detect_rpeaks(sig, ECG_FS)
    if len(peaks_raw) == 0:
        return "no_detected_rpeak", 0, 0

    peaks_in_bounds = peaks_raw[
        (peaks_raw >= hw) & (peaks_raw < len(sig) - hw)
    ]
    if len(peaks_in_bounds) == 0:
        return "no_in_bounds_rpeak", int(len(peaks_raw)), 0

    valid_templates = 0
    for p in peaks_in_bounds:
        template = sig[p - hw : p + hw]
        if template.std() >= 1e-6:
            valid_templates += 1

    if valid_templates == 0:
        return "degenerate_input_template", int(len(peaks_raw)), int(len(peaks_in_bounds))

    return "valid", int(len(peaks_raw)), int(len(peaks_in_bounds))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--risk_sample", action="store_true")
    parser.add_argument("--n_chunks", type=int, default=100)
    parser.add_argument("--half_win_ms", type=int, default=150)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    h5_path = OUTPUT_ROOT / f"{args.split}.h5"
    norm = load_norm_stats(OUTPUT_ROOT / "norm_stats.json")
    n_requested = args.n_chunks * 256 if args.n_chunks is not None else None

    indices = None
    if args.risk_sample:
        with h5py.File(h5_path, "r") as f:
            n_total = int(f.attrs["n_windows"])
        n_requested = n_total if n_requested is None else min(n_requested, n_total)
        indices = build_risk_sample_indices(h5_path, n_requested, seed=args.seed)

    counts = Counter()
    by_emg = defaultdict(Counter)
    by_patient = defaultdict(Counter)
    by_rpeak_reliable = defaultdict(Counter)
    dropped_examples = []
    total = 0
    raw_peak_sum = 0
    in_bound_peak_sum = 0

    for batch in chunk_loader(
        h5_path,
        norm,
        chunk=256,
        n_chunks=args.n_chunks,
        indices=indices,
    ):
        ecg = batch["ecg"].numpy()[:, 0, :]
        emg = batch["emg_flagged"].numpy().astype(bool)
        rel = batch["rpeaks_reliable"].numpy().astype(bool)
        patients = batch["patient_id"]
        days = batch["day"].numpy()
        win_idx = batch["window_idx"].numpy()

        for i, sig in enumerate(ecg):
            cause, n_raw, n_bound = _cause_for_input_template(sig, args.half_win_ms)
            counts[cause] += 1
            by_emg["emg_flagged" if emg[i] else "not_emg"][cause] += 1
            by_patient[str(patients[i])][cause] += 1
            by_rpeak_reliable[
                "rpeak_reliable" if rel[i] else "rpeak_unreliable"
            ][cause] += 1
            total += 1
            raw_peak_sum += n_raw
            in_bound_peak_sum += n_bound
            if cause != "valid" and len(dropped_examples) < 50:
                dropped_examples.append(
                    {
                        "patient_id": str(patients[i]),
                        "day": int(days[i]),
                        "window_idx": int(win_idx[i]),
                        "emg_flagged": bool(emg[i]),
                        "rpeaks_reliable": bool(rel[i]),
                        "cause": cause,
                        "n_raw_peaks": int(n_raw),
                        "n_in_bounds_peaks": int(n_bound),
                    }
                )

    valid = counts["valid"]
    dropped = total - valid
    emg_total = sum(by_emg["emg_flagged"].values())
    emg_valid = by_emg["emg_flagged"]["valid"]

    payload = {
        "split": args.split,
        "risk_sample": bool(args.risk_sample),
        "risk_sample_seed": int(args.seed) if args.risk_sample else None,
        "n_chunks": args.n_chunks,
        "n_windows": int(total),
        "half_win_ms": int(args.half_win_ms),
        "valid_qrs_template_windows": int(valid),
        "dropped_qrs_template_windows": int(dropped),
        "emg_windows": int(emg_total),
        "emg_valid_qrs_template_windows": int(emg_valid),
        "emg_dropped_qrs_template_windows": int(emg_total - emg_valid),
        "counts_by_cause": dict(counts),
        "counts_by_emg": {k: dict(v) for k, v in by_emg.items()},
        "counts_by_rpeak_reliable": {k: dict(v) for k, v in by_rpeak_reliable.items()},
        "mean_raw_peaks_per_window": float(raw_peak_sum / max(total, 1)),
        "mean_in_bounds_peaks_per_window": float(in_bound_peak_sum / max(total, 1)),
        "dropped_examples_first_50": dropped_examples,
    }

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"{args.split}_{'risk' if args.risk_sample else 'sequential'}"
    json_path = LOG_DIR / f"qrs_template_audit_{suffix}.json"
    csv_path = LOG_DIR / f"qrs_template_audit_{suffix}_by_patient.csv"

    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    causes = sorted(counts.keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["patient_id", *causes, "total", "valid_pct"])
        for patient, c in sorted(by_patient.items()):
            p_total = sum(c.values())
            p_valid = c["valid"]
            writer.writerow(
                [
                    patient,
                    *[int(c[cause]) for cause in causes],
                    int(p_total),
                    float(100.0 * p_valid / max(p_total, 1)),
                ]
            )

    print("\nQRS-template availability audit")
    print("=" * 40)
    print(f"split                 : {args.split}")
    print(f"risk_sample           : {args.risk_sample}")
    print(f"windows               : {total:,}")
    print(f"valid                 : {valid:,}")
    print(f"dropped               : {dropped:,}")
    print(f"EMG windows           : {emg_total:,}")
    print(f"EMG valid             : {emg_valid:,}")
    print(f"EMG dropped           : {emg_total - emg_valid:,}")
    print("\nBy cause:")
    for cause, n in counts.most_common():
        print(f"  {cause:<28} {n:>8,}")
    print(f"\nJSON -> {json_path}")
    print(f"CSV  -> {csv_path}")


if __name__ == "__main__":
    main()
