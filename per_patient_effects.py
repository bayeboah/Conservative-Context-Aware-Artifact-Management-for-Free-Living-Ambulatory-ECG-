#!/usr/bin/env python3
"""
per_patient_effects.py
======================
Per-patient (cluster-level) breakdown of the paired method comparisons on the
risk-enriched CACHET-CADB test set.

WHY THIS EXISTS
---------------
The risk-enriched test set has ~25,600 windows but only FOUR patients. The
window-level bootstrap / Wilcoxon results already reported in the manuscript
treat each window as independent, which overstates certainty (pseudoreplication).
`evaluate_stage3.py --paired_stats[_transfer]` already emits an aggregate
patient-level cluster bootstrap CI (columns cluster_*). This script adds the
missing artefact reviewers asked for: the PER-PATIENT effect for each
comparison / stratum / metric, so you can show that the direction of the effect
is consistent across all four patients rather than driven by one.

With N = 4 clusters these numbers are DESCRIPTIVE. Do not report a 4-sample
Wilcoxon p-value as inferential evidence (its smallest possible two-sided value
is 0.125). Report the per-patient point estimates, the across-patient mean, the
cluster bootstrap 95% CI, and the direction-agreement count.

USAGE
-----
  # teacher-gated vs CACHET synthetic-pair model
  python per_patient_effects.py --split test --comparison synthetic \
         --synth_checkpoint checkpoints/denoiser_synth_best.pt

  # teacher-gated vs MIT-BIH/NSTDB transfer model
  python per_patient_effects.py --split test --comparison transfer

OUTPUT
------
  logs/per_patient_effects_{synthetic|transfer}_{split}.csv
  logs/per_patient_effects_{synthetic|transfer}_{split}.json
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import h5py
import torch

# Reuse the exact evaluation pipeline so metrics are identical to the manuscript.
import evaluate_stage3 as E
from config import OUTPUT_ROOT, LOG_DIR, STAGE3_MAX_RESIDUAL
from dataset import load_norm_stats

HEADLINE_STRATA = ("all", "high_mai", "very_high_mai", "emg_flagged")
HEADLINE_METRICS = ("qrs_corr", "artifact_log_reduction",
                    "hf_noise_log_reduction", "legacy_delta_snr")


def build_args_namespace(cli):
    """A Namespace with every attribute evaluate_stage3 helpers may read."""
    ns = argparse.Namespace(
        split=cli.split,
        n_chunks=cli.n_chunks,
        checkpoint=None,
        use_last=False,
        max_residual=cli.max_residual,
        gate_low=0.02,
        gate_high=0.12,
        synth_checkpoint=cli.synth_checkpoint,
    )
    return ns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--comparison", default="synthetic",
                    choices=["synthetic", "transfer"])
    ap.add_argument("--synth_checkpoint",
                    default="checkpoints/denoiser_synth_best.pt")
    ap.add_argument("--n_chunks", type=int, default=100)
    ap.add_argument("--bootstrap_iters", type=int, default=10000)
    ap.add_argument("--bootstrap_seed", type=int, default=123)
    ap.add_argument("--max_residual", type=float, default=STAGE3_MAX_RESIDUAL)
    cli = ap.parse_args()

    args = build_args_namespace(cli)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    h5_path = OUTPUT_ROOT / f"{cli.split}.h5"
    norm = load_norm_stats(OUTPUT_ROOT / "norm_stats.json")

    n_requested = cli.n_chunks * 256
    with h5py.File(h5_path, "r") as f:
        n_requested = min(n_requested, int(f.attrs["n_windows"]))
    risk_indices = E.build_risk_sample_indices(h5_path, n_requested)

    print(f"Matched deterministic risk sample: {len(risk_indices):,} windows")
    print("Evaluating teacher_gated ...")
    teacher = E.evaluate(None, h5_path, norm, device, n_chunks=cli.n_chunks,
                         mode="teacher_gated", risk_sample=True,
                         risk_indices=risk_indices)

    if cli.comparison == "synthetic":
        args.checkpoint = cli.synth_checkpoint
        print(f"Evaluating CACHET synthetic-pair model "
              f"({Path(cli.synth_checkpoint).name}) ...")
        model = E.load_model(device, args)
        comp = E.evaluate(model, h5_path, norm, device, n_chunks=cli.n_chunks,
                          mode="model", risk_sample=True,
                          risk_indices=risk_indices)
        comp_name = "synthetic"
    else:
        args.checkpoint = None
        print("Evaluating MIT-BIH/NSTDB transfer model ...")
        model = E.load_ecg_denoiser(device, args)
        comp = E.evaluate(model, h5_path, norm, device, n_chunks=cli.n_chunks,
                          mode="ecg_denoiser", risk_sample=True,
                          risk_indices=risk_indices)
        comp_name = "transfer"

    masks = E._stratum_masks(teacher)
    patients_all = np.array(teacher["patient_id"], dtype=object)

    rows = []
    for stratum, mask in masks.items():
        mask = np.asarray(mask, dtype=bool)
        if not mask.any():
            continue
        for metric_name, spec in E.PAIRED_METRICS.items():
            t = E._metric_array(teacher, metric_name)[mask]
            m = E._metric_array(comp, metric_name)[mask]
            pts = patients_all[mask]
            finite = np.isfinite(t) & np.isfinite(m)
            t, m, pts = t[finite], m[finite], pts[finite]
            if t.size == 0:
                continue
            diff = t - m  # teacher minus comparator, sign already per PAIRED_METRICS
            uniq = sorted({str(p) for p in pts})
            per_patient = []
            for p in uniq:
                sel = pts.astype(str) == str(p)
                per_patient.append((p, int(sel.sum()), float(np.mean(diff[sel]))))
            pmeans = np.array([x[2] for x in per_patient], dtype=np.float64)

            n_pat, cl_lo, cl_hi, _cl_p = E._cluster_bootstrap_ci(
                diff, pts, n_boot=cli.bootstrap_iters, seed=cli.bootstrap_seed)

            n_pos = int((pmeans > 0).sum())
            n_neg = int((pmeans < 0).sum())
            row = {
                "comparison": f"teacher_gated_minus_{comp_name}",
                "stratum": stratum,
                "metric": metric_name,
                "n_windows": int(t.size),
                "n_patients": int(pmeans.size),
                "across_patient_mean": float(np.mean(pmeans)),
                "across_patient_sd": float(np.std(pmeans, ddof=1)) if pmeans.size > 1 else 0.0,
                "window_level_mean": float(np.mean(diff)),
                "direction_agreement": f"{max(n_pos, n_neg)}/{pmeans.size}",
                "favours_teacher_if_positive": bool(spec["higher_is_better"]),
                "cluster_ci95_low": cl_lo,
                "cluster_ci95_high": cl_hi,
                "unit": spec["unit"],
            }
            for p, npw, d in per_patient:
                row[f"patient[{p}]_diff"] = d
                row[f"patient[{p}]_nwin"] = npw
            rows.append(row)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out_json = LOG_DIR / f"per_patient_effects_{comp_name}_{cli.split}.json"
    out_csv = LOG_DIR / f"per_patient_effects_{comp_name}_{cli.split}.csv"
    with open(out_json, "w") as f:
        json.dump(rows, f, indent=2)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print(f"\n  Per-patient effects JSON -> {out_json}")
    print(f"  Per-patient effects CSV  -> {out_csv}\n")
    print("Headline rows (descriptive; N=4 clusters):")
    for r in rows:
        if r["stratum"] in HEADLINE_STRATA and r["metric"] in HEADLINE_METRICS:
            print(f"  {r['stratum']:<14} {r['metric']:<22} "
                  f"across-patient mean={r['across_patient_mean']:+.5f} "
                  f"SD={r['across_patient_sd']:.5f} "
                  f"CI=[{r['cluster_ci95_low']:+.5f}, {r['cluster_ci95_high']:+.5f}] "
                  f"agree={r['direction_agreement']}")


if __name__ == "__main__":
    main()
