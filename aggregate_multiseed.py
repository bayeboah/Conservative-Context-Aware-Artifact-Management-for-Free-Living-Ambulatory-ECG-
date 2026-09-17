#!/usr/bin/env python3
"""
aggregate_multiseed.py
======================
Aggregates the per-seed evaluation JSONs produced by run_multiseed_baselines.sh
into mean +/- SD for each supervised model, so the manuscript can report
seed variability instead of a single lucky run.

Reads:   logs/multiseed/synth_seed*.json
         logs/multiseed/mitbih_seed*.json
Writes:  logs/multiseed/multiseed_summary.csv
         logs/multiseed/multiseed_summary.json

The evaluation `results` dict schema is discovered automatically: every scalar
numeric leaf present in ALL seeds of a model is summarised. Keys whose name
mentions qrs / artifact / snr / hf / residual are surfaced first as the
headline rows.
"""
import glob
import json
from pathlib import Path
from statistics import mean, pstdev, stdev

LOG_DIR = Path(__file__).resolve().parent / "logs" / "multiseed"

HEADLINE_HINTS = ("qrs", "artifact", "snr", "hf", "residual", "corr")


def flatten(obj, prefix=""):
    """Yield (dotted_key, value) for every scalar numeric leaf."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from flatten(v, f"{prefix}{k}.")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from flatten(v, f"{prefix}{i}.")
    else:
        if isinstance(obj, bool):
            return
        if isinstance(obj, (int, float)):
            yield prefix[:-1], float(obj)


def summarise(files, model_name):
    per_seed = []
    for fp in files:
        with open(fp) as f:
            data = json.load(f)
        per_seed.append(dict(flatten(data)))
    if not per_seed:
        print(f"[warn] no seed files for {model_name}")
        return []

    # keys present in every seed
    common = set(per_seed[0])
    for d in per_seed[1:]:
        common &= set(d)

    rows = []
    for key in sorted(common):
        vals = [d[key] for d in per_seed]
        n = len(vals)
        mu = mean(vals)
        sd = stdev(vals) if n > 1 else 0.0
        rows.append({
            "model": model_name,
            "metric": key,
            "n_seeds": n,
            "mean": mu,
            "sd": sd,
            "min": min(vals),
            "max": max(vals),
            "values": vals,
            "headline": any(h in key.lower() for h in HEADLINE_HINTS),
        })
    return rows


def main():
    synth_files = sorted(glob.glob(str(LOG_DIR / "synth_seed*.json")))
    mitbih_files = sorted(glob.glob(str(LOG_DIR / "mitbih_seed*.json")))

    rows = []
    rows += summarise(synth_files, "CACHET_synthetic")
    rows += summarise(mitbih_files, "MITBIH_transfer")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out_json = LOG_DIR / "multiseed_summary.json"
    out_csv = LOG_DIR / "multiseed_summary.csv"
    with open(out_json, "w") as f:
        json.dump(rows, f, indent=2)
    import csv
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "metric", "n_seeds", "mean", "sd", "min", "max", "values"])
        for r in rows:
            w.writerow([r["model"], r["metric"], r["n_seeds"],
                        f"{r['mean']:.6f}", f"{r['sd']:.6f}",
                        f"{r['min']:.6f}", f"{r['max']:.6f}",
                        ";".join(f"{v:.6f}" for v in r["values"])])

    print(f"  Summary JSON -> {out_json}")
    print(f"  Summary CSV  -> {out_csv}\n")
    print("Headline metrics (mean +/- SD across seeds):")
    for r in rows:
        if r["headline"]:
            print(f"  {r['model']:<18} {r['metric']:<48} "
                  f"{r['mean']:+.5f} +/- {r['sd']:.5f}  (n={r['n_seeds']})")


if __name__ == "__main__":
    main()
