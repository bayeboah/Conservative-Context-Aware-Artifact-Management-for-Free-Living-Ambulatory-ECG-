"""
build_context_stratified_split.py
---------------------------------
Create a deterministic patient-isolated train/val/test split whose context
profile matches the global dataset as closely as possible.

The script profiles the existing HDF5 splits, because those files already
contain the full per-window context produced by preprocessing. It then searches
over patient-level assignments, preserving strict patient isolation.

Outputs:
  output/context_stratified_split.json

Usage:
  python build_context_stratified_split.py
  python build_context_stratified_split.py --seed 7 --write
"""

import argparse
import itertools
import json
from pathlib import Path

import h5py
import numpy as np

from config import OUTPUT_ROOT, ensure_output_dirs
from split_config import PATIENT_REGISTRY


N_ACTIVITY = 16
N_BODY = 16
SPLIT_SIZES = {"train": 16, "val": 4, "test": 4}


def _decode_patient_ids(arr):
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in arr])


def collect_patient_profiles(output_root: Path):
    profiles = {}
    seen_sources = {}
    for split in ("train", "val", "test"):
        path = output_root / f"{split}.h5"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Build the current dataset once before "
                "creating a context-stratified manifest."
            )
        with h5py.File(path, "r") as f:
            patients = _decode_patient_ids(f["patient_id"][:])
            unique = sorted(set(patients))
            for pid in unique:
                mask = patients == pid
                idx = np.where(mask)[0]
                n = len(idx)
                if n == 0:
                    continue
                activity = f["activity_class"][idx].astype(np.int64)
                body = f["body_position"][idx].astype(np.int64)
                mai = f["movement_accel"][idx].astype(np.float32)
                emg_flag = f["emg_flagged"][idx].astype(np.float32)
                emg_ratio = f["emg_ratio"][idx].astype(np.float32)
                noise = f["noise_annotated"][idx].astype(np.float32)
                reliable = f["rpeaks_reliable"][idx].astype(np.float32)
                rhythm = f["rhythm_class"][idx].astype(np.int64)

                act_counts = np.bincount(
                    activity[(activity >= 0) & (activity < N_ACTIVITY)],
                    minlength=N_ACTIVITY,
                ).astype(np.float64)
                body_counts = np.bincount(
                    body[(body >= 0) & (body < N_BODY)],
                    minlength=N_BODY,
                ).astype(np.float64)
                profiles[pid] = {
                    "patient": pid,
                    "n": float(n),
                    "site": PATIENT_REGISTRY[pid]["site"],
                    "has_af_inferred": float(PATIENT_REGISTRY[pid]["has_af"]),
                    "nsr_only": float(PATIENT_REGISTRY[pid]["nsr_only"]),
                    "days": float(PATIENT_REGISTRY[pid]["days"]),
                    "activity_counts": act_counts,
                    "body_counts": body_counts,
                    "high_motion": float((mai > 0.08).sum()),
                    "mai_sum": float(mai.sum()),
                    "emg_flag": float(emg_flag.sum()),
                    "emg_ratio_sum": float(emg_ratio.sum()),
                    "noise": float(noise.sum()),
                    "rpeak_reliable": float(reliable.sum()),
                    "rhythm_af": float((rhythm == 1).sum()),
                    "rhythm_nsr": float((rhythm == 2).sum()),
                    "rhythm_noise": float((rhythm == 3).sum()),
                }
                seen_sources[pid] = split
    expected = set(PATIENT_REGISTRY)
    found = set(profiles)
    if found != expected:
        raise RuntimeError(
            f"Profile patient mismatch. Missing={sorted(expected - found)} "
            f"extra={sorted(found - expected)}"
        )
    return profiles, seen_sources


def combine_profiles(profiles, patients):
    agg = {
        "n": 0.0,
        "days": 0.0,
        "has_af_inferred": 0.0,
        "nsr_only": 0.0,
        "site_dk": 0.0,
        "activity_counts": np.zeros(N_ACTIVITY, dtype=np.float64),
        "body_counts": np.zeros(N_BODY, dtype=np.float64),
        "high_motion": 0.0,
        "mai_sum": 0.0,
        "emg_flag": 0.0,
        "emg_ratio_sum": 0.0,
        "noise": 0.0,
        "rpeak_reliable": 0.0,
        "rhythm_af": 0.0,
        "rhythm_nsr": 0.0,
        "rhythm_noise": 0.0,
    }
    for pid in patients:
        p = profiles[pid]
        agg["n"] += p["n"]
        agg["days"] += p["days"]
        agg["has_af_inferred"] += p["has_af_inferred"]
        agg["nsr_only"] += p["nsr_only"]
        agg["site_dk"] += 1.0 if p["site"] == "DK" else 0.0
        agg["activity_counts"] += p["activity_counts"]
        agg["body_counts"] += p["body_counts"]
        for key in (
            "high_motion", "mai_sum", "emg_flag", "emg_ratio_sum",
            "noise", "rpeak_reliable", "rhythm_af", "rhythm_nsr",
            "rhythm_noise",
        ):
            agg[key] += p[key]
    return agg


def feature_vector(agg):
    n = max(agg["n"], 1.0)
    act = agg["activity_counts"] / max(agg["activity_counts"].sum(), 1.0)
    body = agg["body_counts"] / max(agg["body_counts"].sum(), 1.0)
    scalar = np.array([
        agg["high_motion"] / n,
        agg["mai_sum"] / n,
        agg["emg_flag"] / n,
        agg["emg_ratio_sum"] / n,
        agg["noise"] / n,
        agg["rpeak_reliable"] / n,
        agg["rhythm_af"] / n,
        agg["rhythm_nsr"] / n,
        agg["rhythm_noise"] / n,
        agg["nsr_only"] / 24.0,
        agg["site_dk"] / 24.0,
    ], dtype=np.float64)
    return np.concatenate([scalar * 2.0, act, body])


def split_loss(profiles, global_vec, val, test):
    all_patients = set(profiles)
    train = sorted(all_patients - set(val) - set(test))
    val = sorted(val)
    test = sorted(test)

    loss = 0.0
    for name, patients in (("train", train), ("val", val), ("test", test)):
        vec = feature_vector(combine_profiles(profiles, patients))
        diff = vec - global_vec
        weight = 1.0
        if name in ("val", "test"):
            weight = 2.0
        loss += weight * float(np.mean(diff * diff))

    # Soft structural constraints: keep Danish and PNSR represented in every
    # split, and keep observed AF annotations represented where possible.
    # Per-patient AF history is not source-verified in the publication, so future
    # split searches balance observed rhythm annotations instead of the inferred
    # registry-level has_af flag.
    for patients in (train, val, test):
        sites = {profiles[p]["site"] for p in patients}
        if "DK" not in sites:
            loss += 0.05
        if not any(profiles[p]["nsr_only"] for p in patients):
            loss += 0.02
        if sum(profiles[p]["rhythm_af"] for p in patients) <= 0:
            loss += 0.02
    return loss, {"train": train, "val": val, "test": test}


def search_split(profiles, seed: int, n_random: int):
    patients = sorted(profiles)
    global_vec = feature_vector(combine_profiles(profiles, patients))
    rng = np.random.default_rng(seed)

    best_loss = float("inf")
    best = None

    # Exhaustively choose val; sample test from the remaining patients. This is
    # deterministic with the seed and gives much better coverage than greedy
    # one-pass assignment for only 24 patients.
    for val in itertools.combinations(patients, SPLIT_SIZES["val"]):
        remaining = sorted(set(patients) - set(val))
        candidates = []
        candidates.extend(itertools.combinations(remaining, SPLIT_SIZES["test"]))
        if len(candidates) > n_random:
            pick = rng.choice(len(candidates), size=n_random, replace=False)
            candidates = [candidates[i] for i in pick]
        for test in candidates:
            loss, splits = split_loss(profiles, global_vec, val, test)
            if loss < best_loss:
                best_loss = loss
                best = splits
    return best_loss, best


def serialise_summary(profiles, patients):
    agg = combine_profiles(profiles, patients)
    n = max(agg["n"], 1.0)
    return {
        "patients": patients,
        "n_windows": int(agg["n"]),
        "days": int(agg["days"]),
        "site_dk": int(agg["site_dk"]),
        "has_af_inferred": int(agg["has_af_inferred"]),
        "nsr_only": int(agg["nsr_only"]),
        "high_motion_pct": 100.0 * agg["high_motion"] / n,
        "emg_flag_pct": 100.0 * agg["emg_flag"] / n,
        "noise_pct": 100.0 * agg["noise"] / n,
        "rpeak_reliable_pct": 100.0 * agg["rpeak_reliable"] / n,
        "mai_mean": agg["mai_sum"] / n,
        "emg_ratio_mean": agg["emg_ratio_sum"] / n,
        "rhythm_af": int(agg["rhythm_af"]),
        "rhythm_nsr": int(agg["rhythm_nsr"]),
        "rhythm_noise": int(agg["rhythm_noise"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--n-random-test", type=int, default=600)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    ensure_output_dirs()
    profiles, source_splits = collect_patient_profiles(OUTPUT_ROOT)
    loss, splits = search_split(profiles, args.seed, args.n_random_test)

    manifest = {
        "method": "context-stratified-patient-split",
        "seed": args.seed,
        "objective_loss": loss,
        "splits": splits,
        "summary": {
            split: serialise_summary(profiles, pts)
            for split, pts in splits.items()
        },
        "source_splits": source_splits,
    }

    print(json.dumps(manifest["summary"], indent=2))
    print("\nSplits:")
    for split in ("train", "val", "test"):
        print(f"  {split:<5}: {splits[split]}")
    print(f"\nObjective loss: {loss:.8f}")

    out_path = OUTPUT_ROOT / "context_stratified_split.json"
    if args.write:
        with open(out_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"\nWrote {out_path}")
        print("Next: python dataset.py --force")
    else:
        print("\nDry run only. Add --write to save the manifest.")


if __name__ == "__main__":
    main()
