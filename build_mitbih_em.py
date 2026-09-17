"""
build_mitbih_em.py
------------------
Builds the cross-dataset training corpus for the MIT-BIH / NSTDB electrode-motion
transfer arm (see MITBIH_TRANSFER_EXPERIMENT_DESIGN.md).

Produces clean/noisy 10-second ECG window pairs, harmonised to the CACHET-CADB
representation so that a denoiser trained here can be evaluated on CACHET without
a trivial format-mismatch confound:

    * MLII lead only (single-lead, matched to CACHET)
    * resampled 360 -> 1024 Hz (polyphase, exact 128/45)
    * 0.5-40 Hz 4th-order bandpass (identical to CACHET preprocessing)
    * per-window normalisation (amplitude-agnostic; the key confound control)

Electrode-motion (em) noise is injected at a per-window target SNR sampled from a
grid (or a continuous range) so the corpus spans many degradation levels. The
per-window input SNR is stored so downstream results can be stratified by it.

Output HDF5 datasets (per split):
    clean      (N, 10240) float32   per-window-normalised clean MLII target
    noisy      (N, 10240) float32   per-window-normalised noisy input
    snr_in     (N,)       float32   achieved input SNR (dB)
    record_id  (N,)       int16     source MIT-BIH record number
    window_idx (N,)       int32     window position within record

Usage:
    python build_mitbih_em.py                        # defaults
    python build_mitbih_em.py --snr_mode uniform --snr_low -6 --snr_high 24
    python build_mitbih_em.py --stride 5120          # 50% overlap, ~2x windows
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import h5py
from scipy.signal import resample_poly, butter, sosfiltfilt

try:
    import wfdb
except ImportError:
    sys.exit("wfdb not installed. Run:  pip install wfdb")


# ── Harmonisation constants (match CACHET) ────────────────────────────────────
SRC_FS      = 360                 # MIT-BIH / NSTDB native rate
DST_FS      = 1024                # CACHET rate
UP, DOWN    = 128, 45             # 360 * 128 / 45 = 1024 exactly
WINDOW_SEC  = 10
WINDOW_LEN  = DST_FS * WINDOW_SEC  # 10240
BP_LOW, BP_HIGH, BP_ORDER = 0.5, 40.0, 4
DEFAULT_SNR_GRID = [24, 18, 12, 6, 0, -6]   # NSTDB standard grid (dB)

# MIT-BIH has 48 records from 47 patients.  Records 201 and 202 are from the
# same individual (male, age 68).  All other records map 1-to-1 to a patient.
# Grouping by patient prevents data leakage across train/val splits.
SAME_PATIENT_RECORDS = {"201": "P201", "202": "P201"}


def record_to_patient(rec: str) -> str:
    """Map a MIT-BIH record ID to a patient identifier."""
    return SAME_PATIENT_RECORDS.get(rec, f"P{rec}")


def find_db_root(project_root: Path) -> Path:
    """Locate the folder that directly contains the MIT-BIH .dat records."""
    for hea in project_root.rglob("100.hea"):
        return hea.parent
    sys.exit(f"Could not find MIT-BIH records (100.hea) under {project_root}")


def bandpass(x: np.ndarray) -> np.ndarray:
    sos = butter(BP_ORDER, [BP_LOW, BP_HIGH], btype="bandpass",
                 fs=DST_FS, output="sos")
    return sosfiltfilt(sos, x).astype(np.float32)


def load_mlii(db_root: Path, rec: str):
    """Return the resampled, bandpassed MLII channel for a record, or None."""
    hdr = wfdb.rdheader(str(db_root / rec))
    if "MLII" not in hdr.sig_name:
        return None                      # skip V5-only records (102, 104, 114)
    ch = hdr.sig_name.index("MLII")
    sig = wfdb.rdrecord(str(db_root / rec), channels=[ch]).p_signal[:, 0]
    sig = np.nan_to_num(sig, nan=0.0).astype(np.float64)
    sig = resample_poly(sig, UP, DOWN)   # 360 -> 1024 Hz
    return bandpass(sig)


def load_em_pool(db_root: Path) -> np.ndarray:
    """Electrode-motion noise, resampled + bandpassed to match. Channel 0."""
    em = wfdb.rdrecord(str(db_root / "noise" / "em"), channels=[0]).p_signal[:, 0]
    em = np.nan_to_num(em, nan=0.0).astype(np.float64)
    em = resample_poly(em, UP, DOWN)
    return bandpass(em)


def window_signal(sig: np.ndarray, stride: int):
    """Yield (window_idx, clean_window) non-overlapping (or strided)."""
    n = len(sig)
    for wi, start in enumerate(range(0, n - WINDOW_LEN + 1, stride)):
        yield wi, sig[start:start + WINDOW_LEN]


def sample_snr(rng, mode, grid, low, high) -> float:
    if mode == "grid":
        return float(rng.choice(grid))
    return float(rng.uniform(low, high))


def inject(clean: np.ndarray, em_pool: np.ndarray, target_snr_db: float, rng):
    """
    noisy = clean + a * em_segment, with a set so the input SNR equals target.

    SNR_in(dB) = 10 log10( P_clean / P_noise );  a = sqrt(P_clean/P_em) * 10^(-SNR/20)
    """
    start = rng.integers(0, len(em_pool) - WINDOW_LEN)
    em = em_pool[start:start + WINDOW_LEN].astype(np.float64)
    p_clean = float(np.mean(clean.astype(np.float64) ** 2))
    p_em    = float(np.mean(em ** 2))
    if p_em < 1e-12 or p_clean < 1e-12:
        return None, None
    a = np.sqrt(p_clean / p_em) * (10.0 ** (-target_snr_db / 20.0))
    noisy = clean.astype(np.float64) + a * em
    achieved = 10.0 * np.log10(p_clean / (a * a * p_em))   # == target, for logging
    return noisy.astype(np.float32), float(achieved)


def per_window_norm(clean: np.ndarray, noisy: np.ndarray):
    """
    Normalise by the OBSERVABLE noisy window's statistics (available at inference),
    applying the same shift/scale to the clean target so the residual stays
    consistent. This is the amplitude-agnostic control that lets the model transfer
    across datasets without an absolute-scale mismatch.
    """
    mu = float(noisy.mean())
    sd = float(noisy.std())
    if sd < 1e-8:
        return None, None
    return ((clean - mu) / sd).astype(np.float32), ((noisy - mu) / sd).astype(np.float32)


class StreamWriter:
    """Append windows to resizable HDF5 datasets one record at a time.

    Keeps peak memory to a single record's worth of windows, so the build runs
    safely on low-memory machines (the in-memory-then-stack approach OOMs).
    """

    def __init__(self, path):
        self.f = h5py.File(path, "w")
        self.n = 0
        self.d = {}

    def _init(self):
        mk = lambda name, w, dt: self.f.create_dataset(
            name, shape=((0, w) if w else (0,)),
            maxshape=((None, w) if w else (None,)), dtype=dt,
            chunks=((64, w) if w else (1024,)),
            compression="gzip", compression_opts=4)
        self.d["clean"]      = mk("clean", WINDOW_LEN, "f4")
        self.d["noisy"]      = mk("noisy", WINDOW_LEN, "f4")
        self.d["snr_in"]     = mk("snr_in", 0, "f4")
        self.d["record_id"]  = mk("record_id", 0, "i2")
        self.d["window_idx"] = mk("window_idx", 0, "i4")

    def append(self, clean, noisy, snr, rec, win):
        if not self.d:
            self._init()
        k = len(clean)
        if k == 0:
            return
        for name, arr in (("clean", clean), ("noisy", noisy), ("snr_in", snr),
                          ("record_id", rec), ("window_idx", win)):
            ds = self.d[name]
            ds.resize(self.n + k, axis=0)
            ds[self.n:self.n + k] = arr
        self.n += k

    def close(self):
        self.f.attrs["n_windows"] = self.n
        self.f.attrs["fs"] = DST_FS
        self.f.attrs["window_len"] = WINDOW_LEN
        self.f.attrs["noise"] = "em"
        self.f.attrs["bandpass"] = f"{BP_LOW}-{BP_HIGH}Hz order{BP_ORDER}"
        self.f.attrs["normalisation"] = "per-window (noisy mean/std)"
        self.f.attrs["split_strategy"] = "patient-level"
        self.f.close()


def build_split(records, db_root, em_pool, args, rng, writer):
    total = 0
    for rec in records:
        mlii = load_mlii(db_root, rec)
        if mlii is None:
            print(f"  {rec}: no MLII channel — skipped", flush=True)
            continue
        C, Nz, S, R, W = [], [], [], [], []
        for wi, clean in window_signal(mlii, args.stride):
            target = sample_snr(rng, args.snr_mode, args.snr_grid,
                                args.snr_low, args.snr_high)
            noisy, achieved = inject(clean, em_pool, target, rng)
            if noisy is None:
                continue
            cn, nn = per_window_norm(clean, noisy)
            if cn is None:
                continue
            C.append(cn); Nz.append(nn); S.append(achieved)
            R.append(int(rec)); W.append(wi)
        if C:
            writer.append(np.stack(C), np.stack(Nz),
                          np.asarray(S, np.float32), np.asarray(R, np.int16),
                          np.asarray(W, np.int32))
        total += len(C)
        print(f"  {rec}: {len(C)} windows  (cumulative {total})", flush=True)
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--out_dir", type=Path, default=None,
                    help="Default: <project_root>/output")
    ap.add_argument("--val_frac", type=float, default=0.2,
                    help="Fraction of PATIENTS held out for in-domain validation.")
    ap.add_argument("--stride", type=int, default=WINDOW_LEN,
                    help="Window stride in samples (default = non-overlapping).")
    ap.add_argument("--snr_mode", choices=["grid", "uniform"], default="grid")
    ap.add_argument("--snr_grid", type=float, nargs="+", default=DEFAULT_SNR_GRID)
    ap.add_argument("--snr_low", type=float, default=-6.0)
    ap.add_argument("--snr_high", type=float, default=24.0)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    out_dir = args.out_dir or (args.project_root / "output")
    out_dir.mkdir(parents=True, exist_ok=True)
    db_root = find_db_root(args.project_root)
    rng = np.random.default_rng(args.seed)

    print("=" * 60)
    print("  MIT-BIH + NSTDB electrode-motion corpus build")
    print("=" * 60)
    print(f"  DB root   : {db_root}")
    print(f"  Resample  : {SRC_FS} -> {DST_FS} Hz  ({UP}/{DOWN})")
    print(f"  Window    : {WINDOW_SEC}s = {WINDOW_LEN} samp   stride={args.stride}")
    print(f"  Bandpass  : {BP_LOW}-{BP_HIGH} Hz order {BP_ORDER}")
    print(f"  SNR       : {args.snr_mode} "
          f"{args.snr_grid if args.snr_mode=='grid' else [args.snr_low, args.snr_high]} dB")
    print(f"  Seed      : {args.seed}")
    print()

    all_recs = sorted({p.stem for p in db_root.glob("[0-9][0-9][0-9].hea")})

    # ── Patient-level split (records 201 & 202 share one patient) ─────────────
    patients: dict[str, list[str]] = {}
    for rec in all_recs:
        pid = record_to_patient(rec)
        patients.setdefault(pid, []).append(rec)
    patient_ids = sorted(patients.keys())
    rng.shuffle(patient_ids)
    n_val = max(1, int(round(len(patient_ids) * args.val_frac)))
    val_patient_set = set(patient_ids[:n_val])
    val_recs = sorted(r for pid in val_patient_set for r in patients[pid])
    train_recs = sorted(r for r in all_recs if r not in set(val_recs))

    print(f"  Patients  : {len(patient_ids)} total  ->  "
          f"{len(patient_ids) - n_val} train / {n_val} val "
          f"(patient-level, no leakage)")
    print(f"  Records   : {len(all_recs)} total  ->  "
          f"{len(train_recs)} train / {len(val_recs)} val")
    print(f"  Val patients: {sorted(val_patient_set)}")
    print(f"  Val recs  : {val_recs}\n")

    em_pool = load_em_pool(db_root)
    print(f"  em pool   : {len(em_pool):,} samples "
          f"({len(em_pool)/DST_FS:.0f}s)  std={em_pool.std():.4f}\n")

    print("Building TRAIN ...", flush=True)
    tr_w = StreamWriter(out_dir / "mitbih_em_train.h5")
    n_tr = build_split(train_recs, db_root, em_pool, args, rng, tr_w)
    tr_w.close()

    print("\nBuilding VAL ...", flush=True)
    va_w = StreamWriter(out_dir / "mitbih_em_val.h5")
    n_va = build_split(val_recs, db_root, em_pool, args, rng, va_w)
    va_w.close()

    def snr_hist(path):
        with h5py.File(path, "r") as f:
            snr = f["snr_in"][:]
        edges = [-9, -3, 3, 9, 15, 21, 27]
        h, _ = np.histogram(snr, bins=edges)
        return {f"[{edges[i]},{edges[i+1]})": int(h[i]) for i in range(len(h))}

    print("\n" + "=" * 60)
    print(f"  TRAIN: {n_tr:,} windows  ->  {out_dir/'mitbih_em_train.h5'}")
    print(f"         SNR spread: {snr_hist(out_dir/'mitbih_em_train.h5')}")
    print(f"  VAL  : {n_va:,} windows  ->  {out_dir/'mitbih_em_val.h5'}")
    print(f"         SNR spread: {snr_hist(out_dir/'mitbih_em_val.h5')}")
    print(f"  clean/noisy dtype float32, per-window normalised, "
          f"window={WINDOW_LEN}")
    print("=" * 60)
    print("\nNext: train_mitbih_denoiser.py  (ECG-only, bounded residual)")


if __name__ == "__main__":
    main()
