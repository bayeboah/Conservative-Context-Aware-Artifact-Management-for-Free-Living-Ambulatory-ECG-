"""
evaluate_stage3.py — Stage 3 Denoiser Evaluation
==================================================
Evaluates the trained CrossAttentionUNet denoiser on the test split.

Metrics:
  1. ΔSNR   — signal-to-noise improvement in artifact band (5-40 Hz)
  2. QRS correlation — morphology preservation at R-peak templates
  3. IMU contribution — ΔSNR stratified by activity (high vs low MAI)
  4. Before/after plots — visual comparison of denoised ECG

Usage:
  python evaluate_stage3.py
  python evaluate_stage3.py --split val --n_chunks 1
  python evaluate_stage3.py --checkpoint checkpoints/denoiser_last.pt
"""

import argparse, csv, json
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.signal as sp_sig
import scipy.stats as sp_stats
import h5py
import torch

from config import (
    OUTPUT_ROOT, CHECKPOINT_DIR, LOG_DIR, ECG_FS,
    STAGE3_MAX_RESIDUAL, ensure_output_dirs,
)
from dataset import load_norm_stats
from denoiser import (
    ResidualArtifactRemover,
    generate_pseudo_target,
    gated_teacher_denoise,
)
from siamese_alignment import SiameseAlignmentModule

# ── Paths ─────────────────────────────────────────────────────────────────────
STAGE2_CKPT = CHECKPOINT_DIR / "siamese_best.pt"
STAGE3_CKPT = CHECKPOINT_DIR / "denoiser_best.pt"
STAGE3_LAST = CHECKPOINT_DIR / "denoiser_last.pt"


def safe_torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)

# ── Plot style ─────────────────────────────────────────────────────────────────
BG       = "#0D1117"
PANEL_BG = "#161B22"
TEXT     = "#E6EDF3"
C_ECG    = "#3FB950"   # clean / output (green)
C_NOISY  = "#F78166"   # noisy / input  (red-orange)
C_IMU    = "#79C0FF"   # IMU             (blue)
C_GRID   = "#30363D"

PAPER_NOISY_SIGNAL_PCT = {
    "P1": 7.48, "P2": 6.47, "P3": 8.40, "P4": 5.80,
    "P5": 16.27, "P6": 3.26, "P7": 12.43, "P8": 9.90,
    "P9": 29.61, "P10": 31.72, "P11": 12.72, "P12": 6.88,
    "P13": 2.65, "P14": 29.90, "P15": 28.05, "P16": 6.29,
    "P17": 6.02, "P18": 3.23, "P19": 3.59, "P21": 10.34,
    "P23": 7.24, "PNSR-1": 0.49, "PNSR-3": 27.14,
    "PNSR-4": 5.63,
}


# =============================================================================
# Signal utilities
# =============================================================================
def bandpower(sig: np.ndarray, fs: int,
              flo: float, fhi: float) -> float:
    """Mean power in [flo, fhi] Hz band."""
    nperseg = len(sig) if flo < 1.0 else min(1024, len(sig))
    f, pxx = sp_sig.welch(sig, fs=fs, nperseg=nperseg)
    mask = (f >= flo) & (f <= fhi)
    if not mask.any():
        raise ValueError(
            f"Band [{flo}, {fhi}] Hz contains no Welch bin at "
            f"fs={fs}, nperseg={nperseg}, resolution={fs/nperseg:.4f} Hz."
        )
    return float(pxx[mask].mean())


def snr_ratio(sig: np.ndarray, fs: int = ECG_FS) -> float:
    """Legacy cardiac/artifact ratio kept for continuity with old reports."""
    cardiac  = bandpower(sig, fs, 0.5,  5.0)
    artifact = bandpower(sig, fs, 5.0, 40.0)
    return cardiac / (artifact + 1e-12)


def qrs_motion_snr(sig: np.ndarray, fs: int = ECG_FS) -> float:
    """QRS-focused SNR: QRS band over gait/baseline motion band."""
    qrs = bandpower(sig, fs, 8.0, 20.0)
    motion = bandpower(sig, fs, 1.0, 5.0)
    return qrs / (motion + 1e-12)


def artifact_powers(sig: np.ndarray, fs: int = ECG_FS) -> dict:
    """Artifact bands targeted by the residual-teacher Stage 3 objective."""
    baseline = bandpower(sig, fs, 0.05, 0.5)
    hf_noise = bandpower(sig, fs, 45.0, 100.0)
    qrs      = bandpower(sig, fs, 5.0, 15.0)
    return {
        "baseline": baseline,
        "hf_noise": hf_noise,
        "artifact": baseline + hf_noise,
        "qrs": qrs,
    }


def log_delta(out_power: float, in_power: float) -> float:
    """Natural-log power change. Negative means reduction."""
    return float(np.log((out_power + 1e-12) / (in_power + 1e-12)))


def hist_bins(values: np.ndarray, n_bins: int = 40, pad: float = 1e-3):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return n_bins
    lo, hi = float(finite.min()), float(finite.max())
    if np.isclose(lo, hi):
        return np.linspace(lo - pad, hi + pad, n_bins + 1)
    return n_bins


def detect_rpeaks(sig: np.ndarray, fs: int = ECG_FS) -> np.ndarray:
    """Simple Pan-Tompkins inspired R-peak detection."""
    # Bandpass 8-20 Hz
    b, a   = sp_sig.butter(2, [8/(fs/2), 20/(fs/2)], btype='band')
    filt   = sp_sig.filtfilt(b, a, sig)
    squared = filt ** 2
    # Moving integration window ~150ms
    win = int(0.15 * fs)
    integrated = np.convolve(squared, np.ones(win)/win, mode='same')
    # Peak detection: min distance 400ms
    peaks, _ = sp_sig.find_peaks(integrated, distance=int(0.4 * fs),
                                  height=integrated.max() * 0.2)
    return peaks


def qrs_template_corr(ecg_in: np.ndarray, ecg_out: np.ndarray,
                       fs: int = ECG_FS, half_win_ms: int = 150) -> float:
    """Mean correlation between input and output QRS templates."""
    hw     = int(half_win_ms / 1000 * fs)
    peaks  = detect_rpeaks(ecg_in, fs)
    # Only peaks that fit within the signal
    peaks  = peaks[(peaks >= hw) & (peaks < len(ecg_in) - hw)]
    if len(peaks) == 0:
        return np.nan
    corrs = []
    for p in peaks:
        t_in  = ecg_in[p - hw: p + hw]
        t_out = ecg_out[p - hw: p + hw]
        if t_in.std() < 1e-6 or t_out.std() < 1e-6:
            continue
        c = np.corrcoef(t_in, t_out)[0, 1]
        corrs.append(c)
    return float(np.nanmean(corrs)) if corrs else np.nan


def beat_detection_metrics(sig_in: np.ndarray, sig_out: np.ndarray,
                           fs: int = ECG_FS, tol_ms: int = 50) -> dict:
    """Independent, detection-based morphology endpoint.

    Uses the same R-peak detector on the input and output signals and measures
    how well the beat train survives each method. Unlike QRS template
    correlation (which is 1.0 by construction when a window is left unchanged),
    this quantifies whether beats are lost, spuriously added, or shifted in
    timing after intervention. The input-detected peaks act as the reference
    beat set (no clean reference exists in CACHET-CADB), so this reports
    input-versus-output beat-train agreement rather than accuracy against truth.
    """
    pin = detect_rpeaks(sig_in, fs)
    pout = detect_rpeaks(sig_out, fs)
    n_in, n_out = int(len(pin)), int(len(pout))
    nan = float("nan")
    if n_in == 0:
        return dict(beat_f1=nan, beat_sens=nan, beat_ppv=nan,
                    rr_mae_ms=nan, n_beats_in=0, n_beats_out=n_out)
    tol = int(tol_ms / 1000.0 * fs)
    used = np.zeros(n_out, dtype=bool)
    tp = 0
    for pk in pin:
        if n_out == 0:
            break
        d = np.abs(pout - pk)
        j = int(np.argmin(d))
        if d[j] <= tol and not used[j]:
            used[j] = True
            tp += 1
    sens = tp / n_in
    ppv = (tp / n_out) if n_out > 0 else nan
    f1 = (2 * sens * ppv / (sens + ppv)) if (sens + ppv) > 0 else 0.0
    rr_in = np.diff(pin) / fs * 1000.0
    rr_out = np.diff(pout) / fs * 1000.0
    if rr_in.size >= 1 and rr_out.size >= 1:
        rr_mae = float(abs(np.median(rr_in) - np.median(rr_out)))
    else:
        rr_mae = nan
    return dict(beat_f1=float(f1), beat_sens=float(sens),
                beat_ppv=float(ppv) if np.isfinite(ppv) else nan,
                rr_mae_ms=rr_mae, n_beats_in=n_in, n_beats_out=n_out)


# =============================================================================
# Model loading
# =============================================================================
def resolve_stage3_checkpoint(args) -> Path:
    """Resolve the Stage 3 checkpoint requested by CLI args."""
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    elif args.use_last:
        ckpt_path = STAGE3_LAST
    elif STAGE3_CKPT.exists():
        ckpt_path = STAGE3_CKPT
    elif STAGE3_LAST.exists():
        ckpt_path = STAGE3_LAST
    else:
        available = sorted(p.name for p in CHECKPOINT_DIR.glob("*.pt"))
        raise FileNotFoundError(
            "Stage 3 checkpoint not found.\n"
            f"  Expected: {STAGE3_CKPT}\n"
            f"  Fallback: {STAGE3_LAST}\n"
            f"  Available checkpoints: {available or 'none'}\n"
            "  Run train_stage3.py first, or pass --checkpoint PATH."
        )

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {ckpt_path}")
    return ckpt_path


def load_model(device: torch.device, args) -> ResidualArtifactRemover:
    ckpt_path = resolve_stage3_checkpoint(args)

    model = ResidualArtifactRemover(max_residual=args.max_residual).to(device)
    model.load_imu_encoder(str(STAGE2_CKPT), device)

    ckpt = safe_torch_load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    epoch    = ckpt.get("epoch", "?")
    val_loss = ckpt.get("val_loss", float("nan"))
    print(f"Loaded {ckpt_path.name}  (epoch {epoch}, val_loss={val_loss:.4f})")
    model.eval()
    return model


def load_ecg_denoiser(device: torch.device, args):
    """Load the ECG-only cross-dataset transfer model (MIT-BIH/NSTDB-em).

    Separate architecture from the CACHET ResidualArtifactRemover: no IMU pathway.
    Reconstructs the U-Net from the arch dict stored at training time so the
    checkpoint is self-describing.
    """
    from train_mitbih_denoiser import ECGUNet1D
    ckpt_path = (Path(args.checkpoint) if args.checkpoint
                 else CHECKPOINT_DIR / "mitbih_em_denoiser_best.pt")
    # This checkpoint embeds harmless pathlib objects (from the saved args dict),
    # which the weights_only=True loader rejects by default. Allowlist Path types
    # for this trusted, locally-produced checkpoint.
    try:
        ckpt = safe_torch_load(ckpt_path, map_location=device)
    except Exception:
        import pathlib
        with torch.serialization.safe_globals([
                pathlib.PurePath, pathlib.PurePosixPath, pathlib.PureWindowsPath,
                pathlib.Path, pathlib.PosixPath, pathlib.WindowsPath]):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    arch = ckpt.get("arch", {"base": 16, "depth": 4, "max_residual": 0.35})
    model = ECGUNet1D(base=arch["base"], depth=arch["depth"],
                      max_residual=arch["max_residual"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    imp = ckpt.get("val", {}).get("snr_improvement_mean", float("nan"))
    print(f"Loaded ECG-only transfer model {ckpt_path.name}  "
          f"(epoch {ckpt.get('epoch','?')}, in-domain SNR improvement {imp:+.2f} dB)")
    return model


# =============================================================================
# Chunk loader (same fast approach as evaluate_stage2.py)
# =============================================================================
def _decode_patient_array(arr):
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in arr])


def pseudo_target_for_mode(ecg: torch.Tensor, mode: str) -> torch.Tensor:
    """Evaluation-only pseudo-target variants for frequency-design ablations."""
    if mode in ("teacher_gated", "teacher_ungated"):
        return generate_pseudo_target(ecg, fs=ECG_FS)
    if mode in ("teacher_inband_gated", "teacher_inband_ungated"):
        # Stress test: same teacher form, but attenuation starts inside the
        # diagnostic/high-EMG passband rather than above it. This is deliberately
        # less conservative and is used only to test the frequency-design choice.
        return generate_pseudo_target(
            ecg, fs=ECG_FS, baseline_scale=0.25,
            hf_cut=20.0, hf_edge=4.0, hf_scale=0.10,
        )
    if mode in ("teacher_broadband_gated", "teacher_broadband_ungated"):
        # Stronger broad high-frequency attenuation ablation; useful for showing
        # the morphology cost of a less cautious pseudo-target under the same gate.
        return generate_pseudo_target(
            ecg, fs=ECG_FS, baseline_scale=0.50,
            hf_cut=15.0, hf_edge=5.0, hf_scale=0.25,
        )
    raise ValueError(f"Unknown pseudo-target mode: {mode}")


def gated_teacher_variant(ecg: torch.Tensor, mai: torch.Tensor,
                          noise_annotated: torch.Tensor, mode: str,
                          gate_low: float, gate_high: float) -> tuple:
    pseudo = pseudo_target_for_mode(ecg, mode)
    gate = ResidualArtifactRemover.motion_gate(
        mai, noise_annotated, low=gate_low, high=gate_high
    )
    clean = (1.0 - gate) * ecg + gate * pseudo
    residual = ecg - clean
    return clean, residual


def build_risk_sample_indices(h5_path: Path, n_samples: int, seed: int = 13) -> np.ndarray:
    """Select deterministic windows enriched for publication/context risk groups."""
    rng = np.random.default_rng(seed)
    with h5py.File(h5_path, "r") as f:
        n_total = int(f.attrs["n_windows"])
        patient = _decode_patient_array(f["patient_id"][:])
        day = f["day"][:].astype(np.int32)
        window_idx = f["window_idx"][:].astype(np.int32)
        mai = f["movement_accel"][:].astype(np.float32)
        activity = f["activity_class"][:].astype(np.int32)
        body = f["body_position"][:].astype(np.int32)
        noise = f["noise_annotated"][:].astype(bool)
        reliable = f["rpeaks_reliable"][:].astype(bool)
        emg_flag = f["emg_flagged"][:].astype(bool)

    order = np.lexsort((window_idx, day, patient))
    transition = np.zeros(n_total, dtype=bool)
    prev = order[:-1]
    cur = order[1:]
    adjacent = (
        (patient[prev] == patient[cur])
        & (day[prev] == day[cur])
        & (window_idx[cur] == window_idx[prev] + 1)
    )
    changed = (
        (activity[prev] != activity[cur])
        | (body[prev] != body[cur])
        | (np.abs(mai[cur] - mai[prev]) >= 0.05)
    )
    transition[cur[adjacent & changed]] = True

    high_noise_patient = np.array([
        PAPER_NOISY_SIGNAL_PCT.get(str(p), 0.0) >= 20.0 for p in patient
    ], dtype=bool)

    groups = [
        np.where(mai >= 0.08)[0],
        np.where(mai >= 0.15)[0],
        np.where(transition)[0],
        np.where(noise)[0],
        np.where(emg_flag)[0],
        np.where(~reliable)[0],
        np.where(high_noise_patient)[0],
    ]

    chosen = []
    quota = max(1, n_samples // max(len(groups), 1))
    for idx in groups:
        if len(idx) == 0:
            continue
        take = min(quota, len(idx))
        chosen.extend(rng.choice(idx, size=take, replace=False).tolist())

    chosen = np.array(sorted(set(chosen)), dtype=np.int64)
    if len(chosen) < n_samples:
        remaining = np.setdiff1d(np.arange(n_total, dtype=np.int64), chosen, assume_unique=False)
        take = min(n_samples - len(chosen), len(remaining))
        if take > 0:
            extra = rng.choice(remaining, size=take, replace=False)
            chosen = np.concatenate([chosen, extra])

    if len(chosen) > n_samples:
        chosen = rng.choice(chosen, size=n_samples, replace=False)
    return np.sort(chosen.astype(np.int64))


def _read_h5_selection(f, name: str, selection: np.ndarray):
    order = np.argsort(selection)
    sorted_idx = selection[order]
    inv = np.argsort(order)
    data = f[name][sorted_idx]
    return data[inv]


def chunk_loader(h5_path: Path, norm: dict,
                 chunk: int = 512, n_chunks: int = None,
                 indices: np.ndarray = None):
    ecg_mean = norm["ecg_mean"]
    ecg_std  = norm["ecg_std"]
    imu_mean = np.array(norm["imu_mean"], dtype=np.float32)
    imu_std  = np.array(norm["imu_std"],  dtype=np.float32)

    with h5py.File(h5_path, "r") as f:
        n_total = int(f.attrs["n_windows"])
        if indices is None:
            n_read = n_total if n_chunks is None else min(n_total, n_chunks * chunk)
            batches = (np.arange(start, min(start + chunk, n_read)) for start in range(0, n_read, chunk))
        else:
            indices = np.asarray(indices, dtype=np.int64)
            batches = (indices[start:start + chunk] for start in range(0, len(indices), chunk))

        for sel in batches:
            ecg  = _read_h5_selection(f, "ecg", sel).astype(np.float32)
            acc  = _read_h5_selection(f, "acc", sel).astype(np.float32)
            gyro = _read_h5_selection(f, "gyro", sel).astype(np.float32)
            mai  = _read_h5_selection(f, "movement_accel", sel).astype(np.float32)
            lw   = _read_h5_selection(f, "loss_weight", sel).astype(np.float32)
            act  = _read_h5_selection(f, "activity_class", sel).astype(np.int32)
            body = _read_h5_selection(f, "body_position", sel).astype(np.int32)
            noise = _read_h5_selection(f, "noise_annotated", sel).astype(np.float32)
            rel = _read_h5_selection(f, "rpeaks_reliable", sel).astype(np.float32)
            emg_flag = _read_h5_selection(f, "emg_flagged", sel).astype(np.float32)
            emg_ratio = _read_h5_selection(f, "emg_ratio", sel).astype(np.float32)
            day = _read_h5_selection(f, "day", sel).astype(np.int32)
            window_idx = _read_h5_selection(f, "window_idx", sel).astype(np.int32)
            patient_raw = _read_h5_selection(f, "patient_id", sel)
            patient = [x.decode() if isinstance(x, bytes) else str(x) for x in patient_raw]

            ecg  = (ecg  - ecg_mean) / ecg_std
            acc  = (acc  - imu_mean[:3, None]) / imu_std[:3, None]
            gyro = (gyro - imu_mean[3:, None]) / imu_std[3:, None]
            imu  = np.concatenate([acc, gyro], axis=1)

            yield {
                "ecg":            torch.from_numpy(ecg),
                "imu":            torch.from_numpy(imu),
                "movement_accel": torch.from_numpy(mai),
                "loss_weight":    torch.from_numpy(lw),
                "activity_class": torch.from_numpy(act),
                "body_position": torch.from_numpy(body),
                "noise_annotated": torch.from_numpy(noise),
                "rpeaks_reliable": torch.from_numpy(rel),
                "emg_flagged": torch.from_numpy(emg_flag),
                "emg_ratio": torch.from_numpy(emg_ratio),
                "day": torch.from_numpy(day),
                "window_idx": torch.from_numpy(window_idx),
                "patient_id": patient,
            }


# =============================================================================
# Evaluation loop
# =============================================================================
@torch.no_grad()
def evaluate(model, h5_path: Path, norm: dict,
             device: torch.device, n_chunks: int = 100,
             mode: str = "model", risk_sample: bool = False,
             risk_indices: np.ndarray = None,
             gate_low: float = 0.02,
             gate_high: float = 0.12,
             gate_baseline: bool = False):
    metrics = defaultdict(list)
    prev_patient = None
    prev_day = None
    prev_window = None
    prev_activity = None
    prev_body = None
    prev_mai = None

    n_requested = None if n_chunks is None else n_chunks * 256
    if risk_sample:
        if risk_indices is None:
            with h5py.File(h5_path, "r") as f:
                n_total = int(f.attrs["n_windows"])
            n_requested = n_total if n_requested is None else min(n_requested, n_total)
            risk_indices = build_risk_sample_indices(h5_path, n_requested)
        print(f"  Risk sampling enabled: selected {len(risk_indices):,} windows")

    for batch in chunk_loader(
        h5_path, norm, chunk=256, n_chunks=n_chunks, indices=risk_indices
    ):
        ecg  = batch["ecg"].to(device)     # [B, 1, 10240]
        imu  = batch["imu"].to(device)     # [B, 6,  640]
        mai_t = batch["movement_accel"].to(device)
        noise_t = batch["noise_annotated"].to(device)
        mai  = batch["movement_accel"].numpy()
        act  = batch["activity_class"].numpy()
        body = batch["body_position"].numpy()
        noise_np = batch["noise_annotated"].numpy()
        reliable_np = batch["rpeaks_reliable"].numpy()
        emg_flag_np = batch["emg_flagged"].numpy()
        emg_ratio_np = batch["emg_ratio"].numpy()
        day_np = batch["day"].numpy()
        window_np = batch["window_idx"].numpy()
        patient_ids = batch["patient_id"]

        with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
            if mode == "identity":
                ecg_out = ecg
                residual = torch.zeros_like(ecg)
                shuffled_out = ecg_out
            elif mode in (
                "teacher_ungated",
                "teacher_inband_ungated",
                "teacher_broadband_ungated",
            ):
                ecg_out = pseudo_target_for_mode(ecg, mode)
                residual = ecg - ecg_out
                shuffled_out = ecg_out
            elif mode in (
                "teacher_gated",
                "teacher_inband_gated",
                "teacher_broadband_gated",
            ):
                ecg_out, residual = gated_teacher_variant(
                    ecg, mai_t, noise_t, mode=mode,
                    gate_low=gate_low, gate_high=gate_high,
                )
                shuffled_out = ecg_out
            elif mode == "ecg_denoiser":
                # Cross-dataset ECG-only transfer model. Re-normalise each CACHET
                # window to the per-window convention the MIT-BIH model trained on
                # (mean 0 / std 1), run, then invert so the output returns to the
                # CACHET-normalised scale and every downstream metric compares
                # like-for-like with teacher_gated and the CACHET synthetic model.
                w_mu = ecg.mean(dim=2, keepdim=True)
                w_sd = ecg.std(dim=2, keepdim=True).clamp_min(1e-6)
                clean_pw, _ = model((ecg - w_mu) / w_sd)
                ecg_out = clean_pw * w_sd + w_mu
                if gate_baseline:
                    g = ResidualArtifactRemover.motion_gate(
                        mai_t, noise_t, low=gate_low, high=gate_high)
                    ecg_out = (1.0 - g) * ecg + g * ecg_out
                residual = ecg - ecg_out
                shuffled_out = ecg_out          # no IMU -> imu_effect = 0
            else:
                ecg_out, residual = model(
                    ecg, imu, mai=mai_t, noise_annotated=noise_t,
                    return_residual=True,
                )
                if gate_baseline:
                    g = ResidualArtifactRemover.motion_gate(
                        mai_t, noise_t, low=gate_low, high=gate_high)
                    ecg_out = (1.0 - g) * ecg + g * ecg_out
                    residual = ecg - ecg_out
                shuffled_imu = imu[torch.randperm(imu.shape[0], device=device)]
                shuffled_out, _ = model(
                    ecg, shuffled_imu, mai=mai_t, noise_annotated=noise_t,
                    return_residual=True,
                )

        ecg_np  = ecg.squeeze(1).cpu().numpy()      # [B, 10240]
        out_np  = ecg_out.squeeze(1).cpu().numpy()  # [B, 10240]
        shuf_np = shuffled_out.squeeze(1).cpu().numpy()
        res_np  = residual.squeeze(1).cpu().numpy()

        for i in range(len(ecg_np)):
            sig_in  = ecg_np[i]
            sig_out = out_np[i]
            sig_shuf = shuf_np[i]
            sig_res = res_np[i]

            snr_in  = snr_ratio(sig_in)
            snr_out = snr_ratio(sig_out)
            delta   = snr_out - snr_in
            qrs_motion_in = qrs_motion_snr(sig_in)
            qrs_motion_out = qrs_motion_snr(sig_out)
            qrs_motion_delta = qrs_motion_out - qrs_motion_in
            artifact_in = artifact_powers(sig_in)
            artifact_out = artifact_powers(sig_out)
            imu_effect = float(np.mean(np.abs(sig_out - sig_shuf)))
            residual_rms = float(np.sqrt(np.mean(sig_res ** 2)))
            delta_abs = float(np.mean(np.abs(sig_out - sig_in)))

            qrs_c   = qrs_template_corr(sig_in, sig_out)
            beat_m  = beat_detection_metrics(sig_in, sig_out)

            metrics["snr_in"].append(snr_in)
            metrics["snr_out"].append(snr_out)
            metrics["delta_snr"].append(delta)
            metrics["qrs_motion_snr_in"].append(qrs_motion_in)
            metrics["qrs_motion_snr_out"].append(qrs_motion_out)
            metrics["delta_qrs_motion_snr"].append(qrs_motion_delta)
            metrics["delta_baseline_log"].append(
                log_delta(artifact_out["baseline"], artifact_in["baseline"])
            )
            metrics["delta_hf_noise_log"].append(
                log_delta(artifact_out["hf_noise"], artifact_in["hf_noise"])
            )
            metrics["delta_artifact_log"].append(
                log_delta(artifact_out["artifact"], artifact_in["artifact"])
            )
            metrics["delta_qrs_log"].append(
                log_delta(artifact_out["qrs"], artifact_in["qrs"])
            )
            metrics["qrs_corr"].append(qrs_c)
            metrics["beat_f1"].append(beat_m["beat_f1"])
            metrics["beat_sens"].append(beat_m["beat_sens"])
            metrics["beat_ppv"].append(beat_m["beat_ppv"])
            metrics["rr_mae_ms"].append(beat_m["rr_mae_ms"])
            metrics["n_beats_in"].append(beat_m["n_beats_in"])
            metrics["n_beats_out"].append(beat_m["n_beats_out"])
            metrics["mai"].append(float(mai[i]))
            metrics["activity"].append(int(act[i]))
            metrics["body_position"].append(int(body[i]))
            metrics["noise_annotated"].append(float(noise_np[i]))
            metrics["rpeaks_reliable"].append(float(reliable_np[i]))
            metrics["emg_flagged"].append(float(emg_flag_np[i]))
            metrics["emg_ratio"].append(float(emg_ratio_np[i]))
            metrics["day"].append(int(day_np[i]))
            metrics["window_idx"].append(int(window_np[i]))
            metrics["patient_id"].append(patient_ids[i])
            metrics["imu_effect"].append(imu_effect)
            metrics["residual_rms"].append(residual_rms)
            metrics["delta_abs"].append(delta_abs)

            same_context = (
                prev_patient == patient_ids[i]
                and prev_day == int(day_np[i])
                and prev_window is not None
                and int(window_np[i]) == prev_window + 1
            )
            activity_transition = bool(same_context and int(act[i]) != prev_activity)
            body_transition = bool(same_context and int(body[i]) != prev_body)
            mai_jump = bool(
                same_context
                and prev_mai is not None
                and abs(float(mai[i]) - prev_mai) >= 0.05
            )
            metrics["activity_transition"].append(float(activity_transition))
            metrics["body_transition"].append(float(body_transition))
            metrics["mai_jump"].append(float(mai_jump))
            metrics["context_transition"].append(
                float(activity_transition or body_transition or mai_jump)
            )

            prev_patient = patient_ids[i]
            prev_day = int(day_np[i])
            prev_window = int(window_np[i])
            prev_activity = int(act[i])
            prev_body = int(body[i])
            prev_mai = float(mai[i])

    return metrics


# =============================================================================
# Before / after signal plot
# =============================================================================
def plot_before_after(model, h5_path: Path, norm: dict,
                      device: torch.device, out_path: Path,
                      n_examples: int = 4,
                      mode: str = "model",
                      gate_low: float = 0.02,
                      gate_high: float = 0.12,
                      seed: int = 13):
    """
    Plot ECG before and after denoising for n_examples windows,
    chosen to span a range of MAI values (high motion preferred).
    Also shows IMU magnitude for context.
    """
    rng = np.random.default_rng(seed)
    ecg_mean = norm["ecg_mean"]
    ecg_std  = norm["ecg_std"]
    imu_mean = np.array(norm["imu_mean"], dtype=np.float32)
    imu_std  = np.array(norm["imu_std"],  dtype=np.float32)

    examples = []
    with h5py.File(h5_path, "r") as f:
        n_total = int(f.attrs["n_windows"])
        mai_all = f["movement_accel"][:n_total]
        act_all = f["activity_class"][:n_total]

        # Pick windows: 2 high-MAI (walking) + 2 low-MAI (sitting/lying)
        walk_idx = np.where(mai_all > 0.10)[0]
        rest_idx = np.where(mai_all < 0.02)[0]
        chosen   = []
        if len(walk_idx) >= 2:
            chosen += list(rng.choice(walk_idx, 2, replace=False))
        if len(rest_idx) >= 2:
            chosen += list(rng.choice(rest_idx, 2, replace=False))
        if len(chosen) < n_examples:
            remaining = np.setdiff1d(np.arange(n_total), np.array(chosen, dtype=int))
            n_more = min(n_examples - len(chosen), len(remaining))
            if n_more > 0:
                chosen += list(rng.choice(remaining, n_more, replace=False))
        chosen = chosen[:n_examples]

        for idx in chosen:
            ecg_raw  = f["ecg"][idx].astype(np.float32)
            acc_raw  = f["acc"][idx].astype(np.float32)
            gyro_raw = f["gyro"][idx].astype(np.float32)
            mai_val  = float(f["movement_accel"][idx])
            act_val  = int(f["activity_class"][idx])

            ecg_norm  = (ecg_raw  - ecg_mean) / ecg_std
            acc_norm  = (acc_raw  - imu_mean[:3, None]) / imu_std[:3, None]
            gyro_norm = (gyro_raw - imu_mean[3:, None]) / imu_std[3:, None]
            imu_norm  = np.concatenate([acc_norm, gyro_norm], axis=0)

            examples.append({
                "ecg": ecg_norm,
                "imu": imu_norm,
                "mai": mai_val,
                "act": act_val,
                "noise": float(f["noise_annotated"][idx]),
            })

    if not examples:
        print(f"  Before/after plot skipped: no examples found in {h5_path}")
        return

    # Run model on examples
    ecg_t = torch.from_numpy(
        np.stack([e["ecg"] for e in examples])).to(device)  # [N, 1, 10240]
    imu_t = torch.from_numpy(
        np.stack([e["imu"] for e in examples])).to(device)  # [N, 6, 640]

    with torch.no_grad():
        mai_t = torch.tensor([e["mai"] for e in examples], dtype=torch.float32,
                             device=device)
        noise_t = torch.tensor([e["noise"] for e in examples],
                               dtype=torch.float32, device=device)
        if mode == "identity":
            out_t = ecg_t
            res_t = torch.zeros_like(ecg_t)
        elif mode in (
            "teacher_ungated",
            "teacher_inband_ungated",
            "teacher_broadband_ungated",
        ):
            out_t = pseudo_target_for_mode(ecg_t, mode)
            res_t = ecg_t - out_t
        elif mode in (
            "teacher_gated",
            "teacher_inband_gated",
            "teacher_broadband_gated",
        ):
            out_t, res_t = gated_teacher_variant(
                ecg_t, mai_t, noise_t, mode=mode,
                gate_low=gate_low, gate_high=gate_high,
            )
        elif mode == "ecg_denoiser":
            w_mu = ecg_t.mean(dim=2, keepdim=True)
            w_sd = ecg_t.std(dim=2, keepdim=True).clamp_min(1e-6)
            out_pw, _ = model((ecg_t - w_mu) / w_sd)
            out_t = out_pw * w_sd + w_mu
            res_t = ecg_t - out_t
        else:
            out_t, res_t = model(
                ecg_t, imu_t, mai=mai_t, noise_annotated=noise_t,
                return_residual=True,
            )

    ecg_in  = ecg_t.squeeze(1).cpu().numpy()
    ecg_out = out_t.squeeze(1).cpu().numpy()
    res_out = res_t.squeeze(1).cpu().numpy()
    imu_mag = np.linalg.norm(imu_t[:, :3].cpu().numpy(), axis=1)  # acc mag

    act_names = {1: "Lying", 2: "Sitting", 3: "Walking",
                 4: "Running", 5: "Cycling", 6: "Driving", 7: "Sleeping"}

    t_ecg = np.arange(10240) / ECG_FS          # ECG time axis (seconds)
    t_imu = np.arange(640)   / 64              # IMU time axis (seconds)

    n = len(examples)
    # ── Publication white theme (self-contained; overrides module dark palette) ──
    fig, axes = plt.subplots(n, 3, figsize=(13, 2.7 * n), squeeze=False,
                             facecolor="white")
    fig.suptitle("ECG artifact management — before vs. after", fontsize=13,
                 fontweight="bold", y=0.995)
    c_in, c_out, c_res, c_imu = "#B0B0B0", "#0072B2", "#D55E00", "#009E73"

    for row, (ex, ei, eo, im) in enumerate(
            zip(examples, ecg_in, ecg_out, imu_mag)):
        ax1, ax2, ax3 = axes[row]
        act_name = act_names.get(ex["act"], f"act={ex['act']}")
        snr_i = snr_ratio(ei); snr_o = snr_ratio(eo)
        dsnr = snr_o - snr_i
        qc = qrs_template_corr(ei, eo)
        qc_txt = f"QRS corr {qc:.4f}" if not np.isnan(qc) else "QRS corr n/a"

        for ax in (ax1, ax2, ax3):
            ax.set_facecolor("white")
            ax.grid(True, color="#e6e6e6", linewidth=0.7)
            ax.tick_params(labelsize=7)
            for s in ax.spines.values():
                s.set_color("#333333")
            if row == n - 1:
                ax.set_xlabel("Time (s)", fontsize=8)

        # Panel 1: input ECG
        ax1.plot(t_ecg, ei, color="#C44E52", lw=0.5)
        ax1.set_title(f"Input — {act_name}, MAI {ex['mai']:.3f}g", fontsize=8.5)
        ax1.set_ylabel("Norm. amplitude", fontsize=8)

        # Panel 2: output overlaid on input, with residual
        ax2.plot(t_ecg, ei, color=c_in, lw=0.5, alpha=0.7, label="Input")
        ax2.plot(t_ecg, eo, color=c_out, lw=0.7, label="Denoised")
        ax2.plot(t_ecg, res_out[row], color=c_res, lw=0.5, alpha=0.8,
                 label="Residual")
        ax2.set_title(f"Output — ΔSNR {dsnr:+.3f}, {qc_txt}", fontsize=8.5)
        ax2.legend(loc="upper right", fontsize=6.5)

        # Panel 3: IMU acceleration magnitude
        ax3.plot(t_imu, im, color=c_imu, lw=0.7)
        ax3.axhline(0.08, color="#E69F00", lw=0.9, ls="--",
                    label="MAI gate 0.08g")
        ax3.set_title("IMU acceleration magnitude", fontsize=8.5)
        ax3.set_ylabel("Norm. magnitude", fontsize=8)
        ax3.legend(loc="upper right", fontsize=6.5)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    plt.close()
    print(f"  Before/after plot -> {out_path}")


# =============================================================================
# Summary plot (metrics)
# =============================================================================
def plot_metrics(metrics: dict, results: dict, split: str, out_path: Path):
    mai  = np.array(metrics["mai"])
    dsnr = np.array(metrics["delta_snr"])
    qc   = np.array(metrics["qrs_corr"])
    qc   = qc[~np.isnan(qc)]
    imu_effect = np.array(metrics["imu_effect"])
    residual_rms = np.array(metrics["residual_rms"])

    fig  = plt.figure(figsize=(16, 8), facecolor=BG)
    gs   = gridspec.GridSpec(2, 3, figure=fig,
                              hspace=0.45, wspace=0.35,
                              left=0.07, right=0.97,
                              top=0.90, bottom=0.10)

    def ax_style(ax, title):
        ax.set_facecolor(PANEL_BG)
        ax.set_title(title, color=TEXT, fontsize=9, pad=5)
        ax.tick_params(colors=TEXT, labelsize=8)
        for sp in ax.spines.values():
            sp.set_color(C_GRID)
        ax.grid(True, color=C_GRID, lw=0.4, alpha=0.5)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)

    # 1. ΔSNR distribution
    ax = fig.add_subplot(gs[0, 0])
    ax.hist(dsnr, bins=hist_bins(dsnr), color=C_ECG, alpha=0.85, edgecolor="none")
    ax.axvline(0, color="#EF9F27", lw=1.2, ls="--", label="no change")
    ax.axvline(dsnr.mean(), color=C_NOISY, lw=1.5, ls="-",
               label=f"mean={dsnr.mean():.3f}")
    ax_style(ax, "ΔSNR distribution")
    ax.set_xlabel("ΔSNR (higher = more denoising)")
    ax.set_ylabel("Count")
    ax.legend(fontsize=7, facecolor=PANEL_BG, labelcolor=TEXT,
              edgecolor=C_GRID)

    # 2. ΔSNR vs MAI scatter
    ax = fig.add_subplot(gs[0, 1])
    sc = ax.scatter(mai, dsnr, c=dsnr, cmap="RdYlGn",
                    s=2, alpha=0.4, linewidths=0)
    ax.axhline(0, color="#EF9F27", lw=0.8, ls="--")
    ax.axvline(0.080, color=C_IMU, lw=0.8, ls="--",
               label="MAI gate (0.08g)")
    cb = plt.colorbar(sc, ax=ax, pad=0.02)
    cb.ax.tick_params(colors=TEXT, labelsize=7)
    ax_style(ax, "ΔSNR vs motion intensity")
    ax.set_xlabel("MAI (g)")
    ax.set_ylabel("ΔSNR")
    ax.legend(fontsize=7, facecolor=PANEL_BG, labelcolor=TEXT,
              edgecolor=C_GRID)

    # 3. ΔSNR by activity class
    ax    = fig.add_subplot(gs[0, 2])
    acts  = np.array(metrics["activity"])
    names = {1:"Lying", 2:"Sitting", 3:"Walking",
             4:"Running", 5:"Cycling", 7:"Sleeping"}
    present = sorted([k for k in names if (acts == k).any()])
    means   = [dsnr[acts == k].mean() for k in present]
    bars    = ax.bar([names[k] for k in present], means,
                      color=C_ECG, edgecolor="none", alpha=0.85)
    for bar, val in zip(bars, means):
        if val < 0:
            bar.set_color(C_NOISY)
    ax.axhline(0, color=TEXT, lw=0.5, alpha=0.5)
    ax_style(ax, "Mean ΔSNR by activity")
    ax.set_ylabel("Mean ΔSNR")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=7)

    # 4. QRS correlation distribution
    ax = fig.add_subplot(gs[1, 0])
    ax.hist(qc, bins=hist_bins(qc), color=C_IMU, alpha=0.85, edgecolor="none")
    ax.axvline(qc.mean(), color=C_NOISY, lw=1.5, ls="-",
               label=f"mean={qc.mean():.4f}")
    ax.axvline(0.99, color="#EF9F27", lw=1, ls="--", label="target 0.99")
    ax_style(ax, "QRS template correlation")
    ax.set_xlabel("Correlation (1 = perfect preservation)")
    ax.set_ylabel("Count")
    ax.legend(fontsize=7, facecolor=PANEL_BG, labelcolor=TEXT,
              edgecolor=C_GRID)

    # 5. SNR before vs after scatter
    ax = fig.add_subplot(gs[1, 1])
    snr_i = np.array(metrics["snr_in"])
    snr_o = np.array(metrics["snr_out"])
    ax.scatter(snr_i, snr_o, c=dsnr, cmap="RdYlGn",
               s=2, alpha=0.4, linewidths=0)
    lims = [min(snr_i.min(), snr_o.min()),
            max(snr_i.max(), snr_o.max())]
    ax.plot(lims, lims, color="#EF9F27", lw=0.8, ls="--",
            label="no change")
    ax_style(ax, "SNR: input vs output")
    ax.set_xlabel("Input SNR")
    ax.set_ylabel("Output SNR")
    ax.legend(fontsize=7, facecolor=PANEL_BG, labelcolor=TEXT,
              edgecolor=C_GRID)

    # 6. Summary table
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    rows = [
        ["Metric",                     "Value"],
        ["QRS/motion dSNR",            f"{results['mean_delta_qrs_motion_snr']:+.4f}"],
        ["High-MAI artifact red.",     f"{results['high_mai_artifact_log_reduction']:+.4f}"],
        ["Low-MAI residual RMS",       f"{results['low_mai_residual_rms']:.4f}"],
        ["Mean ΔSNR",                  f"{results['mean_delta_snr']:+.4f}"],
        ["High-MAI ΔSNR (>0.08g)",     f"{results['high_mai_delta_snr']:+.4f}"],
        ["Low-MAI ΔSNR (<0.02g)",      f"{results['low_mai_delta_snr']:+.4f}"],
        ["Mean QRS corr",              f"{results['mean_qrs_corr']:.4f}"],
        ["Mean residual RMS",          f"{results['mean_residual_rms']:.4f}"],
        ["Mean IMU effect",            f"{results['mean_imu_effect']:.4f}"],
        ["Low-MAI change",             f"{results['low_mai_delta_abs']:.4f}"],
        ["% windows improved",         f"{results['pct_improved']:.1f}%"],
        ["Windows evaluated",          f"{results['n_windows']:,}"],
    ]
    tbl = ax.table(cellText=rows[1:], colLabels=rows[0],
                   cellLoc="center", loc="center",
                   bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor(PANEL_BG if r > 0 else "#21262D")
        cell.set_text_props(color=TEXT)
        cell.set_edgecolor(C_GRID)
    ax.set_title("Summary", color=TEXT, fontsize=9, pad=5)

    fig.suptitle(f"Stage 3 Evaluation — {split} split",
                 color=TEXT, fontsize=13)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Metrics plot -> {out_path}")


# =============================================================================
# Risk-stratified metrics
# =============================================================================
def _mean_or_zero(values: np.ndarray, mask: np.ndarray) -> float:
    return float(values[mask].mean()) if mask.any() else 0.0


def stratified_results(metrics: dict) -> dict:
    dsnr = np.array(metrics["delta_snr"], dtype=np.float64)
    dqmsnr = np.array(metrics["delta_qrs_motion_snr"], dtype=np.float64)
    d_artifact = np.array(metrics["delta_artifact_log"], dtype=np.float64)
    d_baseline = np.array(metrics["delta_baseline_log"], dtype=np.float64)
    d_hf = np.array(metrics["delta_hf_noise_log"], dtype=np.float64)
    qrs_corr = np.array(metrics["qrs_corr"], dtype=np.float64)
    residual_rms = np.array(metrics["residual_rms"], dtype=np.float64)
    delta_abs = np.array(metrics["delta_abs"], dtype=np.float64)
    mai = np.array(metrics["mai"], dtype=np.float64)
    activity = np.array(metrics["activity"], dtype=np.int64)
    noise = np.array(metrics["noise_annotated"], dtype=np.float64) > 0.5
    reliable = np.array(metrics["rpeaks_reliable"], dtype=np.float64) > 0.5
    emg_flag = np.array(metrics["emg_flagged"], dtype=np.float64) > 0.5
    emg_ratio = np.array(metrics["emg_ratio"], dtype=np.float64)
    transition = np.array(metrics["context_transition"], dtype=np.float64) > 0.5
    patients = np.array(metrics["patient_id"], dtype=object)

    all_mask = np.ones_like(dsnr, dtype=bool)
    high_noise_patient = np.array([
        PAPER_NOISY_SIGNAL_PCT.get(str(p), 0.0) >= 20.0 for p in patients
    ], dtype=bool)
    high_motion_activity = np.isin(activity, [3, 4, 5])
    if len(emg_ratio):
        emg_ratio_high = emg_ratio >= np.nanpercentile(emg_ratio, 90)
    else:
        emg_ratio_high = all_mask & False

    groups = {
        "all": all_mask,
        "low_risk_identity": (
            (mai < 0.02)
            & ~noise
            & reliable
            & ~emg_flag
            & ~transition
        ),
        "high_mai": mai >= 0.08,
        "very_high_mai": mai >= 0.15,
        "high_motion_activity": high_motion_activity,
        "context_transition": transition,
        "noise_annotated": noise,
        "rpeak_unreliable": ~reliable,
        "emg_flagged": emg_flag,
        "emg_ratio_top10pct": emg_ratio_high,
        "publication_high_noise_patient": high_noise_patient,
        "publication_high_noise_and_high_mai": high_noise_patient & (mai >= 0.08),
    }

    out = {}
    for name, mask in groups.items():
        mask = np.asarray(mask, dtype=bool)
        qrs_group = qrs_corr[mask]
        qrs_group = qrs_group[np.isfinite(qrs_group)]
        dsnr_group = dsnr[mask]
        out[name] = {
            "n": int(mask.sum()),
            "pct_windows": float(100.0 * mask.mean()) if len(mask) else 0.0,
            "mean_delta_snr": _mean_or_zero(dsnr, mask),
            "median_delta_snr": float(np.median(dsnr_group)) if dsnr_group.size else 0.0,
            "q25_delta_snr": float(np.percentile(dsnr_group, 25)) if dsnr_group.size else 0.0,
            "q75_delta_snr": float(np.percentile(dsnr_group, 75)) if dsnr_group.size else 0.0,
            "mean_delta_qrs_motion_snr": _mean_or_zero(dqmsnr, mask),
            "artifact_log_reduction": _mean_or_zero(-d_artifact, mask),
            "baseline_log_reduction": _mean_or_zero(-d_baseline, mask),
            "hf_noise_log_reduction": _mean_or_zero(-d_hf, mask),
            "mean_qrs_corr": float(np.nanmean(qrs_group)) if qrs_group.size else 0.0,
            "mean_residual_rms": _mean_or_zero(residual_rms, mask),
            "mean_delta_abs": _mean_or_zero(delta_abs, mask),
            "pct_improved_legacy_snr": (
                float(100.0 * np.mean(dsnr_group > 0)) if dsnr_group.size else 0.0
            ),
        }
    return out


def print_strata_table(strata: dict) -> None:
    show = [
        "low_risk_identity",
        "high_mai",
        "very_high_mai",
        "context_transition",
        "noise_annotated",
        "rpeak_unreliable",
        "emg_flagged",
        "publication_high_noise_patient",
    ]
    print("\nRisk-stratified results:")
    print(
        "  "
        f"{'group':<34} {'n':>8} {'artifact':>10} "
        f"{'qrs/motion':>11} {'qrs_corr':>9} {'res_rms':>9}"
    )
    for name in show:
        row = strata.get(name, {})
        print(
            "  "
            f"{name:<34} {row.get('n', 0):>8,} "
            f"{row.get('artifact_log_reduction', 0.0):>+10.4f} "
            f"{row.get('mean_delta_qrs_motion_snr', 0.0):>+11.4f} "
            f"{row.get('mean_qrs_corr', 0.0):>9.4f} "
            f"{row.get('mean_residual_rms', 0.0):>9.4f}"
        )


# =============================================================================
# Paired method comparison
# =============================================================================
PAIRED_METRICS = {
    "legacy_delta_snr": {
        "source": "delta_snr",
        "transform": 1.0,
        "unit": "ratio-difference",
        "higher_is_better": True,
    },
    "qrs_motion_delta_snr": {
        "source": "delta_qrs_motion_snr",
        "transform": 1.0,
        "unit": "ratio-difference",
        "higher_is_better": True,
    },
    "artifact_log_reduction": {
        "source": "delta_artifact_log",
        "transform": -1.0,
        "unit": "natural-log power reduction",
        "higher_is_better": True,
    },
    "hf_noise_log_reduction": {
        "source": "delta_hf_noise_log",
        "transform": -1.0,
        "unit": "natural-log power reduction",
        "higher_is_better": True,
    },
    "qrs_corr": {
        "source": "qrs_corr",
        "transform": 1.0,
        "unit": "correlation",
        "higher_is_better": True,
    },
    "residual_rms": {
        "source": "residual_rms",
        "transform": 1.0,
        "unit": "normalised ECG amplitude",
        "higher_is_better": False,
    },
    "beat_f1": {
        "source": "beat_f1",
        "transform": 1.0,
        "unit": "F1 (input-vs-output beat agreement)",
        "higher_is_better": True,
    },
    "beat_sens": {
        "source": "beat_sens",
        "transform": 1.0,
        "unit": "sensitivity (beat retention)",
        "higher_is_better": True,
    },
    "rr_mae_ms": {
        "source": "rr_mae_ms",
        "transform": 1.0,
        "unit": "milliseconds",
        "higher_is_better": False,
    },
}


def _metric_array(metrics: dict, metric_name: str) -> np.ndarray:
    spec = PAIRED_METRICS[metric_name]
    return np.array(metrics[spec["source"]], dtype=np.float64) * spec["transform"]


def _stratum_masks(metrics: dict) -> dict:
    mai = np.array(metrics["mai"], dtype=np.float64)
    activity = np.array(metrics["activity"], dtype=np.int64)
    noise = np.array(metrics["noise_annotated"], dtype=np.float64) > 0.5
    reliable = np.array(metrics["rpeaks_reliable"], dtype=np.float64) > 0.5
    emg_flag = np.array(metrics["emg_flagged"], dtype=np.float64) > 0.5
    emg_ratio = np.array(metrics["emg_ratio"], dtype=np.float64)
    transition = np.array(metrics["context_transition"], dtype=np.float64) > 0.5
    patients = np.array(metrics["patient_id"], dtype=object)
    all_mask = np.ones_like(mai, dtype=bool)
    high_noise_patient = np.array([
        PAPER_NOISY_SIGNAL_PCT.get(str(p), 0.0) >= 20.0 for p in patients
    ], dtype=bool)
    emg_ratio_high = (
        emg_ratio >= np.nanpercentile(emg_ratio, 90)
        if len(emg_ratio) else all_mask & False
    )
    return {
        "all": all_mask,
        "low_risk_identity": (
            (mai < 0.02) & ~noise & reliable & ~emg_flag & ~transition
        ),
        "high_mai": mai >= 0.08,
        "very_high_mai": mai >= 0.15,
        "high_motion_activity": np.isin(activity, [3, 4, 5]),
        "context_transition": transition,
        "noise_annotated": noise,
        "rpeak_unreliable": ~reliable,
        "emg_flagged": emg_flag,
        "emg_ratio_top10pct": emg_ratio_high,
        "publication_high_noise_patient": high_noise_patient,
        "publication_high_noise_and_high_mai": high_noise_patient & (mai >= 0.08),
    }


def _bootstrap_ci(values: np.ndarray, n_boot: int, seed: int) -> tuple:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        sample = rng.choice(values, size=values.size, replace=True)
        boot[b] = float(np.mean(sample))
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def _cluster_bootstrap_ci(diff: np.ndarray, patients: np.ndarray,
                          n_boot: int, seed: int) -> tuple:
    """Patient-level cluster bootstrap over per-patient mean differences.

    The risk-enriched test set contains many windows but only four patients.
    Window-level bootstrap intervals are useful descriptively, but patient-level
    resampling is the conservative uncertainty estimate for manuscript claims.
    """
    diff = np.asarray(diff, dtype=np.float64)
    patients = np.asarray(patients, dtype=object)
    finite = np.isfinite(diff)
    diff = diff[finite]
    patients = patients[finite]
    unique = np.array(sorted({str(p) for p in patients}), dtype=object)
    if unique.size == 0:
        return 0, 0.0, 0.0, 1.0

    patient_means = []
    for p in unique:
        vals = diff[patients.astype(str) == str(p)]
        if vals.size:
            patient_means.append(float(np.mean(vals)))
    patient_means = np.asarray(patient_means, dtype=np.float64)
    n_patients = int(patient_means.size)
    if n_patients == 0:
        return 0, 0.0, 0.0, 1.0
    if n_patients == 1:
        return 1, float(patient_means[0]), float(patient_means[0]), 1.0

    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        sample = rng.choice(patient_means, size=n_patients, replace=True)
        boot[b] = float(np.mean(sample))

    try:
        p_val = float(sp_stats.wilcoxon(
            patient_means[patient_means != 0], zero_method="wilcox"
        ).pvalue)
    except ValueError:
        p_val = 1.0

    return (
        n_patients,
        float(np.percentile(boot, 2.5)),
        float(np.percentile(boot, 97.5)),
        p_val,
    )


def _wilcoxon_p(diff: np.ndarray) -> float:
    diff = np.asarray(diff, dtype=np.float64)
    diff = diff[np.isfinite(diff)]
    diff = diff[diff != 0]
    if diff.size < 2:
        return 1.0
    try:
        return float(sp_stats.wilcoxon(diff, zero_method="wilcox").pvalue)
    except ValueError:
        return 1.0


def paired_stats_table(teacher_metrics: dict, model_metrics: dict,
                       n_boot: int = 2000, seed: int = 123) -> list:
    n_teacher = len(teacher_metrics["delta_snr"])
    n_model = len(model_metrics["delta_snr"])
    if n_teacher != n_model:
        raise ValueError(
            f"Paired comparison requires equal window counts, got "
            f"{n_teacher} teacher and {n_model} model windows."
        )

    masks = _stratum_masks(teacher_metrics)
    all_patients = np.array(teacher_metrics["patient_id"], dtype=object)
    rows = []
    for stratum, mask in masks.items():
        mask = np.asarray(mask, dtype=bool)
        if not mask.any():
            continue
        for metric_name, spec in PAIRED_METRICS.items():
            teacher = _metric_array(teacher_metrics, metric_name)[mask]
            model = _metric_array(model_metrics, metric_name)[mask]
            patients = all_patients[mask]
            finite = np.isfinite(teacher) & np.isfinite(model)
            teacher = teacher[finite]
            model = model[finite]
            patients = patients[finite]
            if teacher.size == 0:
                continue
            diff = teacher - model
            ci_lo, ci_hi = _bootstrap_ci(
                diff, n_boot=n_boot,
                seed=seed + len(rows) * 17 + teacher.size,
            )
            n_patients, cl_lo, cl_hi, cl_p = _cluster_bootstrap_ci(
                diff, patients, n_boot=n_boot,
                seed=seed + len(rows) * 23 + teacher.size,
            )
            rows.append({
                "stratum": stratum,
                "metric": metric_name,
                "n": int(teacher.size),
                "teacher_mean": float(np.mean(teacher)),
                "synthetic_mean": float(np.mean(model)),
                "teacher_minus_synthetic": float(np.mean(diff)),
                "bootstrap_ci95_low": ci_lo,
                "bootstrap_ci95_high": ci_hi,
                "wilcoxon_p": _wilcoxon_p(diff),
                "cluster_n_patients": n_patients,
                "cluster_bootstrap_ci95_low": cl_lo,
                "cluster_bootstrap_ci95_high": cl_hi,
                "patient_wilcoxon_p": cl_p,
                "unit": spec["unit"],
                "higher_is_better": bool(spec["higher_is_better"]),
            })
    return rows


def run_paired_stats(args):
    ensure_output_dirs()
    if args.split == "both":
        raise ValueError("--paired_stats requires --split val or --split test")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    h5_path = OUTPUT_ROOT / f"{args.split}.h5"
    norm = load_norm_stats(OUTPUT_ROOT / "norm_stats.json")
    n_chunks = args.n_chunks or 100
    n_requested = n_chunks * 256
    with h5py.File(h5_path, "r") as f:
        n_requested = min(n_requested, int(f.attrs["n_windows"]))
    risk_indices = build_risk_sample_indices(h5_path, n_requested)

    print("\n" + "="*60)
    print(f"  Paired Stage 3 statistics - {args.split}")
    print("="*60)
    print(f"  Matched deterministic risk sample: {len(risk_indices):,} windows")

    print("\nEvaluating teacher_gated on matched windows...")
    teacher_metrics = evaluate(
        None, h5_path, norm, device, n_chunks=n_chunks,
        mode="teacher_gated", risk_sample=True, risk_indices=risk_indices,
    )

    print("\nEvaluating synthetic/model checkpoint on matched windows...")
    model = load_model(device, args)
    model_metrics = evaluate(
        model, h5_path, norm, device, n_chunks=n_chunks,
        mode="model", risk_sample=True, risk_indices=risk_indices,
    )

    rows = paired_stats_table(
        teacher_metrics, model_metrics,
        n_boot=args.bootstrap_iters,
        seed=args.bootstrap_seed,
    )

    out_json = LOG_DIR / f"stage3_paired_stats_{args.split}.json"
    out_csv = LOG_DIR / f"stage3_paired_stats_{args.split}.csv"
    payload = {
        "split": args.split,
        "n_windows": int(len(risk_indices)),
        "risk_sample_seed": 13,
        "bootstrap_iters": int(args.bootstrap_iters),
        "bootstrap_seed": int(args.bootstrap_seed),
        "comparison": "teacher_gated minus synthetic/model on matched windows",
        "rows": rows,
    }
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    with open(out_csv, "w", newline="") as f:
        fieldnames = [
            "stratum", "metric", "n", "teacher_mean", "synthetic_mean",
            "teacher_minus_synthetic", "bootstrap_ci95_low",
            "bootstrap_ci95_high", "wilcoxon_p", "cluster_n_patients",
            "cluster_bootstrap_ci95_low", "cluster_bootstrap_ci95_high",
            "patient_wilcoxon_p", "unit",
            "higher_is_better",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n  Paired stats JSON -> {out_json}")
    print(f"  Paired stats CSV  -> {out_csv}")
    print("\nKey rows:")
    for row in rows:
        if row["stratum"] in ("all", "high_mai") and row["metric"] in (
            "hf_noise_log_reduction", "artifact_log_reduction",
            "qrs_corr", "legacy_delta_snr",
        ):
            print(
                f"  {row['stratum']:<9} {row['metric']:<24} "
                f"diff={row['teacher_minus_synthetic']:+.4f} "
                f"CI=[{row['bootstrap_ci95_low']:+.4f}, "
                f"{row['bootstrap_ci95_high']:+.4f}] "
                f"p={row['wilcoxon_p']:.3g}"
            )


def run_paired_stats_transfer(args):
    """Paired statistics for the MIT-BIH/NSTDB cross-dataset transfer model.

    Evaluates teacher_gated, the CACHET synthetic-pair model, and the ECG-only
    transfer model on the SAME matched risk-sampled windows, then reports:
      * teacher_gated minus transfer   (primary)
      * synthetic     minus transfer   (secondary, skipped if synthetic ckpt missing)
      * teacher_gated minus synthetic  (reproducibility gate vs Table 6)

    Also prints the transfer model's aggregate QRS-corr / delta-SNR against the
    published values as an unambiguous reproduction gate.

    Output: logs/stage3_paired_stats_transfer_{split}.json / .csv
    Run with:  python evaluate_stage3.py --paired_stats_transfer --split test
    (do NOT pass --checkpoint; each model loads its own default checkpoint).
    """
    ensure_output_dirs()
    if args.split == "both":
        raise ValueError("--paired_stats_transfer requires --split val or --split test")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    h5_path = OUTPUT_ROOT / f"{args.split}.h5"
    norm = load_norm_stats(OUTPUT_ROOT / "norm_stats.json")
    n_chunks = args.n_chunks or 100
    n_requested = n_chunks * 256
    with h5py.File(h5_path, "r") as f:
        n_requested = min(n_requested, int(f.attrs["n_windows"]))
    risk_indices = build_risk_sample_indices(h5_path, n_requested)

    print("\n" + "=" * 60)
    print(f"  Paired transfer-model statistics - {args.split}")
    print("=" * 60)
    print(f"  Matched deterministic risk sample: {len(risk_indices):,} windows")

    print("\nEvaluating teacher_gated on matched windows...")
    teacher_metrics = evaluate(
        None, h5_path, norm, device, n_chunks=n_chunks,
        mode="teacher_gated", risk_sample=True, risk_indices=risk_indices,
    )

    print("\nEvaluating MIT-BIH/NSTDB transfer model on matched windows...")
    args.checkpoint = None  # force the transfer model's own default (mitbih_em_denoiser_best.pt)
    transfer = load_ecg_denoiser(device, args)
    transfer_metrics = evaluate(
        transfer, h5_path, norm, device, n_chunks=n_chunks,
        mode="ecg_denoiser", risk_sample=True, risk_indices=risk_indices,
    )

    synth_metrics = None
    try:
        print(f"\nEvaluating CACHET synthetic-pair model on matched windows "
              f"({Path(args.synth_checkpoint).name})...")
        args.checkpoint = args.synth_checkpoint  # explicit synthetic-pair checkpoint
        synth = load_model(device, args)
        synth_metrics = evaluate(
            synth, h5_path, norm, device, n_chunks=n_chunks,
            mode="model", risk_sample=True, risk_indices=risk_indices,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] synthetic-pair model unavailable ({exc}); "
              f"skipping synthetic comparisons.")

    def _relabel(rows, a_name, b_name):
        out = []
        for r in rows:
            out.append({
                "stratum": r["stratum"], "metric": r["metric"], "n": r["n"],
                f"{a_name}_mean": r["teacher_mean"],
                f"{b_name}_mean": r["synthetic_mean"],
                "difference": r["teacher_minus_synthetic"],
                "bootstrap_ci95_low": r["bootstrap_ci95_low"],
                "bootstrap_ci95_high": r["bootstrap_ci95_high"],
                "wilcoxon_p": r["wilcoxon_p"],
                "cluster_n_patients": r.get("cluster_n_patients", 0),
                "cluster_bootstrap_ci95_low": r.get("cluster_bootstrap_ci95_low", 0.0),
                "cluster_bootstrap_ci95_high": r.get("cluster_bootstrap_ci95_high", 0.0),
                "patient_wilcoxon_p": r.get("patient_wilcoxon_p", 1.0),
                "unit": r["unit"], "higher_is_better": r["higher_is_better"],
            })
        return out

    comparisons = {}
    comparisons["teacher_gated_minus_transfer"] = _relabel(
        paired_stats_table(teacher_metrics, transfer_metrics,
                           n_boot=args.bootstrap_iters, seed=args.bootstrap_seed),
        "teacher_gated", "transfer")
    if synth_metrics is not None:
        comparisons["synthetic_minus_transfer"] = _relabel(
            paired_stats_table(synth_metrics, transfer_metrics,
                               n_boot=args.bootstrap_iters, seed=args.bootstrap_seed),
            "synthetic", "transfer")
        comparisons["teacher_gated_minus_synthetic_GATE"] = _relabel(
            paired_stats_table(teacher_metrics, synth_metrics,
                               n_boot=args.bootstrap_iters, seed=args.bootstrap_seed),
            "teacher_gated", "synthetic")

    out_json = LOG_DIR / f"stage3_paired_stats_transfer_{args.split}.json"
    out_csv = LOG_DIR / f"stage3_paired_stats_transfer_{args.split}.csv"
    payload = {
        "split": args.split,
        "n_windows": int(len(risk_indices)),
        "risk_sample_seed": 13,
        "bootstrap_iters": int(args.bootstrap_iters),
        "bootstrap_seed": int(args.bootstrap_seed),
        "note": "Positive difference favors the first-named method (A) in each comparison.",
        "comparisons": comparisons,
    }
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["comparison", "stratum", "metric", "n",
                         "A_mean", "B_mean", "difference",
                         "bootstrap_ci95_low", "bootstrap_ci95_high",
                         "wilcoxon_p", "cluster_n_patients",
                         "cluster_bootstrap_ci95_low",
                         "cluster_bootstrap_ci95_high", "patient_wilcoxon_p",
                         "unit", "higher_is_better"])
        for comp_name, rows in comparisons.items():
            for r in rows:
                keys = list(r.keys())
                writer.writerow([comp_name, r["stratum"], r["metric"], r["n"],
                                 r[keys[3]], r[keys[4]], r["difference"],
                                 r["bootstrap_ci95_low"], r["bootstrap_ci95_high"],
                                 r["wilcoxon_p"], r.get("cluster_n_patients", 0),
                                 r.get("cluster_bootstrap_ci95_low", 0.0),
                                 r.get("cluster_bootstrap_ci95_high", 0.0),
                                 r.get("patient_wilcoxon_p", 1.0),
                                 r["unit"], r["higher_is_better"]])

    # --- reproduction gate: transfer-model aggregates vs published values ---
    tq = float(np.mean(np.asarray(transfer_metrics["qrs_corr"], dtype=np.float64)))
    td = float(np.mean(np.asarray(transfer_metrics["delta_snr"], dtype=np.float64)))
    print("\n  Reproduction gate (transfer model aggregates):")
    print(f"    QRS template corr : got {tq:.5f}  | published 0.98861")
    print(f"    legacy delta SNR  : got {td:+.4f}  | published +0.6877")

    print(f"\n  Transfer paired stats JSON -> {out_json}")
    print(f"  Transfer paired stats CSV  -> {out_csv}")
    key_metrics = ("artifact_log_reduction", "hf_noise_log_reduction",
                   "qrs_corr", "legacy_delta_snr")
    for comp_name, rows in comparisons.items():
        print(f"\n  [{comp_name}]  (positive favors the first method)")
        for r in rows:
            if r["stratum"] in ("all", "high_mai") and r["metric"] in key_metrics:
                print(f"    {r['stratum']:<9} {r['metric']:<24} "
                      f"diff={r['difference']:+.4f} "
                      f"CI=[{r['bootstrap_ci95_low']:+.4f}, {r['bootstrap_ci95_high']:+.4f}] "
                      f"p={r['wilcoxon_p']:.3g}")


def _sensitivity_summary(metrics: dict) -> dict:
    strata = stratified_results(metrics)
    dsnr = np.array(metrics["delta_snr"], dtype=np.float64)
    qc = np.array(metrics["qrs_corr"], dtype=np.float64)
    d_artifact = np.array(metrics["delta_artifact_log"], dtype=np.float64)
    d_hf = np.array(metrics["delta_hf_noise_log"], dtype=np.float64)
    residual_rms = np.array(metrics["residual_rms"], dtype=np.float64)
    high = strata["high_mai"]
    low = strata["low_risk_identity"]
    very_high = strata["very_high_mai"]
    return {
        "n_windows": int(len(dsnr)),
        "mean_legacy_delta_snr": float(np.mean(dsnr)),
        "mean_artifact_log_reduction": float(-np.mean(d_artifact)),
        "mean_hf_noise_log_reduction": float(-np.mean(d_hf)),
        "mean_qrs_corr": float(np.nanmean(qc)),
        "mean_residual_rms": float(np.mean(residual_rms)),
        "low_risk_n": low["n"],
        "low_risk_residual_rms": low["mean_residual_rms"],
        "low_risk_qrs_corr": low["mean_qrs_corr"],
        "high_mai_n": high["n"],
        "high_mai_artifact_log_reduction": high["artifact_log_reduction"],
        "high_mai_hf_noise_log_reduction": high["hf_noise_log_reduction"],
        "high_mai_qrs_corr": high["mean_qrs_corr"],
        "high_mai_residual_rms": high["mean_residual_rms"],
        "very_high_mai_n": very_high["n"],
        "very_high_mai_artifact_log_reduction": very_high["artifact_log_reduction"],
        "very_high_mai_hf_noise_log_reduction": very_high["hf_noise_log_reduction"],
        "very_high_mai_qrs_corr": very_high["mean_qrs_corr"],
        "very_high_mai_residual_rms": very_high["mean_residual_rms"],
    }


def _config_means(metrics: dict, metric_names, strata=("all", "high_mai", "emg_flagged")) -> dict:
    masks = _stratum_masks(metrics)
    out = {}
    for st in strata:
        m = np.asarray(masks[st], dtype=bool)
        row = {}
        for name in metric_names:
            arr = _metric_array(metrics, name)[m]
            arr = arr[np.isfinite(arr)]
            row[name] = float(np.mean(arr)) if arr.size else float("nan")
        row["n"] = int(m.sum())
        out[st] = row
    return out


def run_gated_baselines(args):
    """Gated-baseline comparison.

    Applies the two supervised models (MIT-BIH/NSTDB transfer and CACHET
    synthetic-pair) with the SAME deterministic MAI/noise gate used by the
    teacher method, so each supervised model acts only inside the gated windows
    and leaves low-risk windows unchanged. Evaluated on the same matched
    25,600-window risk sample as the main comparison, alongside an independent
    beat-detection endpoint (R-peak F1, sensitivity, R-R interval error).

    This isolates whether the supervised limitation is the learned residual map
    itself (gated versions still degrade) or its always-on application (gating
    recovers performance).

    Run with:  python evaluate_stage3.py --gated_baselines --split test
    (do NOT pass --checkpoint; each model loads its own default checkpoint).
    """
    ensure_output_dirs()
    if args.split == "both":
        raise ValueError("--gated_baselines requires --split val or --split test")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    h5_path = OUTPUT_ROOT / f"{args.split}.h5"
    norm = load_norm_stats(OUTPUT_ROOT / "norm_stats.json")
    n_chunks = args.n_chunks or 100
    n_requested = n_chunks * 256
    with h5py.File(h5_path, "r") as f:
        n_requested = min(n_requested, int(f.attrs["n_windows"]))
    risk_indices = build_risk_sample_indices(h5_path, n_requested)

    print("\n" + "=" * 60)
    print(f"  Gated-baseline comparison - {args.split}")
    print("=" * 60)
    print(f"  Matched deterministic risk sample: {len(risk_indices):,} windows")

    def _ev(model, mode, gate_baseline=False):
        return evaluate(model, h5_path, norm, device, n_chunks=n_chunks,
                        mode=mode, risk_sample=True, risk_indices=risk_indices,
                        gate_low=args.gate_low, gate_high=args.gate_high,
                        gate_baseline=gate_baseline)

    print("\n[1/5] teacher_gated ...")
    teacher = _ev(None, "teacher_gated")

    print("[2/5] transfer (MIT-BIH/NSTDB, ungated) ...")
    args.checkpoint = None
    transfer_model = load_ecg_denoiser(device, args)
    transfer = _ev(transfer_model, "ecg_denoiser", gate_baseline=False)

    print("[3/5] transfer + teacher gate ...")
    transfer_gated = _ev(transfer_model, "ecg_denoiser", gate_baseline=True)

    synth = synth_gated = None
    try:
        print(f"[4/5] synthetic-pair (ungated) [{Path(args.synth_checkpoint).name}] ...")
        args.checkpoint = args.synth_checkpoint
        synth_model = load_model(device, args)
        synth = _ev(synth_model, "model", gate_baseline=False)
        print("[5/5] synthetic-pair + teacher gate ...")
        synth_gated = _ev(synth_model, "model", gate_baseline=True)
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] synthetic-pair model unavailable ({exc}); skipping synthetic configs.")

    metric_names = ["artifact_log_reduction", "hf_noise_log_reduction",
                    "qrs_corr", "legacy_delta_snr", "residual_rms",
                    "beat_f1", "beat_sens", "rr_mae_ms"]
    configs = {"teacher_gated": teacher, "transfer": transfer,
               "transfer_gated": transfer_gated}
    if synth is not None:
        configs["synthetic"] = synth
        configs["synthetic_gated"] = synth_gated

    means = {name: _config_means(m, metric_names) for name, m in configs.items()}

    print("\n  Per-config means (overall / high_mai):")
    hdr = f"  {'config':<17}" + "".join(f"{k[:11]:>13}" for k in
            ["artifact", "hf_noise", "qrs_corr", "dSNR", "res_rms",
             "beat_f1", "beat_sens", "rr_mae_ms"])
    print(hdr)
    keymap = ["artifact_log_reduction", "hf_noise_log_reduction", "qrs_corr",
              "legacy_delta_snr", "residual_rms", "beat_f1", "beat_sens", "rr_mae_ms"]
    for st in ("all", "high_mai"):
        print(f"  --- stratum: {st} ---")
        for name, mm in means.items():
            vals = "".join(f"{mm[st][k]:>13.4f}" for k in keymap)
            print(f"  {name:<17}{vals}")

    # Paired contrasts on matched windows
    comparisons = {}
    def _rows(a, b):
        return paired_stats_table(a, b, n_boot=args.bootstrap_iters,
                                  seed=args.bootstrap_seed)
    comparisons["transfer_gated_minus_transfer"] = _rows(transfer_gated, transfer)
    comparisons["teacher_minus_transfer_gated"] = _rows(teacher, transfer_gated)
    if synth is not None:
        comparisons["synthetic_gated_minus_synthetic"] = _rows(synth_gated, synth)
        comparisons["teacher_minus_synthetic_gated"] = _rows(teacher, synth_gated)

    out_json = LOG_DIR / f"stage3_gated_baselines_{args.split}.json"
    out_csv = LOG_DIR / f"stage3_gated_baselines_{args.split}.csv"
    payload = {
        "split": args.split,
        "n_windows": int(len(risk_indices)),
        "risk_sample_seed": 13,
        "gate_low": float(args.gate_low),
        "gate_high": float(args.gate_high),
        "bootstrap_iters": int(args.bootstrap_iters),
        "bootstrap_seed": int(args.bootstrap_seed),
        "note": ("Supervised baselines gated with the teacher's MAI/noise gate. "
                 "Positive difference favors the first-named config."),
        "config_means": means,
        "comparisons": comparisons,
    }
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["comparison", "stratum", "metric", "n", "A_mean", "B_mean",
                    "A_minus_B", "ci95_low", "ci95_high", "wilcoxon_p", "unit",
                    "higher_is_better", "cluster_n_patients",
                    "cluster_ci95_low", "cluster_ci95_high", "patient_wilcoxon_p"])
        for comp, rows in comparisons.items():
            for r in rows:
                w.writerow([comp, r["stratum"], r["metric"], r["n"],
                            f"{r['teacher_mean']:.6f}", f"{r['synthetic_mean']:.6f}",
                            f"{r['teacher_minus_synthetic']:.6f}",
                            f"{r['bootstrap_ci95_low']:.6f}",
                            f"{r['bootstrap_ci95_high']:.6f}",
                            f"{r['wilcoxon_p']:.3g}", r["unit"],
                            r["higher_is_better"],
                            r.get("cluster_n_patients", 0),
                            f"{r.get('cluster_bootstrap_ci95_low', 0.0):.6f}",
                            f"{r.get('cluster_bootstrap_ci95_high', 0.0):.6f}",
                            f"{r.get('patient_wilcoxon_p', 1.0):.3g}"])

    print(f"\n  Gated-baseline JSON -> {out_json}")
    print(f"  Gated-baseline CSV  -> {out_csv}")
    print("\n  Key contrasts (all / high_mai; qrs_corr, artifact_log_reduction, beat_f1, rr_mae_ms):")
    for comp, rows in comparisons.items():
        print(f"  [{comp}]")
        for r in rows:
            if r["stratum"] in ("all", "high_mai") and r["metric"] in (
                    "qrs_corr", "artifact_log_reduction", "beat_f1", "rr_mae_ms"):
                print(f"    {r['stratum']:<9} {r['metric']:<22} "
                      f"diff={r['teacher_minus_synthetic']:+.4f} "
                      f"CI=[{r['bootstrap_ci95_low']:+.4f}, {r['bootstrap_ci95_high']:+.4f}] "
                      f"p={r['wilcoxon_p']:.3g}")


def run_synthetic_gated_baseline(args):
    """Dedicated CACHET-synthetic gated-baseline comparison.

    Evaluates the CACHET synthetic-pair Stage 3 model on the matched
    risk-enriched CACHET windows both with and without the deterministic
    teacher gate, then compares the teacher-gated framework against the gated
    synthetic model. This produces a focused output file for reporting the
    missing CACHET-synthetic gated decomposition in the manuscript/supplement.

    Run with:
        python evaluate_stage3.py --synthetic_gated_baseline --split test --risk_sample --n_chunks 100
    """
    ensure_output_dirs()
    if args.split == "both":
        raise ValueError("--synthetic_gated_baseline requires --split val or --split test")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    h5_path = OUTPUT_ROOT / f"{args.split}.h5"
    norm = load_norm_stats(OUTPUT_ROOT / "norm_stats.json")
    n_chunks = args.n_chunks or 100
    n_requested = n_chunks * 256
    with h5py.File(h5_path, "r") as f:
        n_requested = min(n_requested, int(f.attrs["n_windows"]))
    risk_indices = build_risk_sample_indices(h5_path, n_requested)

    print("\n" + "=" * 60)
    print(f"  CACHET-synthetic gated-baseline comparison - {args.split}")
    print("=" * 60)
    print(f"  Matched deterministic risk sample: {len(risk_indices):,} windows")
    print(f"  Synthetic checkpoint: {args.synth_checkpoint}")

    def _ev(model, mode, gate_baseline=False):
        return evaluate(
            model, h5_path, norm, device, n_chunks=n_chunks,
            mode=mode, risk_sample=True, risk_indices=risk_indices,
            gate_low=args.gate_low, gate_high=args.gate_high,
            gate_baseline=gate_baseline,
        )

    print("\n[1/3] teacher_gated ...")
    teacher = _ev(None, "teacher_gated")

    print("[2/3] CACHET synthetic-pair model (ungated) ...")
    old_checkpoint = args.checkpoint
    args.checkpoint = args.synth_checkpoint
    synth_model = load_model(device, args)
    args.checkpoint = old_checkpoint
    synth = _ev(synth_model, "model", gate_baseline=False)

    print("[3/3] CACHET synthetic-pair model + teacher gate ...")
    synth_gated = _ev(synth_model, "model", gate_baseline=True)

    metric_names = [
        "artifact_log_reduction", "hf_noise_log_reduction", "qrs_corr",
        "legacy_delta_snr", "qrs_motion_delta_snr", "residual_rms",
        "beat_f1", "beat_sens", "rr_mae_ms",
    ]
    configs = {
        "teacher_gated": teacher,
        "synthetic": synth,
        "synthetic_gated": synth_gated,
    }
    means = {name: _config_means(m, metric_names) for name, m in configs.items()}

    comparisons = {
        "synthetic_gated_minus_synthetic": paired_stats_table(
            synth_gated, synth, n_boot=args.bootstrap_iters, seed=args.bootstrap_seed
        ),
        "teacher_minus_synthetic_gated": paired_stats_table(
            teacher, synth_gated, n_boot=args.bootstrap_iters, seed=args.bootstrap_seed
        ),
    }

    out_json = LOG_DIR / f"stage3_synthetic_gated_baseline_{args.split}.json"
    out_csv = LOG_DIR / f"stage3_synthetic_gated_baseline_{args.split}.csv"
    payload = {
        "split": args.split,
        "n_windows": int(len(risk_indices)),
        "risk_sample_seed": 13,
        "gate_low": float(args.gate_low),
        "gate_high": float(args.gate_high),
        "bootstrap_iters": int(args.bootstrap_iters),
        "bootstrap_seed": int(args.bootstrap_seed),
        "synth_checkpoint": str(args.synth_checkpoint),
        "note": (
            "Dedicated CACHET synthetic-pair gated-baseline analysis. "
            "Positive paired differences favour the first-named config."
        ),
        "config_means": means,
        "comparisons": comparisons,
    }
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "comparison", "stratum", "metric", "n", "A_mean", "B_mean",
            "A_minus_B", "ci95_low", "ci95_high", "wilcoxon_p", "unit",
            "higher_is_better", "cluster_n_patients",
            "cluster_ci95_low", "cluster_ci95_high", "patient_wilcoxon_p",
        ])
        for comp, rows in comparisons.items():
            for r in rows:
                w.writerow([
                    comp, r["stratum"], r["metric"], r["n"],
                    f"{r['teacher_mean']:.6f}",
                    f"{r['synthetic_mean']:.6f}",
                    f"{r['teacher_minus_synthetic']:.6f}",
                    f"{r['bootstrap_ci95_low']:.6f}",
                    f"{r['bootstrap_ci95_high']:.6f}",
                    f"{r['wilcoxon_p']:.3g}",
                    r["unit"], r["higher_is_better"],
                    r.get("cluster_n_patients", 0),
                    f"{r.get('cluster_bootstrap_ci95_low', 0.0):.6f}",
                    f"{r.get('cluster_bootstrap_ci95_high', 0.0):.6f}",
                    f"{r.get('patient_wilcoxon_p', 1.0):.3g}",
                ])

    print(f"\n  Synthetic gated-baseline JSON -> {out_json}")
    print(f"  Synthetic gated-baseline CSV  -> {out_csv}")
    print("\n  Key contrasts (all / high_mai):")
    for comp, rows in comparisons.items():
        print(f"  [{comp}]")
        for r in rows:
            if r["stratum"] in ("all", "high_mai") and r["metric"] in (
                    "qrs_corr", "artifact_log_reduction",
                    "qrs_motion_delta_snr", "beat_f1", "rr_mae_ms"):
                print(
                    f"    {r['stratum']:<9} {r['metric']:<22} "
                    f"diff={r['teacher_minus_synthetic']:+.4f} "
                    f"CI=[{r['bootstrap_ci95_low']:+.4f}, {r['bootstrap_ci95_high']:+.4f}] "
                    f"p={r['wilcoxon_p']:.3g}"
                )


def run_gate_sensitivity(args):
    ensure_output_dirs()
    if args.split == "both":
        raise ValueError("--gate_sensitivity requires --split val or --split test")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    h5_path = OUTPUT_ROOT / f"{args.split}.h5"
    norm = load_norm_stats(OUTPUT_ROOT / "norm_stats.json")
    n_chunks = args.n_chunks or 100
    n_requested = n_chunks * 256
    with h5py.File(h5_path, "r") as f:
        n_requested = min(n_requested, int(f.attrs["n_windows"]))
    risk_indices = build_risk_sample_indices(h5_path, n_requested)

    rows = []
    print("\n" + "="*60)
    print(f"  Teacher-gated MAI gate sensitivity - {args.split}")
    print("="*60)
    print(f"  Matched deterministic risk sample: {len(risk_indices):,} windows")
    print(f"  Gate low endpoint fixed at {args.gate_low:.3f}g")

    for high in args.gate_high_values:
        high = float(high)
        if high <= args.gate_low:
            raise ValueError(
                f"Gate high endpoint must exceed gate_low={args.gate_low}, got {high}"
            )
        print(f"\nEvaluating gate_high={high:.3f}g...")
        metrics = evaluate(
            None, h5_path, norm, device, n_chunks=n_chunks,
            mode="teacher_gated", risk_sample=True, risk_indices=risk_indices,
            gate_low=args.gate_low, gate_high=high,
        )
        row = {
            "split": args.split,
            "gate_low": float(args.gate_low),
            "gate_high": high,
        }
        row.update(_sensitivity_summary(metrics))
        rows.append(row)

    out_json = LOG_DIR / f"stage3_gate_sensitivity_{args.split}.json"
    out_csv = LOG_DIR / f"stage3_gate_sensitivity_{args.split}.csv"
    payload = {
        "split": args.split,
        "risk_sample_seed": 13,
        "n_windows": int(len(risk_indices)),
        "gate_low": float(args.gate_low),
        "gate_high_values": [float(x) for x in args.gate_high_values],
        "rows": rows,
    }
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    with open(out_csv, "w", newline="") as f:
        fieldnames = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n  Sensitivity JSON -> {out_json}")
    print(f"  Sensitivity CSV  -> {out_csv}")
    print("\nSummary:")
    for row in rows:
        print(
            f"  high={row['gate_high']:.3f}g  "
            f"high-MAI HF={row['high_mai_hf_noise_log_reduction']:+.4f}  "
            f"low-risk RMS={row['low_risk_residual_rms']:.6f}  "
            f"QRS={row['mean_qrs_corr']:.5f}"
        )


# =============================================================================
# Main
# =============================================================================
def main(args):
    ensure_output_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    norm     = load_norm_stats(OUTPUT_ROOT / "norm_stats.json")
    h5_path  = OUTPUT_ROOT / f"{args.split}.h5"
    n_chunks = args.n_chunks or 100   # 100 × 256 = 25,600 windows

    mode = "teacher_ungated" if args.teacher_baseline else args.mode
    if mode == "model":
        model = load_model(device, args)
    elif mode == "ecg_denoiser":
        model = load_ecg_denoiser(device, args)
    else:
        model = None

    print(f"\nMode: {mode}")
    print(f"Evaluating on {args.split} split "
          f"({n_chunks} chunks × 256 windows = up to "
          f"{n_chunks*256:,} windows)...")

    metrics = evaluate(
        model, h5_path, norm, device, n_chunks=n_chunks,
        mode=mode, risk_sample=args.risk_sample,
        gate_low=args.gate_low, gate_high=args.gate_high,
    )

    # ── Aggregate results ─────────────────────────────────────────────────────
    dsnr = np.array(metrics["delta_snr"])
    dqmsnr = np.array(metrics["delta_qrs_motion_snr"])
    mai  = np.array(metrics["mai"])
    qc   = np.array(metrics["qrs_corr"])
    residual_rms = np.array(metrics["residual_rms"])
    imu_effect = np.array(metrics["imu_effect"])
    delta_abs = np.array(metrics["delta_abs"])
    d_baseline = np.array(metrics["delta_baseline_log"])
    d_hf_noise = np.array(metrics["delta_hf_noise_log"])
    d_artifact = np.array(metrics["delta_artifact_log"])
    d_qrs_power = np.array(metrics["delta_qrs_log"])

    high_mask = mai > 0.080
    low_mask  = mai < 0.020

    results = {
        "split":               args.split,
        "mode":                mode,
        "n_windows":           len(dsnr),
        "mean_delta_snr":      float(dsnr.mean()),
        "median_delta_snr":    float(np.median(dsnr)),
        "q25_delta_snr":       float(np.percentile(dsnr, 25)),
        "q75_delta_snr":       float(np.percentile(dsnr, 75)),
        "iqr_delta_snr":       float(np.percentile(dsnr, 75) - np.percentile(dsnr, 25)),
        "high_mai_delta_snr":  float(dsnr[high_mask].mean()) if high_mask.any() else 0.0,
        "low_mai_delta_snr":   float(dsnr[low_mask].mean())  if low_mask.any()  else 0.0,
        "mean_delta_qrs_motion_snr": float(dqmsnr.mean()),
        "high_mai_delta_qrs_motion_snr": float(dqmsnr[high_mask].mean()) if high_mask.any() else 0.0,
        "low_mai_delta_qrs_motion_snr": float(dqmsnr[low_mask].mean()) if low_mask.any() else 0.0,
        "mean_artifact_log_reduction": float(-d_artifact.mean()),
        "high_mai_artifact_log_reduction": float(-d_artifact[high_mask].mean()) if high_mask.any() else 0.0,
        "low_mai_artifact_log_reduction": float(-d_artifact[low_mask].mean()) if low_mask.any() else 0.0,
        "mean_baseline_log_reduction": float(-d_baseline.mean()),
        "mean_hf_noise_log_reduction": float(-d_hf_noise.mean()),
        "mean_qrs_log_change": float(d_qrs_power.mean()),
        "mean_qrs_corr":       float(np.nanmean(qc)),
        "mean_residual_rms":   float(residual_rms.mean()),
        "mean_imu_effect":     float(imu_effect.mean()),
        "low_mai_delta_abs":   float(delta_abs[low_mask].mean()) if low_mask.any() else 0.0,
        "high_mai_residual_rms": float(residual_rms[high_mask].mean()) if high_mask.any() else 0.0,
        "low_mai_residual_rms": float(residual_rms[low_mask].mean()) if low_mask.any() else 0.0,
        "pct_improved":        float((dsnr > 0).mean() * 100),
        "snr_in_mean":         float(np.array(metrics["snr_in"]).mean()),
        "snr_out_mean":        float(np.array(metrics["snr_out"]).mean()),
        "qrs_motion_snr_in_mean": float(np.array(metrics["qrs_motion_snr_in"]).mean()),
        "qrs_motion_snr_out_mean": float(np.array(metrics["qrs_motion_snr_out"]).mean()),
    }
    results["risk_strata"] = stratified_results(metrics)

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*52}")
    print(f"  Stage 3 Evaluation Results — {args.split}")
    print(f"{'='*52}")
    for k, v in results.items():
        if k in ("split", "mode", "n_windows", "risk_strata"):
            continue
        print(f"  {k:<30}: {v:>10.4f}")
    print(f"  {'n_windows':<30}: {results['n_windows']:>10,}")
    print(f"{'='*52}")

    # ── Interpretation ────────────────────────────────────────────────────────
    print_strata_table(results["risk_strata"])

    print("\nContext-aware interpretation:")
    d = results["mean_delta_snr"]
    print(
        f"  LEGACY mean delta_snr={d:+.4f} "
        "(informational only; not the Stage 3 success criterion)"
    )

    qd = results["mean_delta_qrs_motion_snr"]
    hqd = results["high_mai_delta_qrs_motion_snr"]
    print(f"  QRS/motion dSNR={qd:+.4f}  high-MAI={hqd:+.4f}")

    ar = results["mean_artifact_log_reduction"]
    har = results["high_mai_artifact_log_reduction"]
    lar = results["low_mai_artifact_log_reduction"]
    low_rms = results["low_mai_residual_rms"]
    qc_mean = results["mean_qrs_corr"]

    if low_rms <= 1e-4:
        print(f"  PASS low-risk identity preserved: low-MAI residual RMS={low_rms:.4f}")
    elif low_rms <= 1e-3:
        print(f"  WARN low-risk windows changed slightly: low-MAI residual RMS={low_rms:.4f}")
    else:
        print(f"  FAIL low-risk ECG changed: low-MAI residual RMS={low_rms:.4f}")

    if har > max(lar + 0.01, 0.01):
        print(
            f"  PASS context-gated artifact reduction: "
            f"high-MAI ({har:+.4f}) > low-MAI ({lar:+.4f})"
        )
    elif har > 0:
        print(
            f"  WARN artifact reduction present but weakly selective: "
            f"high-MAI ({har:+.4f}), low-MAI ({lar:+.4f})"
        )
    else:
        print(f"  FAIL high-risk artifact bands not reduced: high-MAI={har:+.4f}")

    if ar > 0.01:
        print(f"  PASS target artifact bands reduced overall: log reduction={ar:+.4f}")
    elif ar > 0:
        print(f"  WARN target artifact bands slightly reduced: log reduction={ar:+.4f}")
    else:
        print(f"  FAIL target artifact bands not reduced: log reduction={ar:+.4f}")

    if qc_mean > 0.99:
        print(f"  PASS QRS corr={qc_mean:.4f} - morphology well preserved")
    elif qc_mean > 0.95:
        print(f"  WARN QRS corr={qc_mean:.4f} - minor morphology distortion")
    else:
        print(f"  FAIL QRS corr={qc_mean:.4f} - significant morphology distortion")

    if low_rms <= 1e-4 and har > max(lar + 0.01, 0.01) and qc_mean > 0.99:
        print("  OVERALL: PASS conservative context-gated artifact management")
    else:
        print("  OVERALL: REVIEW risk-gated behavior before using as official baseline")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    suffix = f"{args.split}_{mode}"
    json_path = LOG_DIR / f"stage3_eval_{suffix}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results -> {json_path}")

    # ── Save per-window metric arrays for future distribution queries ─────────
    npz_path = LOG_DIR / f"stage3_per_window_{suffix}.npz"
    np.savez_compressed(npz_path, **{k: np.asarray(v) for k, v in metrics.items()
                                     if k != "patient_id"})
    np.save(LOG_DIR / f"stage3_per_window_{suffix}_patients.npy",
            np.array(metrics["patient_id"], dtype=object))
    print(f"  Per-window arrays -> {npz_path}")

    # ── Before/after signal plot ───────────────────────────────────────────────
    sig_plot = LOG_DIR / f"stage3_before_after_{suffix}.png"
    plot_before_after(model, h5_path, norm, device,
                      sig_plot, n_examples=4,
                      mode=mode,
                      gate_low=args.gate_low,
                      gate_high=args.gate_high)

    # ── Metrics summary plot ───────────────────────────────────────────────────
    met_plot = LOG_DIR / f"stage3_metrics_{suffix}.png"
    plot_metrics(metrics, results, args.split, met_plot)
    return results


def run_ablation(args):
    modes = ["identity", "teacher_gated", "model"]
    original_mode = args.mode
    rows = []
    print("\n" + "="*52)
    print(f"  Stage 3 ablation table — {args.split}")
    print("="*52)
    for mode in modes:
        args.mode = mode
        args.teacher_baseline = False
        rows.append(main(args))
    args.mode = original_mode

    out_json = LOG_DIR / f"stage3_ablation_{args.split}.json"
    out_csv = LOG_DIR / f"stage3_ablation_{args.split}.csv"
    with open(out_json, "w") as f:
        json.dump(rows, f, indent=2)
    keep = [
        "mode",
        "n_windows",
        "high_mai_artifact_log_reduction",
        "low_mai_artifact_log_reduction",
        "low_mai_residual_rms",
        "high_mai_residual_rms",
        "mean_qrs_corr",
        "mean_qrs_log_change",
        "mean_delta_qrs_motion_snr",
        "high_mai_delta_qrs_motion_snr",
        "mean_delta_snr",
    ]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keep, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  Ablation JSON -> {out_json}")
    print(f"  Ablation CSV  -> {out_csv}")


def run_frequency_ablation(args):
    modes = [
        "teacher_gated",
        "teacher_inband_gated",
        "teacher_broadband_gated",
        "teacher_ungated",
        "teacher_inband_ungated",
        "teacher_broadband_ungated",
    ]
    original_mode = args.mode
    original_risk = args.risk_sample
    args.risk_sample = True
    rows = []
    print("\n" + "="*60)
    print(f"  Stage 3 pseudo-target frequency ablation - {args.split}")
    print("="*60)
    for mode in modes:
        args.mode = mode
        args.teacher_baseline = False
        rows.append(main(args))
    args.mode = original_mode
    args.risk_sample = original_risk

    out_json = LOG_DIR / f"stage3_frequency_ablation_{args.split}.json"
    out_csv = LOG_DIR / f"stage3_frequency_ablation_{args.split}.csv"
    with open(out_json, "w") as f:
        json.dump(rows, f, indent=2)
    keep = [
        "mode",
        "n_windows",
        "mean_artifact_log_reduction",
        "high_mai_artifact_log_reduction",
        "mean_hf_noise_log_reduction",
        "high_mai_residual_rms",
        "low_mai_residual_rms",
        "mean_qrs_corr",
        "mean_qrs_log_change",
        "mean_delta_qrs_motion_snr",
        "mean_delta_snr",
    ]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keep, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  Frequency ablation JSON -> {out_json}")
    print(f"  Frequency ablation CSV  -> {out_csv}")


# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split",    default="both",
                        choices=["train", "val", "test", "both"],
                        help="Split to evaluate. 'both' runs val then test.")
    parser.add_argument("--n_chunks", type=int, default=None,
                        help="Chunks to evaluate (default=100, "
                             "each chunk=256 windows)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Explicit Stage 3 checkpoint path to evaluate.")
    parser.add_argument("--use_last", action="store_true",
                        help="Evaluate denoiser_last.pt instead of denoiser_best.pt.")
    parser.add_argument("--mode", default="teacher_gated",
                        choices=["identity", "teacher_gated", "teacher_ungated",
                                 "teacher_inband_gated", "teacher_inband_ungated",
                                 "teacher_broadband_gated", "teacher_broadband_ungated",
                                 "model", "ecg_denoiser"],
                        help="Evaluation mode for Stage 3 ablations/inference baseline.")
    parser.add_argument("--ablation", action="store_true",
                        help="Run identity, teacher_gated, and model and save a comparison table.")
    parser.add_argument("--paired_stats", action="store_true",
                        help="Run paired teacher_gated vs model statistics on matched risk-sampled windows.")
    parser.add_argument("--paired_stats_transfer", action="store_true",
                        help="Run paired statistics involving the MIT-BIH/NSTDB transfer model "
                             "(teacher_gated vs transfer, synthetic vs transfer) on matched windows.")
    parser.add_argument("--synth_checkpoint", type=str,
                        default=str(CHECKPOINT_DIR / "denoiser_synth_best.pt"),
                        help="Synthetic-pair checkpoint used by --paired_stats_transfer "
                             "(default: checkpoints/denoiser_synth_best.pt).")
    parser.add_argument("--gated_baselines", action="store_true",
                        help="Run supervised baselines gated with the teacher MAI/noise gate.")
    parser.add_argument("--synthetic_gated_baseline", action="store_true",
                        help="Run only the CACHET-synthetic ungated/gated baseline decomposition.")
    parser.add_argument("--gate_sensitivity", action="store_true",
                        help="Run teacher_gated sensitivity over MAI gate high endpoints.")
    parser.add_argument("--frequency_ablation", action="store_true",
                        help="Run same-gate pseudo-target frequency-design ablations.")
    parser.add_argument("--gate_low", type=float, default=0.02,
                        help="Teacher-gated MAI ramp low endpoint in g.")
    parser.add_argument("--gate_high", type=float, default=0.12,
                        help="Teacher-gated MAI ramp high endpoint in g for normal evaluation.")
    parser.add_argument("--gate_high_values", type=float, nargs="+",
                        default=[0.06, 0.08, 0.10, 0.12],
                        help="MAI ramp high endpoints in g for --gate_sensitivity.")
    parser.add_argument("--bootstrap_iters", type=int, default=2000,
                        help="Bootstrap iterations for --paired_stats confidence intervals.")
    parser.add_argument("--bootstrap_seed", type=int, default=123,
                        help="Random seed for --paired_stats bootstrap resampling.")
    parser.add_argument("--risk_sample", action="store_true",
                        help="Evaluate a deterministic sample enriched for high-risk context windows.")
    parser.add_argument("--teacher_baseline", action="store_true",
                        help="Deprecated alias for --mode teacher_ungated.")
    parser.add_argument("--max_residual", type=float, default=STAGE3_MAX_RESIDUAL)
    args = parser.parse_args()

    if args.gated_baselines:
        run_gated_baselines(args)
    elif args.synthetic_gated_baseline:
        run_synthetic_gated_baseline(args)
    elif args.gate_sensitivity:
        run_gate_sensitivity(args)
    elif args.frequency_ablation:
        run_frequency_ablation(args)
    elif args.paired_stats_transfer:
        run_paired_stats_transfer(args)
    elif args.paired_stats:
        run_paired_stats(args)
    elif args.ablation:
        if args.split == "both":
            for split in ("val", "test"):
                args.split = split
                run_ablation(args)
        else:
            run_ablation(args)
    elif args.split == "both":
        print("\n" + "="*52)
        print("  Running evaluation on VAL split")
        print("="*52)
        args.split = "val"
        main(args)

        print("\n" + "="*52)
        print("  Running evaluation on TEST split")
        print("  (held-out — final unbiased results)")
        print("="*52)
        args.split = "test"
        main(args)
    else:
        main(args)
