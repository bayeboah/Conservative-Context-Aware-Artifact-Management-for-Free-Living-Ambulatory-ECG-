"""
train_stage3_synthetic.py
-------------------------
Stage 3B: supervised synthetic-pair training for the residual denoiser.

CACHET-CADB has no true clean/noisy ECG pairs. This script creates controlled
pairs inside each split:

  clean-ish ECG target + artifact residual from a risky donor = synthetic input

The donor's IMU/context is paired with the synthetic ECG, so the denoiser learns
to remove artifact patterns that are supported by motion/context.

Usage:
  python train_stage3_synthetic.py --epochs 5 --pairs_per_epoch 32768
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import (
    OUTPUT_ROOT, CHECKPOINT_DIR, LOG_DIR, ECG_FS, STAGE3_MAX_RESIDUAL,
    ensure_output_dirs,
)
from dataset import load_norm_stats
from denoiser import ResidualArtifactRemover, generate_pseudo_target


STAGE2_CKPT = CHECKPOINT_DIR / "siamese_best.pt"
SYNTH_BEST = CHECKPOINT_DIR / "denoiser_synth_best.pt"
SYNTH_LAST = CHECKPOINT_DIR / "denoiser_synth_last.pt"
SYNTH_HISTORY = LOG_DIR / "stage3_synthetic_history.json"


def set_all_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _decode_patient_ids(arr):
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in arr])


def _read_h5_selection(f: h5py.File, name: str, idx: np.ndarray):
    """h5py needs sorted fancy indices; restore original order after read."""
    idx = np.asarray(idx, dtype=np.int64)
    unique_idx, inverse = np.unique(idx, return_inverse=True)
    data = f[name][unique_idx]
    return data[inverse]


def build_candidate_indices(h5_path: Path, max_scan: int | None = None) -> Dict[str, np.ndarray]:
    """Return clean target and risky artifact donor indices for one split."""
    with h5py.File(h5_path, "r") as f:
        n = int(f.attrs["n_windows"])
        if max_scan is not None:
            n = min(n, int(max_scan))
        mai = f["movement_accel"][:n].astype(np.float32)
        noise = f["noise_annotated"][:n].astype(bool)
        reliable = f["rpeaks_reliable"][:n].astype(bool)
        emg_flag = f["emg_flagged"][:n].astype(bool)
        emg_ratio = f["emg_ratio"][:n].astype(np.float32)

    low_emg = emg_ratio <= np.percentile(emg_ratio, 50)
    high_emg = emg_ratio >= np.percentile(emg_ratio, 90)

    clean = np.where(
        (mai < 0.02)
        & (~noise)
        & reliable
        & (~emg_flag)
        & low_emg
    )[0]
    donor = np.where(
        (mai >= 0.08)
        | noise
        | emg_flag
        | (~reliable)
        | high_emg
    )[0]

    if len(clean) == 0:
        raise RuntimeError(f"No clean synthetic targets found in {h5_path}")
    if len(donor) == 0:
        raise RuntimeError(f"No artifact donors found in {h5_path}")
    return {"clean": clean.astype(np.int64), "donor": donor.astype(np.int64)}


def sample_batch_indices(candidates: Dict[str, np.ndarray], batch_size: int, rng) -> Tuple[np.ndarray, np.ndarray]:
    clean_idx = rng.choice(candidates["clean"], size=batch_size, replace=True)
    donor_idx = rng.choice(candidates["donor"], size=batch_size, replace=True)
    return clean_idx.astype(np.int64), donor_idx.astype(np.int64)


def load_pair_batch(
    h5_path: Path,
    norm: dict,
    clean_idx: np.ndarray,
    donor_idx: np.ndarray,
    device: torch.device,
    alpha_range: Tuple[float, float],
    rng,
) -> Dict[str, torch.Tensor]:
    ecg_mean = norm["ecg_mean"]
    ecg_std = norm["ecg_std"]
    imu_mean = np.array(norm["imu_mean"], dtype=np.float32)
    imu_std = np.array(norm["imu_std"], dtype=np.float32)

    with h5py.File(h5_path, "r") as f:
        clean_ecg = _read_h5_selection(f, "ecg", clean_idx).astype(np.float32)
        donor_ecg = _read_h5_selection(f, "ecg", donor_idx).astype(np.float32)
        acc = _read_h5_selection(f, "acc", donor_idx).astype(np.float32)
        gyro = _read_h5_selection(f, "gyro", donor_idx).astype(np.float32)
        mai = _read_h5_selection(f, "movement_accel", donor_idx).astype(np.float32)
        noise = _read_h5_selection(f, "noise_annotated", donor_idx).astype(np.float32)

    clean_ecg = (clean_ecg - ecg_mean) / ecg_std
    donor_ecg = (donor_ecg - ecg_mean) / ecg_std
    acc = (acc - imu_mean[:3, None]) / imu_std[:3, None]
    gyro = (gyro - imu_mean[3:, None]) / imu_std[3:, None]
    imu = np.concatenate([acc, gyro], axis=1)

    clean_t = torch.from_numpy(clean_ecg).to(device)
    donor_t = torch.from_numpy(donor_ecg).to(device)
    imu_t = torch.from_numpy(imu).to(device)
    mai_t = torch.from_numpy(mai).to(device)
    noise_t = torch.from_numpy(noise).to(device)

    with torch.no_grad():
        donor_pseudo = generate_pseudo_target(donor_t, fs=ECG_FS)
        artifact = donor_t - donor_pseudo
        artifact = artifact - artifact.mean(dim=-1, keepdim=True)
        clean_rms = clean_t.square().mean(dim=[1, 2], keepdim=True).sqrt().clamp_min(1e-4)
        art_rms = artifact.square().mean(dim=[1, 2], keepdim=True).sqrt().clamp_min(1e-4)
        artifact = artifact * (clean_rms / art_rms).clamp(0.05, 2.0)
        alpha = rng.uniform(alpha_range[0], alpha_range[1], size=(len(clean_idx), 1, 1)).astype(np.float32)
        alpha_t = torch.from_numpy(alpha).to(device)
        noisy_t = clean_t + alpha_t * artifact

    return {
        "noisy": noisy_t,
        "target": clean_t,
        "imu": imu_t,
        "mai": mai_t,
        "noise": noise_t,
    }


def supervised_synthetic_loss(
    clean_pred: torch.Tensor,
    target: torch.Tensor,
    residual: torch.Tensor,
    noisy: torch.Tensor,
    lambda_mse: float,
    lambda_qrs: float,
    lambda_residual: float,
    lambda_smooth: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    l_mse = F.smooth_l1_loss(clean_pred, target)

    pred_qrs = _rfft_log_band(clean_pred, 5.0, 20.0)
    target_qrs = _rfft_log_band(target, 5.0, 20.0)
    l_qrs = (pred_qrs - target_qrs).abs().mean()

    true_residual = noisy - target
    l_residual = F.smooth_l1_loss(residual, true_residual)
    l_smooth = (residual[..., 1:] - residual[..., :-1]).abs().mean()

    total = (
        lambda_mse * l_mse
        + lambda_qrs * l_qrs
        + lambda_residual * l_residual
        + lambda_smooth * l_smooth
    )
    return total, {
        "mse": float(l_mse.item()),
        "qrs": float(l_qrs.item()),
        "residual": float(l_residual.item()),
        "smooth": float(l_smooth.item()),
        "residual_rms": float(residual.square().mean().sqrt().item()),
    }


def _rfft_log_band(x: torch.Tensor, flo: float, fhi: float) -> torch.Tensor:
    sig = x.float().squeeze(1)
    n = sig.shape[-1]
    freqs = torch.fft.rfftfreq(n, d=1.0 / ECG_FS).to(x.device)
    spec = torch.fft.rfft(sig, dim=-1).abs().clamp_min(1e-6).log()
    mask = (freqs >= flo) & (freqs <= fhi)
    return spec[:, mask].mean(dim=1)


def count_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return trainable, frozen


def run_epoch(model, h5_path, norm, candidates, args, device, optimiser=None, scaler=None, seed=0):
    training = optimiser is not None
    model.train(training)
    model.imu_encoder.eval()
    rng = np.random.default_rng(seed)
    n_steps = max(1, args.pairs_per_epoch // args.batch)
    losses = []
    infos = []
    iterator = range(n_steps)
    if training:
        iterator = tqdm(iterator, desc="synthetic train", leave=False)

    for _ in iterator:
        clean_idx, donor_idx = sample_batch_indices(candidates, args.batch, rng)
        batch = load_pair_batch(
            h5_path, norm, clean_idx, donor_idx, device,
            (args.alpha_min, args.alpha_max), rng,
        )
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            pred, residual = model(
                batch["noisy"], batch["imu"],
                mai=batch["mai"], noise_annotated=batch["noise"],
                return_residual=True,
            )
            loss, info = supervised_synthetic_loss(
                pred, batch["target"], residual, batch["noisy"],
                args.lambda_mse, args.lambda_qrs,
                args.lambda_residual, args.lambda_smooth,
            )

        if training:
            optimiser.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            scaler.step(optimiser)
            scaler.update()

        losses.append(float(loss.item()))
        infos.append(info)

    mean_info = {
        k: float(np.mean([x[k] for x in infos])) for k in infos[0]
    }
    return float(np.mean(losses)), mean_info


def main(args):
    ensure_output_dirs()
    set_all_seeds(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    norm = load_norm_stats(OUTPUT_ROOT / "norm_stats.json")
    train_h5 = OUTPUT_ROOT / "train.h5"
    val_h5 = OUTPUT_ROOT / "val.h5"

    print("Building synthetic-pair candidate pools...")
    train_candidates = build_candidate_indices(train_h5, args.max_scan)
    val_candidates = build_candidate_indices(val_h5, args.max_scan)
    print(f"  train clean={len(train_candidates['clean']):,} donors={len(train_candidates['donor']):,}")
    print(f"  val   clean={len(val_candidates['clean']):,} donors={len(val_candidates['donor']):,}")

    model = ResidualArtifactRemover(max_residual=args.max_residual).to(device)
    model.load_imu_encoder(str(STAGE2_CKPT), device)
    n_trainable, n_frozen = count_params(model)
    print(f"Model trainable={n_trainable:,} frozen={n_frozen:,}")
    print(f"Seed: {args.seed}")

    optimiser = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_val = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_info = run_epoch(
            model, train_h5, norm, train_candidates, args, device,
            optimiser=optimiser, scaler=scaler, seed=args.seed + epoch,
        )
        with torch.no_grad():
            val_loss, val_info = run_epoch(
                model, val_h5, norm, val_candidates, args, device,
                optimiser=None, scaler=None, seed=args.seed + 10_000 + epoch,
            )
        elapsed = time.time() - t0
        row = {
            "epoch": epoch,
            "train": train_loss,
            "val": val_loss,
            "train_info": train_info,
            "val_info": val_info,
        }
        history.append(row)
        print(
            f"Epoch {epoch:3d}/{args.epochs} "
            f"train={train_loss:.4f} val={val_loss:.4f} "
            f"mse={val_info['mse']:.4f} qrs={val_info['qrs']:.4f} "
            f"res={val_info['residual']:.4f} rms={val_info['residual_rms']:.4f} "
            f"{elapsed:.1f}s"
        )

        ckpt = {
            "epoch": epoch,
            "val_loss": val_loss,
            "seed": int(args.seed),
            "argv": sys.argv,
            "model_state_dict": model.state_dict(),
            "optimiser_state_dict": optimiser.state_dict(),
            "history": history,
            "synthetic_config": vars(args),
        }
        torch.save(ckpt, SYNTH_LAST)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, SYNTH_BEST)
            print(f"  New best synthetic val={best_val:.4f} -> {SYNTH_BEST.name}")

    with open(SYNTH_HISTORY, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"History -> {SYNTH_HISTORY}")
    print(f"Best checkpoint -> {SYNTH_BEST}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Stage 3 on synthetic clean/noisy pairs.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--pairs_per_epoch", type=int, default=32768)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_residual", type=float, default=STAGE3_MAX_RESIDUAL)
    parser.add_argument("--alpha_min", type=float, default=0.25)
    parser.add_argument("--alpha_max", type=float, default=1.0)
    parser.add_argument("--lambda_mse", type=float, default=1.0)
    parser.add_argument("--lambda_qrs", type=float, default=0.5)
    parser.add_argument("--lambda_residual", type=float, default=1.0)
    parser.add_argument("--lambda_smooth", type=float, default=0.05)
    parser.add_argument("--max_scan", type=int, default=None,
                        help="Optional cap while building candidate pools.")
    parser.add_argument("--seed", type=int, default=123)
    main(parser.parse_args())
