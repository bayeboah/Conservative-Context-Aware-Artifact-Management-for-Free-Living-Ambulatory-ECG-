"""
train_stage3.py — Stage 3: Cross-Attention U-Net ECG Denoiser Training
=======================================================================
Self-supervised denoising:
  Input:  noisy ECG + Gaussian augmentation noise (σ=0.1)
  Target: original Stage 1 preprocessed ECG (without added noise)
  Guide:  IMU signal via frozen Stage 2 IMU encoder

The model learns to remove Gaussian noise + residual motion artifact
using the IMU feature map as a frame-by-frame motion indicator.

Usage:
  python train_stage3.py
  python train_stage3.py --resume checkpoints/denoiser_last.pt
"""

import argparse, json, os, random, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import (
    OUTPUT_ROOT, CHECKPOINT_DIR, LOG_DIR,
    RANDOM_SEED,
    ECG_FS, BATCH_SIZE, NUM_WORKERS,
    LEARNING_RATE, MAX_EPOCHS, EARLY_STOP_PATIENCE,
    STAGE3_MAX_RESIDUAL, STAGE3_LAMBDA_IDENTITY,
    STAGE3_LAMBDA_RESIDUAL, STAGE3_LAMBDA_SMOOTH,
    STAGE3_LAMBDA_QRS, STAGE3_LAMBDA_MOTION,
    ensure_output_dirs,
)
from dataset import load_norm_stats
from denoiser import ResidualArtifactRemover, residual_artifact_loss


def set_all_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def safe_torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)

# ── Hyperparameters ───────────────────────────────────────────────────────────
STAGE2_CKPT  = CHECKPOINT_DIR / "siamese_best.pt"
STAGE3_BEST  = CHECKPOINT_DIR / "denoiser_best.pt"
STAGE3_LAST  = CHECKPOINT_DIR / "denoiser_last.pt"
HISTORY_FILE = LOG_DIR / "stage3_history.json"

LAMBDA_IDENTITY = STAGE3_LAMBDA_IDENTITY
LAMBDA_RESIDUAL = STAGE3_LAMBDA_RESIDUAL
LAMBDA_SMOOTH   = STAGE3_LAMBDA_SMOOTH
LAMBDA_QRS      = STAGE3_LAMBDA_QRS
LAMBDA_MOTION   = STAGE3_LAMBDA_MOTION
MAX_RESIDUAL    = STAGE3_MAX_RESIDUAL


# =============================================================================
# Dataset
# =============================================================================
def chunked_loader(h5_path: Path, norm: dict,
                   chunk: int = 2048,
                   max_windows: int = None,
                   shuffle: bool = True):
    """
    Fast HDF5 chunked reader — replaces per-window DataLoader.
    Reads `chunk` windows at once → ~2000× fewer HDF5 ops per epoch.
    With max_windows, subsamples the dataset for faster epochs.
    """
    import h5py
    ecg_mean = norm['ecg_mean']
    ecg_std  = norm['ecg_std']
    imu_mean = np.array(norm['imu_mean'], dtype=np.float32)
    imu_std  = np.array(norm['imu_std'],  dtype=np.float32)

    with h5py.File(h5_path, 'r') as f:
        n_total = int(f.attrs['n_windows'])

    # Build index — optionally subsample
    indices = np.arange(n_total)
    if shuffle:
        np.random.shuffle(indices)
    if max_windows and max_windows < n_total:
        indices = indices[:max_windows]
    indices.sort()   # sort for sequential HDF5 access (faster)

    with h5py.File(h5_path, 'r') as f:
        for start in range(0, len(indices), chunk):
            idx  = indices[start: start + chunk]
            ecg  = f['ecg'][idx].astype(np.float32)
            acc  = f['acc'][idx].astype(np.float32)
            gyro = f['gyro'][idx].astype(np.float32)
            lw   = f['loss_weight'][idx].astype(np.float32)
            mai  = f['movement_accel'][idx].astype(np.float32)
            noise = f['noise_annotated'][idx].astype(np.float32)
            rel   = f['rpeaks_reliable'][idx].astype(np.float32)

            ecg  = (ecg  - ecg_mean) / ecg_std
            acc  = (acc  - imu_mean[:3, None]) / imu_std[:3, None]
            gyro = (gyro - imu_mean[3:, None]) / imu_std[3:, None]
            imu  = np.concatenate([acc, gyro], axis=1)  # [B, 6, 640]

            yield {
                'ecg':         torch.from_numpy(ecg),
                'imu':         torch.from_numpy(imu),
                'loss_weight': torch.from_numpy(lw),
                'movement_accel': torch.from_numpy(mai),
                'noise_annotated': torch.from_numpy(noise),
                'rpeaks_reliable': torch.from_numpy(rel),
            }


# =============================================================================
# Training utilities
# =============================================================================
def n_windows_in(h5_path: Path) -> int:
    import h5py
    with h5py.File(h5_path, 'r') as f:
        return int(f.attrs['n_windows'])


def count_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return trainable, frozen


# =============================================================================
# Main
# =============================================================================
def main(args):
    ensure_output_dirs()
    set_all_seeds(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    # ── Hyperparameters ───────────────────────────────────────────────────────
    batch_size  = args.batch   or 64
    lr          = args.lr      or 1e-4
    max_epochs  = args.epochs  or MAX_EPOCHS
    patience    = 15
    n_workers   = 0            # 0 avoids Windows multiprocessing issues

    print(f"\nHyperparameters:")
    print(f"  seed        : {args.seed}")
    print(f"  batch_size  : {batch_size}")
    print(f"  lr          : {lr:.0e}")
    print(f"  max_epochs  : {max_epochs}")
    print(f"  patience    : {patience}")
    print(f"  lambda_id   : {args.lambda_identity}")
    print(f"  lambda_res  : {args.lambda_residual}")
    print(f"  lambda_smooth: {args.lambda_smooth}")
    print(f"  lambda_qrs  : {args.lambda_qrs}")
    print(f"  lambda_motion: {args.lambda_motion}")
    print(f"  max_residual: {args.max_residual}")
    print(f"  stage2_ckpt : {STAGE2_CKPT}")

    # ── Data ─────────────────────────────────────────────────────────────────
    norm      = load_norm_stats(OUTPUT_ROOT / "norm_stats.json")
    train_h5  = OUTPUT_ROOT / "train.h5"
    val_h5    = OUTPUT_ROOT / "val.h5"
    n_train   = n_windows_in(train_h5)
    n_val     = n_windows_in(val_h5)

    # Use full dataset — chunked reader eliminates HDF5 overhead
    # 750,364 windows / 256-chunk = 2,930 reads (vs 750,364 before)
    n_batches_train = (n_train + batch_size - 1) // batch_size
    n_batches_val   = (n_val   + batch_size - 1) // batch_size
    train_chunks    = (n_train + (batch_size * 4) - 1) // (batch_size * 4)

    print(f"\nData:")
    print(f"  train: {n_train:,} windows → {n_batches_train:,} batches")
    print(f"  val  : {n_val:,} windows → {n_batches_val:,} batches")
    print(f"  chunk: 256 windows/read  ({n_train//256:,} reads/epoch)")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = ResidualArtifactRemover(max_residual=args.max_residual).to(device)

    if not STAGE2_CKPT.exists():
        raise FileNotFoundError(
            f"Stage 2 checkpoint not found: {STAGE2_CKPT}\n"
            f"Run train_stage2.py first."
        )
    model.load_imu_encoder(str(STAGE2_CKPT), device)

    n_trainable, n_frozen = count_params(model)
    print(f"\nModel: ResidualArtifactRemover")
    print(f"  Trainable params: {n_trainable:,}")
    print(f"  Frozen IMU params:{n_frozen:,}")

    # ── Optimiser + scheduler ─────────────────────────────────────────────────
    # Only optimise trainable (non-frozen) parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'weight' in name and 'norm' not in name.lower():
            decay.append(param)
        else:
            no_decay.append(param)

    optimiser = torch.optim.AdamW([
        {'params': decay,    'weight_decay': 1e-4},
        {'params': no_decay, 'weight_decay': 0.0},
    ], lr=lr)

    # ReduceLROnPlateau responds to val loss oscillations —
    # cuts LR by 0.5 when val doesn't improve for 5 epochs.
    # Fixed cosine scheduler was blind to val spikes; ReduceLROnPlateau
    # directly addresses instability by reducing gradient step size.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser,
        mode     = 'min',
        factor   = 0.5,       # halve LR on plateau
        patience = 5,         # wait 5 epochs before reducing
        min_lr   = 1e-6,      # floor
    )

    # Mixed precision
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

    # ── Resume from checkpoint ────────────────────────────────────────────────
    start_epoch = 1
    best_val    = float("inf")
    no_improve  = 0
    history     = []

    if args.resume and Path(args.resume).exists():
        ckpt = safe_torch_load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimiser.load_state_dict(ckpt["optimiser_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            print(f"  Scheduler state restored")
        start_epoch = ckpt["epoch"] + 1
        history     = ckpt.get("history", [])
        best_val    = ckpt.get("best_val", ckpt["val_loss"])
        no_improve  = ckpt.get("no_improve", 0)
        if history and "best_val" not in ckpt:
            best_val = min(float(h.get("val", best_val)) for h in history)
            best_epoch_idx = min(
                range(len(history)),
                key=lambda i: float(history[i].get("val", float("inf")))
            )
            no_improve = max(0, len(history) - best_epoch_idx - 1)
        print(f"\nResumed from epoch {ckpt['epoch']} "
              f"(val={best_val:.4f})")
        print(f"  no_improve={no_improve}")

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Stage 3 Training")
    print(f"{'='*55}")

    for epoch in range(start_epoch, max_epochs + 1):
        t0 = time.time()

        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        model.imu_encoder.eval()   # IMU encoder always in eval mode (frozen)

        t_losses = []
        train_gen = chunked_loader(train_h5, norm, chunk=batch_size*4,
                                   shuffle=True)
        train_bar = tqdm(train_gen,
                         total=train_chunks,
                         desc=f"Epoch {epoch:3d}/{max_epochs} [train]",
                         leave=False)

        for chunk_batch in train_bar:
            # Split chunk into mini-batches of batch_size
            ecg_chunk = chunk_batch['ecg']
            imu_chunk = chunk_batch['imu']
            mai_chunk = chunk_batch['movement_accel']
            noise_chunk = chunk_batch['noise_annotated']
            rel_chunk = chunk_batch['rpeaks_reliable']
            n_mb      = (len(ecg_chunk) + batch_size - 1) // batch_size
            for mb in range(n_mb):
                batch = {
                    'ecg':         ecg_chunk[mb*batch_size:(mb+1)*batch_size],
                    'imu':         imu_chunk[mb*batch_size:(mb+1)*batch_size],
                    'movement_accel': mai_chunk[mb*batch_size:(mb+1)*batch_size],
                    'noise_annotated': noise_chunk[mb*batch_size:(mb+1)*batch_size],
                    'rpeaks_reliable': rel_chunk[mb*batch_size:(mb+1)*batch_size],
                }
                if len(batch['ecg']) == 0:
                    continue

                ecg    = batch["ecg"].to(device)           # [B, 1, 10240]
                imu    = batch["imu"].to(device)           # [B, 6,  640]
                mai    = batch["movement_accel"].to(device)
                noise  = batch["noise_annotated"].to(device)
                rel    = batch["rpeaks_reliable"].to(device)

                with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                    clean, residual = model(
                        ecg, imu, mai=mai, noise_annotated=noise,
                        return_residual=True,
                    )
                    loss, info = residual_artifact_loss(
                        ecg, clean, residual, mai, noise, rel,
                        lambda_identity=args.lambda_identity,
                        lambda_residual=args.lambda_residual,
                        lambda_smooth=args.lambda_smooth,
                        lambda_qrs=args.lambda_qrs,
                        lambda_motion=args.lambda_motion,
                    )

                optimiser.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimiser)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                scaler.step(optimiser)
                scaler.update()

                t_losses.append(loss.item())
                train_bar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    id=f"{info['L_identity']:.4f}",
                    rms=f"{info['residual_rms']:.3f}",
                )

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        v_losses, v_id, v_res, v_qrs, v_motion = [], [], [], [], []
        v_rms, v_rms_low, v_rms_high = [], [], []

        with torch.no_grad():
            for batch in chunked_loader(val_h5, norm, chunk=batch_size*4,
                                        shuffle=False):
                ecg    = batch["ecg"].to(device)
                imu    = batch["imu"].to(device)
                mai    = batch["movement_accel"].to(device)
                noise  = batch["noise_annotated"].to(device)
                rel    = batch["rpeaks_reliable"].to(device)

                with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                    clean, residual = model(
                        ecg, imu, mai=mai, noise_annotated=noise,
                        return_residual=True,
                    )
                    loss, info = residual_artifact_loss(
                        ecg, clean, residual, mai, noise, rel,
                        lambda_identity=args.lambda_identity,
                        lambda_residual=args.lambda_residual,
                        lambda_smooth=args.lambda_smooth,
                        lambda_qrs=args.lambda_qrs,
                        lambda_motion=args.lambda_motion,
                    )

                v_losses.append(loss.item())
                v_id.append(info["L_identity"])
                v_res.append(info["L_residual"])
                v_qrs.append(info["L_qrs"])
                v_motion.append(info["L_motion"])
                v_rms.append(info["residual_rms"])
                v_rms_low.append(info["residual_rms_low"])
                v_rms_high.append(info["residual_rms_high"])

        t_mean = float(np.mean(t_losses))
        v_mean = float(np.mean(v_losses))
        v_id_mean  = float(np.mean(v_id))
        v_res_mean = float(np.mean(v_res))
        v_qrs_mean = float(np.mean(v_qrs))
        v_motion_mean = float(np.mean(v_motion))
        v_rms_mean = float(np.mean(v_rms))
        v_rms_low_mean = float(np.mean(v_rms_low))
        v_rms_high_mean = float(np.mean(v_rms_high))
        elapsed = time.time() - t0
        current_lr = optimiser.param_groups[0]["lr"]

        tqdm.write(
            f"Epoch {epoch:3d}/{max_epochs}  "
            f"train={t_mean:.4f}  val={v_mean:.4f}  "
            f"id={v_id_mean:.4f}  res={v_res_mean:.4f}  "
            f"qrs={v_qrs_mean:.4f}  motion={v_motion_mean:.4f}  "
            f"rms={v_rms_mean:.4f}  "
            f"rms_lo={v_rms_low_mean:.4f}  rms_hi={v_rms_high_mean:.4f}  "
            f"lr={current_lr:.1e}  {elapsed:.1f}s"
        )

        history.append({
            "epoch": epoch, "train": t_mean, "val": v_mean,
            "identity": v_id_mean, "residual": v_res_mean,
            "qrs": v_qrs_mean, "motion": v_motion_mean,
            "residual_rms": v_rms_mean,
            "residual_rms_low": v_rms_low_mean,
            "residual_rms_high": v_rms_high_mean,
        })

        # ── Checkpoint ────────────────────────────────────────────────────────
        def save(path):
            torch.save({
                "epoch":              epoch,
                "seed":               int(args.seed),
                "argv":               sys.argv,
                "val_loss":           v_mean,
                "model_state_dict":   model.state_dict(),
                "optimiser_state_dict": optimiser.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val":           best_val,
                "no_improve":         no_improve,
                "history":            history,
            }, path)

        # ReduceLROnPlateau needs val loss to decide whether to reduce LR
        scheduler.step(v_mean)

        if v_mean < best_val:
            best_val   = v_mean
            no_improve = 0
            save(STAGE3_BEST)
            tqdm.write(f"  ↑ New best val={best_val:.4f} → denoiser_best.pt")
        else:
            no_improve += 1
            if no_improve >= patience:
                tqdm.write(
                    f"  Early stopping at epoch {epoch} "
                    f"(no improvement for {patience} epochs)"
                )
                save(STAGE3_LAST)
                break

        save(STAGE3_LAST)

    # ── Save history ──────────────────────────────────────────────────────────
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  History → {HISTORY_FILE}")

    print(f"\n{'='*55}")
    print(f"  Stage 3 complete.")
    print(f"  Best val loss : {best_val:.4f}")
    print(f"  Checkpoint    : {STAGE3_BEST}")
    print(f"{'='*55}")
    print(f"Next step: python evaluate_stage3.py")


# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Stage 3 ECG Denoiser")
    parser.add_argument("--batch",  type=int,   default=None)
    parser.add_argument("--lr",     type=float, default=None)
    parser.add_argument("--epochs", type=int,   default=None)
    parser.add_argument("--max_residual", type=float, default=MAX_RESIDUAL)
    parser.add_argument("--lambda_identity", type=float, default=LAMBDA_IDENTITY)
    parser.add_argument("--lambda_residual", type=float, default=LAMBDA_RESIDUAL)
    parser.add_argument("--lambda_smooth", type=float, default=LAMBDA_SMOOTH)
    parser.add_argument("--lambda_qrs", type=float, default=LAMBDA_QRS)
    parser.add_argument("--lambda_motion", type=float, default=LAMBDA_MOTION)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED,
                        help="Random seed for Python, NumPy, PyTorch, and CUDA")
    parser.add_argument("--resume", type=str,   default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()
    main(args)
