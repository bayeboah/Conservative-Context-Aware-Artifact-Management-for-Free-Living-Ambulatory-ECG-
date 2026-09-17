"""
train_stage2.py
---------------
Training loop for Stage 2 Siamese Temporal Alignment.

Usage:
    python train_stage2.py
    python train_stage2.py --resume          # resume from last checkpoint
    python train_stage2.py --epochs 100      # override max epochs
    python train_stage2.py --batch 128       # override batch size
    python train_stage2.py --device cpu      # force CPU

Outputs:
    checkpoints/siamese_best.pt   — best validation loss (used by Stage 3)
    checkpoints/siamese_last.pt   — last epoch (for resuming)
    logs/stage2_history.json      — loss curves
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── Project imports ───────────────────────────────────────────────────────────
from config import (
    OUTPUT_ROOT, CHECKPOINT_DIR, LOG_DIR,
    RANDOM_SEED,
    ECG_FS, BATCH_SIZE, NUM_WORKERS,
    LEARNING_RATE, MAX_EPOCHS, EARLY_STOP_PATIENCE,
    STAGE2_LAMBDA_AUX, STAGE2_LAMBDA_CONT,
    ensure_output_dirs,
)
from dataset import CACHETDataset, load_norm_stats
from siamese_alignment import (
    SiameseAlignmentModule, alignment_loss,
    EMBED_DIM, FEAT_LEN,
)
from hdf5_loader import HDF5BatchLoader, PatientDiverseLoader


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


# =============================================================================
# Stage2Dataset — thin wrapper over CACHETDataset
# =============================================================================
class Stage2Dataset(torch.utils.data.Dataset):
    """
    Wraps CACHETDataset to return only the fields Stage 2 needs.

    Returns per window:
        ecg:            [1, 10240]  float32  normalised ECG
        imu:            [6,   640]  float32  normalised acc + gyro
        movement_accel: scalar      float32  MAI for gate
    """
    def __init__(self, h5_path: Path, norm_stats: dict):
        self.base = CACHETDataset(h5_path, norm_stats)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx: int) -> dict:
        w = self.base[idx]
        imu = np.concatenate([w["acc"], w["gyro"]], axis=0)   # [6, 640]
        return {
            "ecg":            torch.from_numpy(w["ecg"]).float(),
            "imu":            torch.from_numpy(imu).float(),
            "movement_accel": torch.tensor(w["movement_accel"], dtype=torch.float32),
            "activity_class": torch.tensor(w["activity_class"], dtype=torch.long),
            "body_position": torch.tensor(w["body_position"], dtype=torch.long),
            "noise_annotated": torch.tensor(w["noise_annotated"], dtype=torch.float32),
            "rpeaks_reliable": torch.tensor(w["rpeaks_reliable"], dtype=torch.float32),
            "emg_flagged": torch.tensor(w["emg_flagged"], dtype=torch.float32),
            "emg_ratio": torch.tensor(w["emg_ratio"], dtype=torch.float32),
        }


# =============================================================================
# Sequential Chunk Sampler — fixes HDF5 random seek bottleneck
# =============================================================================
class SequentialChunkSampler(torch.utils.data.Sampler):
    """
    Reads HDF5 in consecutive chunks instead of random individual windows.

    WHY: With shuffle=True, DataLoader reads random indices from HDF5.
    Each f['ecg'][random_idx] is a separate disk seek on a 27 GB file.
    At 64 seeks per batch × 11,724 batches = 750,336 random seeks per epoch.
    On NVMe SSD each seek costs ~0.5-2ms → total seek overhead dominates.

    FIX: Shuffle at the CHUNK level, not the window level.
    - Divide dataset into chunks of CHUNK_SIZE consecutive windows
    - Shuffle the chunk order each epoch (maintains randomness)
    - Within each chunk, read windows sequentially (one HDF5 slice call)
    - Result: ~2,344 HDF5 reads per epoch instead of 750,336
    - Speedup: 50-100× reduction in seek overhead

    Data diversity is preserved because chunks come from different patients
    and days, so consecutive windows within a chunk are from the same
    patient-day but chunks themselves are shuffled across the epoch.
    """
    def __init__(self, n: int, chunk_size: int = 320, shuffle: bool = True):
        self.n          = n
        self.chunk_size = chunk_size
        self.shuffle    = shuffle

        # Build list of (start, end) chunk boundaries
        self.chunks = [
            (i, min(i + chunk_size, n))
            for i in range(0, n, chunk_size)
        ]

    def __iter__(self):
        chunks = self.chunks.copy()
        if self.shuffle:
            import random
            random.shuffle(chunks)
        for start, end in chunks:
            yield from range(start, end)

    def __len__(self):
        return self.n


# =============================================================================
# Contrastive accuracy metric
# =============================================================================
@torch.no_grad()
def contrastive_accuracy(z_ecg: torch.Tensor,
                          z_imu: torch.Tensor,
                          mai_mask: torch.Tensor) -> float:
    """
    Fraction of windows where argmax of cosine similarity row is the
    diagonal (same-window pair). Chance = 1/B.
    Only computed on high-motion windows.
    """
    if mai_mask.sum() < 2:
        return 0.0
    z_e = z_ecg[mai_mask]
    z_i = z_imu[mai_mask]
    sim = torch.mm(z_e, z_i.T)                          # [M, M]
    preds   = sim.argmax(dim=1)
    targets = torch.arange(len(z_e), device=sim.device)
    return float((preds == targets).float().mean().item())


def class_weights_from_h5(h5_path, dataset_name: str, n_classes: int,
                          device: torch.device) -> torch.Tensor:
    import h5py
    with h5py.File(h5_path, "r") as f:
        labels = f[dataset_name][:]
    labels = labels[(labels >= 0) & (labels < n_classes)]
    counts = np.bincount(labels.astype(np.int64), minlength=n_classes).astype(np.float32)
    total = float(counts.sum())
    weights = np.zeros(n_classes, dtype=np.float32)
    present = counts > 0
    weights[present] = np.sqrt(total / (present.sum() * counts[present]))
    weights = np.clip(weights, 0.25, 5.0)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def balanced_acc_from_counts(correct: np.ndarray, total: np.ndarray) -> float:
    present = total > 0
    if not present.any():
        return 0.0
    return float((correct[present] / np.maximum(total[present], 1)).mean())


def binary_f1(tp: int, fp: int, fn: int) -> float:
    denom = 2 * tp + fp + fn
    return float((2 * tp) / denom) if denom > 0 else 0.0


def binary_pos_weight_from_h5(h5_path, dataset_name: str,
                              device: torch.device,
                              max_weight: float = 100.0) -> torch.Tensor:
    import h5py
    with h5py.File(h5_path, "r") as f:
        y = f[dataset_name][:].astype(np.float32)
    pos = float((y >= 0.5).sum())
    neg = float((y < 0.5).sum())
    weight = neg / max(pos, 1.0)
    weight = min(max(weight, 1.0), max_weight)
    return torch.tensor([weight], dtype=torch.float32, device=device)


def best_binary_f1(scores: list, targets: list) -> float:
    if not scores:
        return 0.0
    s = np.asarray(scores, dtype=np.float32)
    y = np.asarray(targets, dtype=bool)
    if y.sum() == 0:
        return 0.0
    best = 0.0
    for threshold in np.linspace(0.01, 0.99, 99):
        pred = s >= threshold
        tp = int((pred & y).sum())
        fp = int((pred & ~y).sum())
        fn = int((~pred & y).sum())
        best = max(best, binary_f1(tp, fp, fn))
    return best

# =============================================================================
# Main training function
# =============================================================================
def main(args):
    ensure_output_dirs()
    seed = int(args.seed)
    set_all_seeds(seed)
    if args.lambda_cont > 0:
        raise ValueError(
            "Stage 2 is now IMU-only context pretraining. "
            "Set --lambda_cont 0, or restore ECG loading for contrastive experiments."
        )
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # ── Device ────────────────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
    print()

    # ── Hyperparameters ───────────────────────────────────────────────────────
    batch_size   = args.batch  or 128   # 8 windows × 16 patients
    max_epochs   = args.epochs or MAX_EPOCHS
    lr           = args.lr or 1e-4   # harder task (within-patient) needs stronger LR
    patience     = 25   # sim recovery appeared late (epochs 51-53 in v4)
    # With __getitems__ batch reading, num_workers=0 is fastest:
    # one process reads full batch in one HDF5 slice — no spawn overhead.
    # Set to 0 for Windows; Linux can use 2-4 for overlapping IO.
    import sys as _sys
    n_workers = 0 if _sys.platform == 'win32' else min(NUM_WORKERS, 2)

    print(f"Hyperparameters:")
    print(f"  seed       : {seed}")
    print(f"  batch_size : {batch_size}")
    print(f"  lr         : {lr}")
    print(f"  max_epochs : {max_epochs}")
    print(f"  patience   : {patience}")
    print(f"  n_workers  : {n_workers}")
    print(f"  embed_dim  : {EMBED_DIM}")
    print(f"  Feat length: {FEAT_LEN} positions (125ms/position)")
    print(f"  lambda_aux : {args.lambda_aux}")
    print(f"  lambda_cont: {args.lambda_cont}")
    print()

    # ── Data ──────────────────────────────────────────────────────────────────
    norm_path = OUTPUT_ROOT / "norm_stats.json"
    if not norm_path.exists():
        raise FileNotFoundError(
            f"norm_stats.json not found at {norm_path}\n"
            "Run compute_norm.py first."
        )
    norm = load_norm_stats(norm_path)

    # Train: PatientDiverseLoader — 4 windows × 16 patients per batch
    # Gives InfoNCE hard negatives from different patients each batch.
    # Fixes the trivial-negatives problem that caused val loss > log(64).
    # RAM: 16 patients × 400 windows × 55KB = 344 MB.
    print()
    train_loader = PatientDiverseLoader(
        OUTPUT_ROOT / 'train.h5', norm,
        batch_size           = batch_size,
        chunk_per_patient    = 1200,    # 1200 × 16 = 19,200 resist memorisation
        refresh_every        = 1,       # every epoch — fresh windows
        shuffle              = True,
        mai_threshold        = 0.0,
        n_patients_per_batch = 2,       # 2 patients × 64: within-patient hard negatives
    )
    print()

    # Val: PatientDiverseLoader(shuffle=False) — hard negatives in val
    # HDF5BatchLoader gives sequential val batches where all 128 windows
    # come from the same val patient-day. The model hasn't seen val patients,
    # so all their windows land close together → InfoNCE can't find the
    # positive → val loss explodes beyond log(128). 
    # PatientDiverseLoader gives 32 windows × 4 val patients per batch,
    # with genuine cross-patient diversity in negatives → meaningful val loss.
    val_loader = PatientDiverseLoader(
        OUTPUT_ROOT / 'val.h5', norm,
        batch_size        = batch_size,
        chunk_per_patient = 1200,       # 400 windows per val patient
        refresh_every     = 999,       # load once — val set fixed
        shuffle           = False,     # deterministic val
        n_patients_per_batch = 2,       # match train/evaluate_stage2 protocol
        mai_threshold     = 0.0,       # all windows — soft MAI weighting in loss
    )

    activity_weight = class_weights_from_h5(
        OUTPUT_ROOT / 'train.h5', 'activity_class', 16, device
    )
    body_weight = class_weights_from_h5(
        OUTPUT_ROOT / 'train.h5', 'body_position', 16, device
    )
    noise_pos_weight = binary_pos_weight_from_h5(
        OUTPUT_ROOT / 'train.h5', 'noise_annotated', device, max_weight=10.0
    )
    rpeak_pos_weight = binary_pos_weight_from_h5(
        OUTPUT_ROOT / 'train.h5', 'rpeaks_reliable', device, max_weight=10.0
    )
    emg_pos_weight = binary_pos_weight_from_h5(
        OUTPUT_ROOT / 'train.h5', 'emg_flagged', device, max_weight=15.0
    )
    print(
        "  BCE pos_weight: "
        f"noise={noise_pos_weight.item():.1f}  "
        f"rpeak={rpeak_pos_weight.item():.1f}  "
        f"emg={emg_pos_weight.item():.1f}"
    )

    print(f"Data:")
    print(f"  train: {train_loader.n:,} windows  → {len(train_loader):,} batches")
    print(f"  val  : {val_loader.n:,} windows  → {len(val_loader):,} batches")
    print(f"  train batch: {batch_size} "
          f"({train_loader.windows_per_patient} × {len(train_loader.patients)} patients)")
    print(f"  val   batch: {batch_size} "
          f"({val_loader.windows_per_patient} × {len(val_loader.patients)} patients)")
    print()

    # ── Model ─────────────────────────────────────────────────────────────────
    model = SiameseAlignmentModule(embed_dim=EMBED_DIM).to(device)
    for module in (model.ecg_encoder, model.ecg_proj, model.imu_proj):
        for p in module.parameters():
            p.requires_grad = False

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: IMU context encoder")
    print(f"  Trainable params: {n_params:,}")
    print()

    # ── Optimiser + scheduler ─────────────────────────────────────────────────
    # Two param groups — logit_scale is now a buffer (not a parameter)
    #   1. Weight matrices: weight_decay=1e-4
    #   2. Biases + GN/BN params: no weight decay
    decay_params    = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if ('weight' in name and
                'norm' not in name.lower() and
                'bn' not in name.lower()):
            decay_params.append(param)
        else:
            no_decay_params.append(param)
    optimiser = torch.optim.AdamW([
        {'params': decay_params,    'weight_decay': 1e-2},  # strong regularisation
        {'params': no_decay_params, 'weight_decay': 0.0},
    ], lr=lr)

    # CosineAnnealingLR + 3-epoch linear warmup
    # Standard for contrastive pretraining (SimCLR, MoCo, CLIP).
    # Smooth monotonic decay is better than OneCycleLR's aggressive peak
    # for cross-modal tasks where the embedding space needs stable shaping.
    # 5-epoch warmup for within-patient hard negatives task
    WARMUP_EPOCHS = 5
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser,
        T_max   = max_epochs - WARMUP_EPOCHS,
        eta_min = lr * 0.01,
    )

    def warmup_step(epoch: int):
        if epoch <= WARMUP_EPOCHS:
            warmup_lr = lr * epoch / WARMUP_EPOCHS
            optimiser.param_groups[0]['lr'] = warmup_lr
            optimiser.param_groups[1]['lr'] = warmup_lr

    # AMP scaler (mixed precision on CUDA)
    use_amp = (device.type == "cuda")
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 1
    best_val    = float("inf")
    best_score  = -float("inf")
    no_improve  = 0
    history     = {
        "train": [], "val": [], "acc": [], "acc_motion": [],
        "sim": [], "sim_motion": [], "cont": [], "aux": [],
        "aux_mai": [], "aux_activity": [], "aux_body": [],
        "aux_noise": [], "aux_rpeak": [], "aux_emg_flag": [],
        "aux_emg_ratio": [], "best_epoch": 0
    }

    last_ckpt = CHECKPOINT_DIR / "siamese_last.pt"
    if args.resume and last_ckpt.exists():
        ckpt = safe_torch_load(last_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimiser.load_state_dict(ckpt["optim_state"])
        if "scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        if "scaler_state" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val    = ckpt.get("best_val", float("inf"))
        best_score  = ckpt.get("best_score", max(history.get("score", [-float("inf")])))
        no_improve  = ckpt.get("no_improve", 0)
        history     = ckpt.get("history",  history)
        for key in (
            "aux", "acc_motion", "sim", "sim_motion", "cont",
            "aux_mai", "aux_activity", "aux_body", "aux_noise", "aux_rpeak",
            "aux_emg_flag", "aux_emg_ratio"
        ):
            history.setdefault(key, [])
        print(f"Resumed from epoch {ckpt['epoch']}  (val={best_val:.4f})")
        print()
    elif args.resume:
        raise FileNotFoundError(
            f"Resume checkpoint not found: {last_ckpt}\n"
            "Run at least one epoch first to create a checkpoint."
        )

    # ── Training loop ─────────────────────────────────────────────────────────
    print("="*65)
    print("  Stage 2 Training")
    print("="*65)
    print()

    for epoch in range(start_epoch, max_epochs + 1):
        t0 = time.time()

        # ── learning rate (warmup or cosine) ──────────────────────────────────
        # ── learning rate ─────────────────────────────────────────────────────
        warmup_step(epoch)

        # ── train epoch ───────────────────────────────────────────────────────
        model.train()
        t_losses = []
        train_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch:3d}/{max_epochs} [train]",
            unit="batch",
            leave=False,
            dynamic_ncols=True,
        )
        for batch in train_bar:
            ecg = batch["ecg"].to(device) if args.lambda_cont > 0 else None
            imu = batch["imu"].to(device)
            mai = batch["movement_accel"].to(device)
            activity = batch["activity_class"].to(device)
            body = batch["body_position"].to(device)
            noise = batch["noise_annotated"].to(device)
            reliable = batch["rpeaks_reliable"].to(device)
            emg_flag = batch["emg_flagged"].to(device)
            emg_ratio = batch["emg_ratio"].to(device)

            optimiser.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                out  = model(
                    ecg, imu, mai,
                    activity=activity,
                    body_position=body,
                    noise_annotated=noise,
                    rpeaks_reliable=reliable,
                    emg_flagged=emg_flag,
                    emg_ratio=emg_ratio,
                )
                loss, info = alignment_loss(
                    out, lambda_var=0.0,
                    lambda_aux=args.lambda_aux,
                    lambda_cont=args.lambda_cont,
                    activity_weight=activity_weight,
                    body_weight=body_weight,
                    noise_pos_weight=noise_pos_weight,
                    rpeak_pos_weight=rpeak_pos_weight,
                    emg_pos_weight=emg_pos_weight,
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimiser)
            scaler.update()
            t_losses.append(loss.item())
            train_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                cont=f"{info.get('L_cont', 0.0):.3f}",
                acc=f"{info.get('acc', 0.0):.1%}",
                aux=f"{info.get('L_aux', 0.0):.3f}",
                T=f"{info.get('temperature', 0.2):.3f}"
            )

        # ── val epoch ─────────────────────────────────────────────────────────
        model.eval()
        v_losses, v_accs, v_motion_accs, v_sims, v_motion_sims = [], [], [], [], []
        v_temps, v_vars, v_aux, v_cont = [], [], [], []
        v_mai, v_activity, v_body, v_noise, v_rpeak = [], [], [], [], []
        v_emg_flag, v_emg_ratio = [], []
        act_correct = np.zeros(16, dtype=np.float64)
        act_total = np.zeros(16, dtype=np.float64)
        body_correct = np.zeros(16, dtype=np.float64)
        body_total = np.zeros(16, dtype=np.float64)
        noise_tp = noise_fp = noise_fn = 0
        emg_tp = emg_fp = emg_fn = 0
        noise_scores, noise_targets = [], []
        emg_scores, emg_targets = [], []
        rpeak_correct = np.zeros(2, dtype=np.float64)
        rpeak_total = np.zeros(2, dtype=np.float64)
        val_bar = tqdm(
            val_loader,
            desc=f"Epoch {epoch:3d}/{max_epochs} [val]  ",
            unit="batch",
            leave=False,
            dynamic_ncols=True,
        )
        with torch.no_grad():
            for batch in val_bar:
                ecg = batch["ecg"].to(device) if args.lambda_cont > 0 else None
                imu = batch["imu"].to(device)
                mai = batch["movement_accel"].to(device)
                activity = batch["activity_class"].to(device)
                body = batch["body_position"].to(device)
                noise = batch["noise_annotated"].to(device)
                reliable = batch["rpeaks_reliable"].to(device)
                emg_flag = batch["emg_flagged"].to(device)
                emg_ratio = batch["emg_ratio"].to(device)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    out = model(
                        ecg, imu, mai,
                        activity=activity,
                        body_position=body,
                        noise_annotated=noise,
                        rpeaks_reliable=reliable,
                        emg_flagged=emg_flag,
                        emg_ratio=emg_ratio,
                    )
                    _, info = alignment_loss(
                        out, lambda_var=0.0,
                        lambda_aux=args.lambda_aux,
                        lambda_cont=args.lambda_cont,
                        activity_weight=activity_weight,
                        body_weight=body_weight,
                        noise_pos_weight=noise_pos_weight,
                        rpeak_pos_weight=rpeak_pos_weight,
                        emg_pos_weight=emg_pos_weight,
                    )

                v_losses.append(info["L_total"])
                v_cont.append(info.get("L_cont", 0.0))
                # .get() handles both old and new siamese_alignment versions
                v_accs.append(info.get("acc", 0.0))
                v_sims.append(info.get("sim_mean", 0.0))
                v_temps.append(info.get("temperature", 0.1))
                v_vars.append(info.get("L_var", 0.0))
                v_aux.append(info.get("L_aux", 0.0))
                v_mai.append(info.get("L_mai", 0.0))
                v_activity.append(info.get("L_activity", 0.0))
                v_body.append(info.get("L_body", 0.0))
                v_noise.append(info.get("L_noise", 0.0))
                v_rpeak.append(info.get("L_rpeak", 0.0))
                v_emg_flag.append(info.get("L_emg_flag", 0.0))
                v_emg_ratio.append(info.get("L_emg_ratio", 0.0))
                act_pred = out["activity_logits"].argmax(dim=1).cpu().numpy()
                act_true = activity.cpu().numpy()
                body_pred = out["body_logits"].argmax(dim=1).cpu().numpy()
                body_true = body.cpu().numpy()
                for c in range(16):
                    m = act_true == c
                    act_total[c] += float(m.sum())
                    act_correct[c] += float((act_pred[m] == c).sum())
                    m = body_true == c
                    body_total[c] += float(m.sum())
                    body_correct[c] += float((body_pred[m] == c).sum())
                noise_prob = torch.sigmoid(out["noise_logit"]).cpu().numpy()
                noise_pred = noise_prob >= 0.5
                noise_true = (noise.cpu().numpy() >= 0.5)
                noise_scores.extend(noise_prob.tolist())
                noise_targets.extend(noise_true.tolist())
                noise_tp += int((noise_pred & noise_true).sum())
                noise_fp += int((noise_pred & ~noise_true).sum())
                noise_fn += int((~noise_pred & noise_true).sum())
                emg_prob = torch.sigmoid(out["emg_logit"]).cpu().numpy()
                emg_pred = emg_prob >= 0.5
                emg_true = (emg_flag.cpu().numpy() >= 0.5)
                emg_scores.extend(emg_prob.tolist())
                emg_targets.extend(emg_true.tolist())
                emg_tp += int((emg_pred & emg_true).sum())
                emg_fp += int((emg_pred & ~emg_true).sum())
                emg_fn += int((~emg_pred & emg_true).sum())
                rpeak_pred = (torch.sigmoid(out["rpeak_logit"]) >= 0.5).cpu().numpy().astype(np.int64)
                rpeak_true = (reliable.cpu().numpy() >= 0.5).astype(np.int64)
                for c in range(2):
                    m = rpeak_true == c
                    rpeak_total[c] += float(m.sum())
                    rpeak_correct[c] += float((rpeak_pred[m] == c).sum())
                motion_mask = mai > 0.08
                if args.lambda_cont > 0 and motion_mask.sum() >= 2:
                    v_motion_accs.append(
                        contrastive_accuracy(
                            out["p_ecg"], out["p_imu"], motion_mask
                        )
                    )
                    v_motion_sims.append(
                        float(out["sim"][motion_mask].mean().item())
                    )
                val_bar.set_postfix(loss=f"{info['L_total']:.4f}")

        t_mean  = float(np.mean(t_losses))
        v_mean  = float(np.mean(v_losses))
        v_acc   = float(np.mean(v_accs))
        v_sim   = float(np.mean(v_sims))   # positive pair cosine similarity
        v_aux_mean = float(np.mean(v_aux))
        v_cont_mean = float(np.mean(v_cont))
        v_motion_acc = float(np.mean(v_motion_accs)) if v_motion_accs else 0.0
        v_motion_sim = float(np.mean(v_motion_sims)) if v_motion_sims else 0.0
        v_mai_mean = float(np.mean(v_mai))
        v_activity_mean = float(np.mean(v_activity))
        v_body_mean = float(np.mean(v_body))
        v_noise_mean = float(np.mean(v_noise))
        v_rpeak_mean = float(np.mean(v_rpeak))
        v_emg_flag_mean = float(np.mean(v_emg_flag))
        v_emg_ratio_mean = float(np.mean(v_emg_ratio))
        v_activity_bal = balanced_acc_from_counts(act_correct, act_total)
        v_body_bal = balanced_acc_from_counts(body_correct, body_total)
        v_rpeak_bal = balanced_acc_from_counts(rpeak_correct, rpeak_total)
        v_noise_f1 = binary_f1(noise_tp, noise_fp, noise_fn)
        v_emg_f1 = binary_f1(emg_tp, emg_fp, emg_fn)
        v_noise_best_f1 = best_binary_f1(noise_scores, noise_targets)
        v_emg_best_f1 = best_binary_f1(emg_scores, emg_targets)
        v_score = (
            0.35 * v_activity_bal
            + 0.25 * v_body_bal
            + 0.25 * v_emg_best_f1
            + 0.10 * v_rpeak_bal
            - 0.05 * v_mai_mean
        )

        history["train"].append(t_mean)
        history["val"].append(v_mean)
        history["acc"].append(v_acc)
        history["acc_motion"].append(v_motion_acc)
        history["sim"].append(v_sim)
        history["sim_motion"].append(v_motion_sim)
        history["cont"].append(v_cont_mean)
        history["aux"].append(v_aux_mean)
        history["aux_mai"].append(v_mai_mean)
        history["aux_activity"].append(v_activity_mean)
        history["aux_body"].append(v_body_mean)
        history["aux_noise"].append(v_noise_mean)
        history["aux_rpeak"].append(v_rpeak_mean)
        history["aux_emg_flag"].append(v_emg_flag_mean)
        history["aux_emg_ratio"].append(v_emg_ratio_mean)
        history.setdefault("score", []).append(v_score)
        history.setdefault("activity_bal_acc", []).append(v_activity_bal)
        history.setdefault("body_bal_acc", []).append(v_body_bal)
        history.setdefault("emg_f1", []).append(v_emg_f1)
        history.setdefault("noise_f1", []).append(v_noise_f1)
        history.setdefault("emg_best_f1", []).append(v_emg_best_f1)
        history.setdefault("noise_best_f1", []).append(v_noise_best_f1)
        history.setdefault("rpeak_bal_acc", []).append(v_rpeak_bal)

        current_lr = optimiser.param_groups[0]['lr']
        elapsed    = time.time() - t0

        # Close both bars before printing — prevents race with next epoch
        val_bar.close()
        train_bar.close()
        v_temp = float(np.mean(v_temps)) if v_temps else 0.2
        tqdm.write(
            f"Epoch {epoch:3d}/{max_epochs}  "
            f"train={t_mean:.4f}  val={v_mean:.4f}  "
            f"score={v_score:.4f}  "
            f"cont={v_cont_mean:.3f}  acc={v_acc:.1%}  sim={v_sim:.3f}  "
            f"macc={v_motion_acc:.1%}  msim={v_motion_sim:.3f}  "
            f"aux={v_aux_mean:.3f}  "
            f"T=0.100  lr={current_lr:.1e}  {elapsed:.1f}s"
        )
        tqdm.write(
            f"  aux heads: mai={v_mai_mean:.3f}  "
            f"act={v_activity_mean:.3f}  body={v_body_mean:.3f}  "
            f"noise={v_noise_mean:.3f}  rpeak={v_rpeak_mean:.3f}  "
            f"emg={v_emg_flag_mean:.3f}  emg_ratio={v_emg_ratio_mean:.3f}"
        )
        tqdm.write(
            f"  balanced: act={v_activity_bal:.1%}  body={v_body_bal:.1%}  "
            f"emg_f1={v_emg_f1:.1%}  noise_f1={v_noise_f1:.1%}  "
            f"emg_best={v_emg_best_f1:.1%}  noise_best={v_noise_best_f1:.1%}  "
            f"rpeak={v_rpeak_bal:.1%}"
        )

        if epoch > WARMUP_EPOCHS:
            scheduler.step()

        # ── checkpoint ────────────────────────────────────────────────────────
        last_state = {
            "epoch":           epoch,
            "seed":            seed,
            "argv":            sys.argv,
            "model_state":     model.state_dict(),
            "optim_state":     optimiser.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state":    scaler.state_dict(),
            "val_loss":        v_mean,
            "best_val":        best_val,
            "best_score":      best_score,
            "no_improve":      no_improve,
            "history":         history,
        }
        torch.save(last_state, CHECKPOINT_DIR / "siamese_last.pt")

        can_update_best = epoch > WARMUP_EPOCHS
        if can_update_best and (v_score > best_score or
                                (v_score == best_score and v_mean < best_val)):
            best_val   = v_mean
            best_score  = v_score
            no_improve = 0
            history["best_epoch"] = epoch
            torch.save(
                {"epoch": epoch,
                 "seed":  seed,
                 "argv":  sys.argv,
                 "model_state": model.state_dict(),
                 "val_loss":    best_val,
                 "val_score":   best_score,
                 "embed_dim":   EMBED_DIM},
                CHECKPOINT_DIR / "siamese_best.pt"
            )
            tqdm.write(f"  ↑ New best val={best_val:.4f} → siamese_best.pt")
        else:
            if not can_update_best:
                tqdm.write(
                    f"  Warmup epoch {epoch}: not eligible for best checkpoint"
                )
            else:
                no_improve += 1

        if no_improve >= patience:
            print(f"\n  Early stopping at epoch {epoch} "
                  f"(no improvement for {patience} epochs)")
            break

    # ── Save history ──────────────────────────────────────────────────────────
    history_path = LOG_DIR / "stage2_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n  History → {history_path}")

    print()
    print("="*65)
    print(f"  Stage 2 complete. Best epoch: {history['best_epoch']}")
    print(f"  Best val loss : {best_val:.4f}")
    print(f"  Checkpoint    : {CHECKPOINT_DIR / 'siamese_best.pt'}")
    print("="*65)
    print()
    print("Next step: python train_stage3.py")
    print("  Stage 3 will load the frozen IMU encoder from siamese_best.pt")

    return history


# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Stage 2 Siamese Alignment")
    parser.add_argument("--resume",  action="store_true",
                        help="Resume from siamese_last.pt")
    parser.add_argument("--epochs",  type=int,   default=None)
    parser.add_argument("--batch",   type=int,   default=None)
    parser.add_argument("--lr",      type=float, default=None)
    parser.add_argument("--lambda_aux", type=float, default=STAGE2_LAMBDA_AUX,
                        help="Weight for IMU context/quality auxiliary losses")
    parser.add_argument("--lambda_cont", type=float, default=STAGE2_LAMBDA_CONT,
                        help="Optional weight for ECG-IMU contrastive loss")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED,
                        help="Random seed for Python, NumPy, PyTorch, and CUDA")
    parser.add_argument("--device",  type=str,   default=None,
                        choices=["cpu", "cuda"])
    args = parser.parse_args()
    main(args)
