"""
train_mitbih_denoiser.py
------------------------
Trains the ECG-only denoiser for the cross-dataset transfer arm
(see MITBIH_TRANSFER_EXPERIMENT_DESIGN.md).

Architecture: 1-D U-Net that predicts a BOUNDED residual r (clean = noisy - r),
mirroring the CACHET ResidualArtifactRemover so the two synthetic arms differ
only in training data, not in output parameterisation. No IMU pathway — MIT-BIH
has no accelerometer, and ECG-only matches the cited literature (Chiang [4],
Dias [10], Arsene [11]).

Trained on MIT-BIH clean MLII + injected NSTDB electrode-motion noise. Reports
in-domain validation with the STANDARD denoising metric (SNR improvement in dB)
so the model is shown to succeed on its own benchmark BEFORE it is transferred to
CACHET — the structure that makes the eventual transfer failure meaningful.

Usage:
    python train_mitbih_denoiser.py --seed 7 --epochs 60
Outputs:
    checkpoints/mitbih_em_denoiser_best.pt
    logs/mitbih_em_history.json
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

FS = 1024
WINDOW_LEN = 10240
QRS_LO, QRS_HI = 5.0, 20.0          # QRS-band spectral preservation term


# ── Reproducibility (same convention as the CACHET training scripts) ──────────
def set_all_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Model ─────────────────────────────────────────────────────────────────────
def conv_block(cin, cout, k=9):
    p = k // 2
    return nn.Sequential(
        nn.Conv1d(cin, cout, k, padding=p), nn.GroupNorm(min(8, cout), cout),
        nn.ReLU(inplace=True),
        nn.Conv1d(cout, cout, k, padding=p), nn.GroupNorm(min(8, cout), cout),
        nn.ReLU(inplace=True),
    )


class ECGUNet1D(nn.Module):
    """1-D U-Net predicting a bounded residual. clean = noisy - max_res*tanh(out)."""

    def __init__(self, base=16, depth=4, max_residual=0.35):
        super().__init__()
        self.max_residual = float(max_residual)
        chs = [1] + [base * (2 ** i) for i in range(depth)]      # e.g. 1,16,32,64,128
        self.downs = nn.ModuleList(conv_block(chs[i], chs[i + 1]) for i in range(depth))
        self.pool = nn.MaxPool1d(2)
        self.bottleneck = conv_block(chs[-1], chs[-1] * 2)
        self.ups = nn.ModuleList(
            nn.ConvTranspose1d(chs[-1] * 2 if i == 0 else chs[depth - i + 1],
                               chs[depth - i], 2, stride=2)
            for i in range(depth))
        self.up_blocks = nn.ModuleList(
            conv_block(chs[depth - i] * 2, chs[depth - i]) for i in range(depth))
        self.head = nn.Conv1d(chs[1], 1, 1)

    def forward(self, x):                      # x: (B,1,L)
        skips = []
        h = x
        for d in self.downs:
            h = d(h); skips.append(h); h = self.pool(h)
        h = self.bottleneck(h)
        for up, blk, skip in zip(self.ups, self.up_blocks, reversed(skips)):
            h = up(h)
            if h.shape[-1] != skip.shape[-1]:
                h = F.pad(h, (0, skip.shape[-1] - h.shape[-1]))
            h = blk(torch.cat([h, skip], dim=1))
        r = self.max_residual * torch.tanh(self.head(h))
        return x - r, r


# ── Loss ──────────────────────────────────────────────────────────────────────
def _log_band_power(x, lo, hi):
    n = x.shape[-1]
    freqs = torch.fft.rfftfreq(n, d=1.0 / FS).to(x.device)
    spec = torch.fft.rfft(x.squeeze(1), dim=-1).abs().clamp_min(1e-6).log()
    mask = (freqs >= lo) & (freqs <= hi)
    return spec[:, mask].mean(dim=1)


def denoise_loss(clean_pred, clean, r, lam_qrs=0.5, lam_smooth=0.05):
    l_mse = F.smooth_l1_loss(clean_pred, clean)
    l_qrs = (_log_band_power(clean_pred, QRS_LO, QRS_HI)
             - _log_band_power(clean, QRS_LO, QRS_HI)).abs().mean()
    l_smooth = (r[..., 1:] - r[..., :-1]).abs().mean()
    total = l_mse + lam_qrs * l_qrs + lam_smooth * l_smooth
    return total, {"mse": l_mse.item(), "qrs": l_qrs.item(), "smooth": l_smooth.item()}


# ── Metrics (in-domain: true clean available) ─────────────────────────────────
def snr_db(clean, recovered):
    p_sig = (clean ** 2).sum(-1)
    p_err = ((clean - recovered) ** 2).sum(-1).clamp_min(1e-12)
    return 10.0 * torch.log10(p_sig / p_err)


def pearson(a, b):
    a = a - a.mean(-1, keepdim=True)
    b = b - b.mean(-1, keepdim=True)
    num = (a * b).sum(-1)
    den = (a.norm(dim=-1) * b.norm(dim=-1)).clamp_min(1e-8)
    return num / den


# ── Data ──────────────────────────────────────────────────────────────────────
def load_arrays(h5_path):
    with h5py.File(h5_path, "r") as f:
        clean = f["clean"][:]
        noisy = f["noisy"][:]
        snr_in = f["snr_in"][:]
    return (torch.from_numpy(clean).float().unsqueeze(1),
            torch.from_numpy(noisy).float().unsqueeze(1),
            torch.from_numpy(snr_in).float())


# ── Train / eval loops ────────────────────────────────────────────────────────
def evaluate(model, loader, device):
    model.eval()
    snr_in_all, snr_out_all, rmse_all, corr_all = [], [], [], []
    with torch.no_grad():
        for noisy, clean, snr_in in loader:
            noisy, clean = noisy.to(device), clean.to(device)
            clean_pred, _ = model(noisy)
            snr_in_all.append(snr_in)
            snr_out_all.append(snr_db(clean.squeeze(1), clean_pred.squeeze(1)).cpu())
            rmse_all.append(((clean_pred - clean) ** 2).mean(dim=[1, 2]).sqrt().cpu())
            corr_all.append(pearson(clean_pred.squeeze(1), clean.squeeze(1)).cpu())
    snr_in = torch.cat(snr_in_all); snr_out = torch.cat(snr_out_all)
    rmse = torch.cat(rmse_all); corr = torch.cat(corr_all)
    imp = snr_out - snr_in
    return {
        "snr_in_mean": float(snr_in.mean()),
        "snr_out_mean": float(snr_out.mean()),
        "snr_improvement_mean": float(imp.mean()),
        "rmse_mean": float(rmse.mean()),
        "qrs_corr_mean": float(corr.mean()),
        "per_snr_bin": _per_snr(snr_in, imp, corr),
    }, float(imp.mean())


def _per_snr(snr_in, imp, corr):
    edges = [-9, -3, 3, 9, 15, 21, 27]
    out = {}
    for i in range(len(edges) - 1):
        m = (snr_in >= edges[i]) & (snr_in < edges[i + 1])
        if m.any():
            out[f"[{edges[i]},{edges[i+1]})dB"] = {
                "n": int(m.sum()),
                "snr_improvement": round(float(imp[m].mean()), 3),
                "qrs_corr": round(float(corr[m].mean()), 5),
            }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--base", type=int, default=16)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--max_residual", type=float, default=0.35)
    ap.add_argument("--lam_qrs", type=float, default=0.5)
    ap.add_argument("--lam_smooth", type=float, default=0.05)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    set_all_seeds(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out = args.root / "output"
    ckpt_dir = args.root / "checkpoints"; ckpt_dir.mkdir(exist_ok=True)
    log_dir = args.root / "logs"; log_dir.mkdir(exist_ok=True)

    tr_path, va_path = out / "mitbih_em_train.h5", out / "mitbih_em_val.h5"
    if not tr_path.exists():
        sys.exit(f"{tr_path} not found — run build_mitbih_em.py first.")

    ctr = load_arrays(tr_path); cva = load_arrays(va_path)
    tr_loader = DataLoader(TensorDataset(ctr[1], ctr[0], ctr[2]),
                           batch_size=args.batch, shuffle=True, drop_last=True)
    va_loader = DataLoader(TensorDataset(cva[1], cva[0], cva[2]),
                           batch_size=args.batch, shuffle=False)

    model = ECGUNet1D(args.base, args.depth, args.max_residual).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    n_params = sum(p.numel() for p in model.parameters())

    print("=" * 60)
    print("  MIT-BIH/NSTDB-em ECG-only denoiser (bounded residual)")
    print("=" * 60)
    print(f"  device {device} | seed {args.seed} | params {n_params:,}")
    print(f"  train {len(ctr[0]):,} / val {len(cva[0]):,} windows")
    print(f"  bounded residual max={args.max_residual}\n")

    history, best_imp, best_ep, since = [], -1e9, 0, 0
    for ep in range(1, args.epochs + 1):
        model.train(); tl = 0.0
        for noisy, clean, _ in tr_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            clean_pred, r = model(noisy)
            loss, _ = denoise_loss(clean_pred, clean, r, args.lam_qrs, args.lam_smooth)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tl += float(loss)
        sched.step()
        val, imp = evaluate(model, va_loader, device)
        history.append({"epoch": ep, "train_loss": tl / len(tr_loader), **val})
        print(f"  ep {ep:3d}  train {tl/len(tr_loader):.4f}  "
              f"val SNRimp {val['snr_improvement_mean']:+.2f}dB  "
              f"QRScorr {val['qrs_corr_mean']:.4f}  RMSE {val['rmse_mean']:.4f}")

        if imp > best_imp:
            best_imp, best_ep, since = imp, ep, 0
            # Store a JSON-safe args dict (no pathlib objects) so the checkpoint
            # loads cleanly under torch.load(weights_only=True) in PyTorch >=2.6.
            safe_args = {k: (str(v) if isinstance(v, Path) else v)
                         for k, v in vars(args).items()}
            torch.save({"model_state": model.state_dict(), "seed": args.seed,
                        "epoch": ep, "val": val, "args": safe_args,
                        "arch": {"base": args.base, "depth": args.depth,
                                 "max_residual": args.max_residual}},
                       ckpt_dir / "mitbih_em_denoiser_best.pt")
        else:
            since += 1
            if since >= args.patience:
                print(f"  early stop (no improvement {args.patience} epochs)")
                break

    with open(log_dir / "mitbih_em_history.json", "w") as f:
        json.dump({"seed": args.seed, "best_epoch": best_ep,
                   "best_snr_improvement": best_imp, "history": history}, f, indent=2)

    print("\n" + "=" * 60)
    print(f"  Best epoch {best_ep}: in-domain SNR improvement {best_imp:+.2f} dB")
    print(f"  -> {ckpt_dir/'mitbih_em_denoiser_best.pt'}")
    print(f"  -> {log_dir/'mitbih_em_history.json'}")
    print("  In-domain success established. Next: evaluate on CACHET via")
    print("  evaluate_stage3.py --mode ecg_denoiser (transfer test).")
    print("=" * 60)


if __name__ == "__main__":
    main()
