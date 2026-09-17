"""
evaluate_stage2.py
------------------
Evaluate the Stage 2 Siamese ECG-IMU alignment model.

This evaluator intentionally mirrors the Stage 2 training/validation batch
contract: patient-diverse batches, projection-head retrieval for InfoNCE, and
encoder cosine diagnostics for the representation later consumed by Stage 3.

Usage:
    python evaluate_stage2.py                  # val + test
    python evaluate_stage2.py --split val
    python evaluate_stage2.py --split test
    python evaluate_stage2.py --n_batches 50   # subset
    python evaluate_stage2.py --n_batches 0    # all batches
"""

import argparse
import json
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch

from config import (
    CHECKPOINT_DIR, LOG_DIR, OUTPUT_ROOT, STAGE2_LAMBDA_AUX,
    STAGE2_LAMBDA_CONT,
    ensure_output_dirs,
)
from dataset import load_norm_stats
from hdf5_loader import PatientDiverseLoader
from siamese_alignment import EMBED_DIM, SiameseAlignmentModule, alignment_loss
from train_stage2 import binary_pos_weight_from_h5, best_binary_f1


def safe_torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


BG = "#0D1117"
PANEL_BG = "#161B22"
GRID = "#21262D"
TEXT = "#E6EDF3"
SUB = "#8B949E"
C1, C2, C3 = "#4FC3F7", "#81C995", "#FFB74D"


def load_model(device: torch.device) -> SiameseAlignmentModule:
    ckpt_path = CHECKPOINT_DIR / "siamese_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"siamese_best.pt not found at {ckpt_path}\n"
            "Run train_stage2.py first."
        )

    ckpt = safe_torch_load(ckpt_path, map_location=device)
    embed = ckpt.get("embed_dim", EMBED_DIM)
    model = SiameseAlignmentModule(embed_dim=embed).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(
        f"Loaded siamese_best.pt  (epoch {ckpt['epoch']}, "
        f"val_loss={ckpt['val_loss']:.4f})"
    )
    return model


def make_loader(split: str, norm: dict, args) -> PatientDiverseLoader:
    return PatientDiverseLoader(
        OUTPUT_ROOT / f"{split}.h5",
        norm,
        batch_size=args.batch_size,
        chunk_per_patient=args.chunk_per_patient,
        refresh_every=999,
        shuffle=False,
        mai_threshold=0.0,
        n_patients_per_batch=args.n_patients_per_batch,
    )


@torch.no_grad()
def evaluate(model, loader, device, args, n_batches=None):
    """Run Stage 2 evaluation with training-compatible batch semantics."""
    metrics = defaultdict(list)
    use_contrastive = args.lambda_cont > 0
    noise_pos_weight = binary_pos_weight_from_h5(
        OUTPUT_ROOT / "train.h5", "noise_annotated", device, max_weight=10.0
    )
    rpeak_pos_weight = binary_pos_weight_from_h5(
        OUTPUT_ROOT / "train.h5", "rpeaks_reliable", device, max_weight=10.0
    )
    emg_pos_weight = binary_pos_weight_from_h5(
        OUTPUT_ROOT / "train.h5", "emg_flagged", device, max_weight=15.0
    )

    for i, batch in enumerate(loader):
        if n_batches and i >= n_batches:
            break

        ecg = batch["ecg"].to(device) if use_contrastive else None
        imu = batch["imu"].to(device)
        mai = batch["movement_accel"].to(device)
        activity = batch["activity_class"].to(device)
        body = batch["body_position"].to(device)
        noise = batch["noise_annotated"].to(device)
        reliable = batch["rpeaks_reliable"].to(device)
        emg_flag = batch["emg_flagged"].to(device)
        emg_ratio = batch["emg_ratio"].to(device)

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
            noise_pos_weight=noise_pos_weight,
            rpeak_pos_weight=rpeak_pos_weight,
            emg_pos_weight=emg_pos_weight,
        )

        batch_size = imu.shape[0]
        targets = torch.arange(batch_size, device=device)

        p_ecg = out["p_ecg"]
        p_imu = out["p_imu"]
        z_ecg = out["z_ecg"]
        z_imu = out["z_imu"]

        metrics["loss"].append(info["L_total"])
        metrics["cont_loss"].append(info.get("L_cont", 0.0))
        metrics["aux_loss"].append(info.get("L_aux", 0.0))
        metrics["emg_flag_loss"].append(info.get("L_emg_flag", 0.0))
        metrics["emg_ratio_loss"].append(info.get("L_emg_ratio", 0.0))
        if use_contrastive and p_ecg is not None:
            sim_proj = torch.mm(p_ecg, p_imu.T) * out["logit_scale"].exp()
            sim_embed = torch.mm(z_ecg, z_imu.T)
            metrics["r1_proj"].append(
                float((sim_proj.argmax(dim=1) == targets).float().mean().item())
            )
            metrics["r1_embed"].append(
                float((sim_embed.argmax(dim=1) == targets).float().mean().item())
            )
            metrics["embedding_cosine"].extend(
                ((z_ecg * z_imu).sum(dim=-1)).cpu().tolist()
            )
            metrics["projection_cosine"].extend(
                ((p_ecg * p_imu).sum(dim=-1)).cpu().tolist()
            )

            sim_np = sim_proj.cpu().numpy()
            hardest_negative = np.max(
                sim_np + np.eye(batch_size, dtype=np.float32) * -1e9,
                axis=1,
            )
            metrics["positive_margin"].extend(
                (np.diag(sim_np) - hardest_negative).tolist()
            )
        metrics["n_active"].append(int(out["mai_mask"].sum().item()))
        metrics["n_total"].append(batch_size)

        mai_pred = torch.expm1(out["mai_pred"].float()).clamp_min(0.0)
        metrics["mai_abs_error"].extend((mai_pred - mai).abs().cpu().tolist())

        activity_pred = out["activity_logits"].argmax(dim=1)
        activity_valid = (
            (activity >= 0)
            & (activity < out["activity_logits"].shape[1])
        )
        if activity_valid.any():
            metrics["activity_acc"].append(float(
                (activity_pred[activity_valid] == activity[activity_valid])
                .float().mean().item()
            ))

        body_pred = out["body_logits"].argmax(dim=1)
        body_valid = (body >= 0) & (body < out["body_logits"].shape[1])
        if body_valid.any():
            metrics["body_acc"].append(float(
                (body_pred[body_valid] == body[body_valid])
                .float().mean().item()
            ))

        noise_prob = torch.sigmoid(out["noise_logit"])
        metrics["noise_prob"].extend(noise_prob.cpu().tolist())
        metrics["noise_target"].extend((noise >= 0.5).cpu().tolist())
        noise_pred = (noise_prob >= 0.5).float()
        metrics["noise_acc"].append(float((noise_pred == noise).float().mean().item()))
        reliable_pred = (torch.sigmoid(out["rpeak_logit"]) >= 0.5).float()
        metrics["rpeak_acc"].append(float(
            (reliable_pred == reliable).float().mean().item()
        ))
        emg_prob = torch.sigmoid(out["emg_logit"])
        metrics["emg_prob"].extend(emg_prob.cpu().tolist())
        metrics["emg_target"].extend((emg_flag >= 0.5).cpu().tolist())
        emg_pred = (emg_prob >= 0.5).float()
        metrics["emg_flag_acc"].append(float(
            (emg_pred == emg_flag).float().mean().item()
        ))
        emg_ratio_pred = torch.expm1(out["emg_ratio_pred"].float()).clamp_min(0.0)
        metrics["emg_ratio_mae"].extend(
            (emg_ratio_pred - emg_ratio).abs().cpu().tolist()
        )

    emb_sim = np.array(metrics["embedding_cosine"])
    proj_sim = np.array(metrics["projection_cosine"])
    margins = np.array(metrics["positive_margin"])
    n_active_tot = sum(metrics["n_active"])
    n_total_tot = sum(metrics["n_total"])

    results = {
        "loss": float(np.mean(metrics["loss"])) if metrics["loss"] else 0.0,
        "cont_loss": float(np.mean(metrics["cont_loss"])) if metrics["cont_loss"] else 0.0,
        "aux_loss": float(np.mean(metrics["aux_loss"])) if metrics["aux_loss"] else 0.0,
        "emg_flag_loss": float(np.mean(metrics["emg_flag_loss"])) if metrics["emg_flag_loss"] else 0.0,
        "emg_ratio_loss": float(np.mean(metrics["emg_ratio_loss"])) if metrics["emg_ratio_loss"] else 0.0,
        "R@1_projection": float(np.mean(metrics["r1_proj"])) if metrics["r1_proj"] else 0.0,
        "R@1_encoder": float(np.mean(metrics["r1_embed"])) if metrics["r1_embed"] else 0.0,
        "embedding_cosine_mean": float(emb_sim.mean()) if len(emb_sim) else 0.0,
        "projection_cosine_mean": float(proj_sim.mean()) if len(proj_sim) else 0.0,
        "positive_margin_mean": float(margins.mean()) if len(margins) else 0.0,
        "positive_margin_median": float(np.median(margins)) if len(margins) else 0.0,
        "mai_gate_pct": 100.0 * n_active_tot / max(n_total_tot, 1),
        "mai_mae": float(np.mean(metrics["mai_abs_error"])) if metrics["mai_abs_error"] else 0.0,
        "activity_acc": float(np.mean(metrics["activity_acc"])) if metrics["activity_acc"] else 0.0,
        "body_position_acc": float(np.mean(metrics["body_acc"])) if metrics["body_acc"] else 0.0,
        "noise_acc": float(np.mean(metrics["noise_acc"])) if metrics["noise_acc"] else 0.0,
        "noise_best_f1": best_binary_f1(metrics["noise_prob"], metrics["noise_target"]),
        "rpeak_reliable_acc": float(np.mean(metrics["rpeak_acc"])) if metrics["rpeak_acc"] else 0.0,
        "emg_flag_acc": float(np.mean(metrics["emg_flag_acc"])) if metrics["emg_flag_acc"] else 0.0,
        "emg_best_f1": best_binary_f1(metrics["emg_prob"], metrics["emg_target"]),
        "emg_ratio_mae": float(np.mean(metrics["emg_ratio_mae"])) if metrics["emg_ratio_mae"] else 0.0,
    }
    return results, metrics


def make_plots(metrics: dict, results: dict, split: str):
    fig = plt.figure(figsize=(18, 10), facecolor=BG)
    gs = gridspec.GridSpec(
        2, 3, figure=fig, hspace=0.45, wspace=0.35,
        left=0.07, right=0.97, top=0.90, bottom=0.08,
    )

    def ax_style(ax, title):
        ax.set_facecolor(PANEL_BG)
        ax.set_title(title, color=TEXT, fontsize=9, pad=5)
        ax.tick_params(colors=SUB, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)
        ax.grid(color=GRID, lw=0.5, alpha=0.6)
        ax.xaxis.label.set_color(SUB)
        ax.yaxis.label.set_color(SUB)

    ax = fig.add_subplot(gs[0, 0])
    losses = np.array(metrics["loss"])
    ax.plot(losses, color=C1, lw=1.0)
    ax.axhline(results["loss"], color=C3, lw=1.5, linestyle="--",
               label=f"mean {results['loss']:.3f}")
    ax.set_xlabel("Batch")
    ax.set_ylabel("InfoNCE loss")
    ax.legend(fontsize=8, facecolor=PANEL_BG, edgecolor=GRID, labelcolor=TEXT)
    ax_style(ax, "Validation loss per patient-diverse batch")

    ax = fig.add_subplot(gs[0, 1])
    ax.hist(metrics["r1_proj"], bins=20, color=C1, alpha=0.85, edgecolor=PANEL_BG)
    ax.axvline(results["R@1_projection"], color=C3, lw=1.5, linestyle="--",
               label=f"mean {results['R@1_projection']:.1%}")
    ax.set_xlabel("Projection-head R@1")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8, facecolor=PANEL_BG, edgecolor=GRID, labelcolor=TEXT)
    ax_style(ax, "R@1 retrieval used by Stage 2 loss")

    ax = fig.add_subplot(gs[0, 2])
    ax.hist(metrics["r1_embed"], bins=20, color=C2, alpha=0.85, edgecolor=PANEL_BG)
    ax.axvline(results["R@1_encoder"], color=C3, lw=1.5, linestyle="--",
               label=f"mean {results['R@1_encoder']:.1%}")
    ax.set_xlabel("Encoder-space R@1")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8, facecolor=PANEL_BG, edgecolor=GRID, labelcolor=TEXT)
    ax_style(ax, "R@1 in encoder space")

    ax = fig.add_subplot(gs[1, 0])
    emb = np.array(metrics["embedding_cosine"])
    proj = np.array(metrics["projection_cosine"])
    ax.hist(emb, bins=50, alpha=0.65, color=C2,
            label=f"encoder mean={emb.mean():.3f}", edgecolor=PANEL_BG)
    ax.hist(proj, bins=50, alpha=0.65, color=C3,
            label=f"projection mean={proj.mean():.3f}", edgecolor=PANEL_BG)
    ax.set_xlabel("Positive-pair cosine")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8, facecolor=PANEL_BG, edgecolor=GRID, labelcolor=TEXT)
    ax_style(ax, "Positive ECG-IMU cosine similarity")

    ax = fig.add_subplot(gs[1, 1])
    margins = np.array(metrics["positive_margin"])
    ax.hist(margins, bins=50, color=C1, alpha=0.85, edgecolor=PANEL_BG)
    ax.axvline(0, color=SUB, lw=1.0)
    ax.axvline(results["positive_margin_mean"], color=C3, lw=1.5,
               linestyle="--", label=f"mean {results['positive_margin_mean']:.3f}")
    ax.set_xlabel("Positive logit minus hardest negative")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8, facecolor=PANEL_BG, edgecolor=GRID, labelcolor=TEXT)
    ax_style(ax, "Positive-pair margin")

    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    rows = [
        ["Metric", "Value"],
        ["Loss", f"{results['loss']:.4f}"],
        ["Contrastive loss", f"{results['cont_loss']:.4f}"],
        ["Aux loss", f"{results['aux_loss']:.4f}"],
        ["R@1 projection", f"{results['R@1_projection']:.1%}"],
        ["R@1 encoder", f"{results['R@1_encoder']:.1%}"],
        ["Encoder cosine", f"{results['embedding_cosine_mean']:.4f}"],
        ["Projection cosine", f"{results['projection_cosine_mean']:.4f}"],
        ["Margin mean", f"{results['positive_margin_mean']:.4f}"],
        ["Margin median", f"{results['positive_margin_median']:.4f}"],
        ["MAI MAE", f"{results['mai_mae']:.4f}"],
        ["Activity acc", f"{results['activity_acc']:.1%}"],
        ["Body acc", f"{results['body_position_acc']:.1%}"],
        ["Noise acc", f"{results['noise_acc']:.1%}"],
        ["R-peak acc", f"{results['rpeak_reliable_acc']:.1%}"],
        ["MAI gate active", f"{results['mai_gate_pct']:.1f}%"],
    ]
    table = ax.table(cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    for (row, _col), cell in table.get_celld().items():
        cell.set_facecolor(PANEL_BG if row > 0 else "#1F2937")
        cell.set_text_props(color=TEXT if row > 0 else C3)
        cell.set_edgecolor(GRID)
    ax.set_facecolor(PANEL_BG)

    fig.suptitle(
        f"Stage 2 Siamese Alignment Evaluation ({split.upper()} split)",
        color=TEXT, fontsize=13, y=0.96,
    )

    out_path = LOG_DIR / f"stage2_eval_{split}.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Plot -> {out_path}")


def run_split(split: str, model, norm: dict, device: torch.device, args):
    loader = make_loader(split, norm, args)
    n_eval = None if args.n_batches == 0 else args.n_batches
    batches = len(loader) if n_eval is None else min(len(loader), n_eval)
    n_windows = batches * args.batch_size
    print(
        f"\nEvaluating on {split} split "
        f"({n_windows:,} windows, {batches} batches, batch={args.batch_size})..."
    )

    results, metrics = evaluate(model, loader, device, args, n_eval)

    print()
    print("=" * 56)
    print(f"  Stage 2 Evaluation Results - {split}")
    print("=" * 56)
    for key, value in results.items():
        unit = " %" if key.endswith("_pct") else ""
        print(f"  {key:<30}: {value:>8.4f}{unit}")

    out_json = LOG_DIR / f"stage2_eval_{split}.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results -> {out_json}")

    make_plots(metrics, results, split)

    print()
    print("Interpretation:")
    print("  R@1_projection is the metric aligned with Stage 2 InfoNCE training.")
    print("  R@1_encoder and encoder cosine describe the representation passed to Stage 3.")
    if results["R@1_projection"] > (2.0 / args.batch_size):
        print(f"  OK: R@1_projection is above chance ({1.0 / args.batch_size:.1%}).")
    else:
        print(f"  Warning: R@1_projection is near chance ({1.0 / args.batch_size:.1%}).")

    return results


def main(args):
    ensure_output_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    norm = load_norm_stats(OUTPUT_ROOT / "norm_stats.json")

    splits = ["val", "test"] if args.split == "both" else [args.split]
    all_results = {}
    for split in splits:
        all_results[split] = run_split(split, model, norm, device, args)

    if len(all_results) > 1:
        out_json = LOG_DIR / "stage2_eval_both.json"
        with open(out_json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nCombined results -> {out_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="both", choices=["both", "train", "val", "test"])
    parser.add_argument(
        "--n_batches", type=int, default=200,
        help="Batches to evaluate per split (default=200, use 0 for all).",
    )
    parser.add_argument(
        "--batch_size", type=int, default=128,
        help="Evaluation batch size. Use the Stage 2 training batch size.",
    )
    parser.add_argument("--chunk_per_patient", type=int, default=1200)
    parser.add_argument("--n_patients_per_batch", type=int, default=2)
    parser.add_argument("--lambda_aux", type=float, default=STAGE2_LAMBDA_AUX)
    parser.add_argument("--lambda_cont", type=float, default=STAGE2_LAMBDA_CONT)
    main(parser.parse_args())
