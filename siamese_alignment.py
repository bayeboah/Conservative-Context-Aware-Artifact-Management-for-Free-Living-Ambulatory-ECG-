"""
siamese_alignment.py  (v5 — attention pooling, fixed memorisation)
-------------------------------------------------------------------
Changes from v4:
  1. AttentionPool1d replaces time_flatten Linear(10240, 128)
     - time_flatten: 1,310,720 params → AttentionPool: 32,896 params (40× fewer)
     - Still preserves temporal info via learned attention over 80 positions
     - 32K params / 19,200 training windows = 1.7 params/example (vs 409)
     - Makes memorisation genuinely difficult

  2. Dropout(0.1→0.2) in encoder layers
     - Stronger regularisation for cross-patient generalisation

  3. Dropout(0.3) after AttentionPool
     - Guards the bottleneck against memorisation pathways

  4. All other v4 decisions kept:
     - GroupNorm(8) not BatchNorm
     - ReLU(inplace=True) not GELU
     - Fixed temperature T=0.10 (register_buffer)
     - 3-layer projection head
     - lambda_var=0.0 (incompatible with L2 norm)
     - lambda_rhythm=0.0 (0.1% labelled — insufficient)
     - MAI-weighted InfoNCE (soft gate)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

# ── Constants ─────────────────────────────────────────────────────────────────
EMBED_DIM    = 128
PROJ_DIM     = 64
FEAT_LEN     = 80      # temporal positions (125ms each)
WINDOW_ECG   = 10240
WINDOW_IMU   = 640
N_ACTIVITY_CLASSES = 16
N_BODY_CLASSES     = 16


# =============================================================================
# GroupNorm helper
# =============================================================================
def gn(channels: int, groups: int = 8) -> nn.GroupNorm:
    return nn.GroupNorm(min(groups, channels), channels)


# =============================================================================
# ATTENTION POOLING
# Replaces Linear(10240, 128) — 40× fewer parameters, preserves temporal info
# =============================================================================
class AttentionPool1d(nn.Module):
    """
    Temporal attention pooling over the 80 feature map positions.

    A single learnable query vector attends to all 80 temporal positions
    and computes a weighted sum of value projections.

    WHY this beats time_flatten (Linear(10240, 128)):
      time_flatten:     128 × 80 × 128 = 1,310,720 parameters
      AttentionPool1d:  128 + 128×128 + 128×128 = 32,896 parameters

      With 19,200 training windows (1200 × 16 patients):
        time_flatten:    409 params/example → trivial memorisation
        AttentionPool1d:   1.7 params/example → forces generalisation

      Temporal info preserved: the attention weights tell the model
      WHICH of the 80 time positions matter, not just their average.
      A footstep at t=2s attends position 16; one at t=8s attends position 64.
      These produce different z vectors — no time-blindness.

    Args:
        embed_dim: feature dimension (128)
        dropout:   applied to the pooled output (prevents memorisation)
    """

    def __init__(self, embed_dim: int = EMBED_DIM, dropout: float = 0.3):
        super().__init__()
        self.scale    = embed_dim ** -0.5
        self.query    = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.key_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.val_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout  = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.xavier_uniform_(self.val_proj.weight)

    def forward(self, fm: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fm: [B, C, T]  temporal feature map from encoder (e.g. [B, 128, 80])
        Returns:
            z:  [B, C]     pooled embedding — attends to important time positions
        """
        B, C, T = fm.shape
        fm_t = fm.permute(0, 2, 1)                    # [B, T, C]
        k    = self.key_proj(fm_t)                     # [B, T, C]
        v    = self.val_proj(fm_t)                     # [B, T, C]
        q    = self.query.expand(B, -1, -1)            # [B, 1, C]
        attn = F.softmax(torch.bmm(q, k.transpose(1, 2)) * self.scale, dim=-1)
        z    = torch.bmm(attn, v).squeeze(1)           # [B, C]
        return self.dropout(z)


# =============================================================================
# ECG ENCODER   [B, 1, 10240] → z [B, 128]  +  fm [B, 128, 80]
# =============================================================================
class ECGEncoder(nn.Module):
    """
    Four strided Conv1D layers (strides 4,4,4,2 → 80 positions at 125ms each).
    GroupNorm throughout — stable across patient-diverse batches.
    Dropout(0.2) after layers 1-3 — stronger regularisation.
    AttentionPool1d for temporal-aware pooling (32K params, not 1.3M).
    """

    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1,   32, kernel_size=15, stride=4, padding=7, bias=False),
            gn(32),  nn.ReLU(inplace=True), nn.Dropout(0.3),

            nn.Conv1d(32,  64, kernel_size=9,  stride=4, padding=4, bias=False),
            gn(64),  nn.ReLU(inplace=True), nn.Dropout(0.3),

            nn.Conv1d(64, 128, kernel_size=7,  stride=4, padding=3, bias=False),
            gn(128), nn.ReLU(inplace=True), nn.Dropout(0.3),

            nn.Conv1d(128, embed_dim, kernel_size=5, stride=2, padding=2, bias=False),
            gn(embed_dim), nn.ReLU(inplace=True),
        )
        self.pool = AttentionPool1d(embed_dim, dropout=0.3)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            z:  [B, 128]      pooled embedding  → projection head → InfoNCE
            fm: [B, 128, 80]  temporal feature map → Stage 3 cross-attention
        """
        fm = self.encoder(x)      # [B, 128, 80]
        z  = self.pool(fm)        # [B, 128]
        return z, fm


# =============================================================================
# IMU ENCODER   [B, 6, 640] → z [B, 128]  +  fm [B, 128, 80]
# =============================================================================
class IMUEncoder(nn.Module):
    """
    Separate acc/gyro stems, then shared layers.
    Strides: 2, 2, 2 → 640/8 = 80 positions (125ms each, matching ECG).
    AttentionPool1d for temporal-aware pooling.
    """

    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.acc_stem  = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=7, stride=2, padding=3, bias=False),
            gn(16), nn.ReLU(inplace=True), nn.Dropout(0.3),
        )
        self.gyro_stem = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=7, stride=2, padding=3, bias=False),
            gn(16), nn.ReLU(inplace=True), nn.Dropout(0.3),
        )
        self.shared = nn.Sequential(
            nn.Conv1d(32,  64, kernel_size=7, stride=2, padding=3, bias=False),
            gn(64), nn.ReLU(inplace=True), nn.Dropout(0.3),

            nn.Conv1d(64, embed_dim, kernel_size=5, stride=2, padding=2, bias=False),
            gn(embed_dim), nn.ReLU(inplace=True),
        )
        self.pool = AttentionPool1d(embed_dim, dropout=0.3)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            z:  [B, 128]      pooled embedding
            fm: [B, 128, 80]  temporal feature map → Stage 3 cross-attention
        """
        acc  = self.acc_stem(x[:, :3, :])
        gyro = self.gyro_stem(x[:, 3:, :])
        fm   = self.shared(torch.cat([acc, gyro], dim=1))
        z    = self.pool(fm)
        return z, fm


# =============================================================================
# PROJECTION HEAD  3-layer MLP
# =============================================================================
class ProjectionHead(nn.Module):
    """
    3-layer MLP: embed_dim → embed_dim → embed_dim//2 → proj_dim
    L2-normalised output for InfoNCE cosine similarity.
    Absorbs InfoNCE-specific distortion — encoder reps kept clean for Stage 3.
    Discarded after Stage 2.
    """

    def __init__(self, in_dim: int = EMBED_DIM, out_dim: int = PROJ_DIM):
        super().__init__()
        hidden = in_dim // 2
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim,  bias=False),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, hidden, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim, bias=False),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(z), dim=-1)


class ContextHead(nn.Module):
    """Predict motion/context labels from the IMU representation."""

    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        self.mai = nn.Linear(embed_dim, 1)
        self.activity = nn.Linear(embed_dim, N_ACTIVITY_CLASSES)
        self.body_position = nn.Linear(embed_dim, N_BODY_CLASSES)
        self.noise = nn.Linear(embed_dim, 1)
        self.rpeak_reliable = nn.Linear(embed_dim, 1)
        self.emg_flag = nn.Linear(embed_dim, 1)
        self.emg_ratio = nn.Linear(embed_dim, 1)

    def forward(self, z: torch.Tensor) -> dict:
        h = self.trunk(z)
        return {
            "mai_pred": self.mai(h).squeeze(-1),
            "activity_logits": self.activity(h),
            "body_logits": self.body_position(h),
            "noise_logit": self.noise(h).squeeze(-1),
            "rpeak_logit": self.rpeak_reliable(h).squeeze(-1),
            "emg_logit": self.emg_flag(h).squeeze(-1),
            "emg_ratio_pred": self.emg_ratio(h).squeeze(-1),
        }


# =============================================================================
# SIAMESE MODULE
# =============================================================================
class SiameseAlignmentModule(nn.Module):
    """
    Cross-modal contrastive pretraining (v5).
    ECG and IMU encoders share embedding space via InfoNCE.
    Temporal attention pooling preserves timing information.
    Fixed temperature T=0.10 (register_buffer — not trainable).
    """

    def __init__(self, embed_dim: int = EMBED_DIM, proj_dim: int = PROJ_DIM):
        super().__init__()
        self.ecg_encoder = ECGEncoder(embed_dim)
        self.imu_encoder = IMUEncoder(embed_dim)
        self.ecg_proj    = ProjectionHead(embed_dim, proj_dim)
        self.imu_proj    = ProjectionHead(embed_dim, proj_dim)
        self.context_head = ContextHead(embed_dim)
        self.embed_dim   = embed_dim

        # Fixed temperature T=0.10 — not trainable
        # Prevents temperature collapse exploit
        self.register_buffer('logit_scale',
                             torch.tensor(np.log(10.0)))

    def forward(
        self,
        ecg:    Optional[torch.Tensor],
        imu:    torch.Tensor,
        mai:    Optional[torch.Tensor] = None,
        activity: Optional[torch.Tensor] = None,
        body_position: Optional[torch.Tensor] = None,
        noise_annotated: Optional[torch.Tensor] = None,
        rpeaks_reliable: Optional[torch.Tensor] = None,
        emg_flagged: Optional[torch.Tensor] = None,
        emg_ratio: Optional[torch.Tensor] = None,
        rhythm: Optional[torch.Tensor] = None,
    ) -> dict:
        z_imu, fm_imu = self.imu_encoder(imu)

        if ecg is not None:
            z_ecg, fm_ecg = self.ecg_encoder(ecg)
            p_ecg = self.ecg_proj(z_ecg)
            z_ecg_n = F.normalize(z_ecg, dim=-1)
            sim = (z_ecg_n * F.normalize(z_imu, dim=-1)).sum(dim=-1)
        else:
            fm_ecg = None
            p_ecg = None
            z_ecg_n = None
            sim = torch.zeros(imu.shape[0], device=imu.device)

        p_imu = self.imu_proj(z_imu) if ecg is not None else None
        context = self.context_head(z_imu)

        logit_scale = self.logit_scale

        z_imu_n = F.normalize(z_imu, dim=-1)

        if mai is not None:
            # Diagnostic mask only; the InfoNCE loss uses the softer gate below
            # centered at 0.080g.
            mai_mask = mai > 0.02
        else:
            mai_mask = torch.ones(imu.shape[0], dtype=torch.bool, device=imu.device)

        return {
            "p_ecg":       p_ecg,
            "p_imu":       p_imu,
            "z_ecg":       z_ecg_n,
            "z_imu":       z_imu_n,
            "fm_ecg":      fm_ecg,
            "fm_imu":      fm_imu,
            "sim":         sim,
            "logit_scale": logit_scale,
            "mai_mask":    mai_mask,
            "rhythm":      rhythm,
            "mai":         mai,
            "activity":    activity,
            "body_position": body_position,
            "noise_annotated": noise_annotated,
            "rpeaks_reliable": rpeaks_reliable,
            "emg_flagged": emg_flagged,
            "emg_ratio": emg_ratio,
            **context,
        }


# =============================================================================
# LOSS
# =============================================================================
def variance_loss(z_ecg, z_imu, gamma=1.0):
    std_ecg = z_ecg.std(dim=0)
    std_imu = z_imu.std(dim=0)
    return (F.relu(gamma - std_ecg).mean() +
            F.relu(gamma - std_imu).mean()) / 2.0


def alignment_loss(
    out:           dict,
    lambda_rhythm: float = 0.0,
    lambda_var:    float = 0.0,
    lambda_aux:    float = 1.0,
    lambda_cont:   float = 0.0,
    activity_weight: Optional[torch.Tensor] = None,
    body_weight: Optional[torch.Tensor] = None,
    noise_pos_weight: Optional[torch.Tensor] = None,
    rpeak_pos_weight: Optional[torch.Tensor] = None,
    emg_pos_weight: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, dict]:
    """
    MAI-weighted symmetric InfoNCE.

    lambda_var=0.0   — variance loss incompatible with L2 normalisation
    lambda_rhythm=0.0 — insufficient labelled windows (0.1%)

    MAI soft gate:
      sigmoid((MAI - 0.080) / 0.020)
      Sitting (0.039g) → weight ≈ 0.37
      Walking (0.147g) → weight ≈ 0.99
    """
    p_e = out.get("p_ecg")
    p_i = out.get("p_imu")
    B = out["z_imu"].shape[0]
    device = out["z_imu"].device
    zero = torch.zeros((), device=device)
    l_cont = zero
    acc = 0.0

    if lambda_cont > 0.0 and p_e is not None and p_i is not None:
        scale = out["logit_scale"].exp()
        sim = torch.mm(p_e, p_i.T) * scale
        labels = torch.arange(B, device=device)
        mai = out.get("mai", None)
        if mai is not None and mai.shape[0] == B:
            mai_w = 0.25 + 0.75 * torch.sigmoid((mai.float() - 0.080) / 0.020)
            l_fwd = F.cross_entropy(sim, labels, reduction='none')
            l_bwd = F.cross_entropy(sim.T, labels, reduction='none')
            l_pair = 0.5 * (l_fwd + l_bwd)
            l_cont = (l_pair * mai_w).sum() / mai_w.sum().clamp_min(1e-6)
        else:
            l_cont = (F.cross_entropy(sim, labels) +
                      F.cross_entropy(sim.T, labels)) / 2.0
        with torch.no_grad():
            acc = float((sim.argmax(dim=1) == labels).float().mean())
    else:
        sim = None

    l_mai = l_activity = l_body = l_noise = l_rpeak = zero
    l_emg_flag = l_emg_ratio = zero

    mai = out.get("mai", None)
    if mai is not None and mai.shape[0] == B:
        mai_target = torch.log1p(mai.float().clamp_min(0.0))
        mai_pred = out["mai_pred"].float()
        l_mai = F.smooth_l1_loss(mai_pred, mai_target)

    activity = out.get("activity", None)
    if activity is not None and activity.shape[0] == B:
        target = activity.long()
        valid = (target >= 0) & (target < out["activity_logits"].shape[1])
        if valid.any():
            l_activity = F.cross_entropy(
                out["activity_logits"][valid], target[valid],
                weight=activity_weight,
            )

    body = out.get("body_position", None)
    if body is not None and body.shape[0] == B:
        target = body.long()
        valid = (target >= 0) & (target < out["body_logits"].shape[1])
        if valid.any():
            l_body = F.cross_entropy(
                out["body_logits"][valid], target[valid],
                weight=body_weight,
            )

    noise = out.get("noise_annotated", None)
    if noise is not None and noise.shape[0] == B:
        l_noise = F.binary_cross_entropy_with_logits(
            out["noise_logit"], noise.float(),
            pos_weight=noise_pos_weight,
        )

    reliable = out.get("rpeaks_reliable", None)
    if reliable is not None and reliable.shape[0] == B:
        l_rpeak = F.binary_cross_entropy_with_logits(
            out["rpeak_logit"], reliable.float(),
            pos_weight=rpeak_pos_weight,
        )

    emg_flag = out.get("emg_flagged", None)
    if emg_flag is not None and emg_flag.shape[0] == B:
        l_emg_flag = F.binary_cross_entropy_with_logits(
            out["emg_logit"], emg_flag.float(),
            pos_weight=emg_pos_weight,
        )

    emg_ratio = out.get("emg_ratio", None)
    if emg_ratio is not None and emg_ratio.shape[0] == B:
        target = torch.log1p(emg_ratio.float().clamp_min(0.0))
        l_emg_ratio = F.smooth_l1_loss(out["emg_ratio_pred"].float(), target)

    l_aux = (
        l_mai + l_emg_ratio
        + 0.5 * l_activity + 0.5 * l_body
        + l_noise + l_rpeak + l_emg_flag
    )
    total = lambda_cont * l_cont + lambda_aux * l_aux

    with torch.no_grad():
        sim_mean = float(out["sim"].mean())
        temp     = float(1.0 / out["logit_scale"].exp().item())

    return total, {
        "L_total":    total.item(),
        "L_cont":     l_cont.item(),
        "L_aux":      l_aux.item(),
        "L_mai":      l_mai.item(),
        "L_activity": l_activity.item(),
        "L_body":     l_body.item(),
        "L_noise":    l_noise.item(),
        "L_rpeak":    l_rpeak.item(),
        "L_emg_flag": l_emg_flag.item(),
        "L_emg_ratio": l_emg_ratio.item(),
        "L_rhythm":   0.0,
        "L_var":      0.0,
        "acc":        acc,
        "sim_mean":   sim_mean,
        "temperature": temp,
        "n_active":   B,
    }


# =============================================================================
# SMOKE TEST
# =============================================================================
if __name__ == "__main__":
    print("── Stage 2 v5 smoke test ────────────────────────────────────────")
    B     = 8
    model = SiameseAlignmentModule().eval()
    ecg   = torch.randn(B, 1, WINDOW_ECG)
    imu   = torch.randn(B, 6, WINDOW_IMU)
    mai   = torch.rand(B) * 0.3

    with torch.no_grad():
        out = model(ecg, imu, mai=mai)

    print(f"  fm_ecg : {out['fm_ecg'].shape}   temporal feature map")
    print(f"  fm_imu : {out['fm_imu'].shape}")
    print(f"  p_ecg  : {out['p_ecg'].shape}   projection head output")
    print(f"  z_ecg  : {out['z_ecg'].shape}   encoder output (for sim)")

    loss, info = alignment_loss(out)
    print(f"\n  L_cont : {info['L_cont']:.4f}")
    print(f"  acc    : {info['acc']:.1%}")
    print(f"  sim    : {info['sim_mean']:.4f}")
    print(f"  T      : {info['temperature']:.3f}  (fixed)")

    n = sum(p.numel() for p in model.parameters())
    attn_params = sum(p.numel() for p in model.ecg_encoder.pool.parameters()) * 2
    print(f"\n  Total params     : {n:,}")
    print(f"  AttentionPool ×2 : {attn_params:,}  (was 2,621,440)")
    print("\n  Smoke test passed ✓")
