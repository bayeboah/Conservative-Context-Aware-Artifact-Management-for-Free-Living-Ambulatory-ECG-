"""
denoiser.py — Stage 3: Cross-Attention U-Net ECG Denoiser
==========================================================
Architecture:
  ECG U-Net encoder (trainable) + IMU encoder (frozen from Stage 2)
  → cross-attention at bottleneck (ECG queries IMU for artifact context)
  → ECG U-Net decoder (trainable) with skip connections

The frozen IMU encoder's feature map acts as a frame-by-frame
motion artifact highlighter. Cross-attention allows each ECG
temporal position to query the IMU for whether motion occurred
at that moment — and suppresses it in the reconstruction.

Shapes:
  ECG input:   [B, 1, 10240]   10s at 1024 Hz
  IMU input:   [B, 6,  640]    10s at 64 Hz (acc + gyro)
  ECG output:  [B, 1, 10240]   denoised ECG

Temporal resolution:
  Bottleneck and IMU feature map both at 80 positions (125ms each)
  — matched so cross-attention aligns frame-by-frame
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple
from siamese_alignment import SiameseAlignmentModule

# ── Constants ─────────────────────────────────────────────────────────────────
ECG_FS   = 1024
EMBED_DIM = 128
FEAT_LEN  = 80   # temporal positions (125ms each)


def _safe_torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


# =============================================================================
# Helpers
# =============================================================================
def gn(channels: int, groups: int = 8) -> nn.GroupNorm:
    return nn.GroupNorm(min(groups, channels), channels)


class ConvBlock(nn.Module):
    """Conv1D + GroupNorm + ReLU. Optionally strided."""
    def __init__(self, in_ch, out_ch, kernel=7, stride=1, padding=None):
        super().__init__()
        if padding is None:
            padding = kernel // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, stride=stride,
                      padding=padding, bias=False),
            gn(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResConvBlock(nn.Module):
    """Two Conv1D layers with residual connection."""
    def __init__(self, ch, kernel=7):
        super().__init__()
        p = kernel // 2
        self.block = nn.Sequential(
            nn.Conv1d(ch, ch, kernel, padding=p, bias=False),
            gn(ch), nn.ReLU(inplace=True),
            nn.Conv1d(ch, ch, kernel, padding=p, bias=False),
            gn(ch),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.block(x))


class UpBlock(nn.Module):
    """Upsample + conv. Concatenates skip connection."""
    def __init__(self, in_ch, skip_ch, out_ch, scale=4, kernel=7):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=scale, mode='linear',
                                 align_corners=False)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch, kernel=kernel)
        self.res  = ResConvBlock(out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # Handle any off-by-one from integer strides
        if x.shape[-1] != skip.shape[-1]:
            x = F.interpolate(x, size=skip.shape[-1], mode='linear',
                              align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.res(self.conv(x))


# =============================================================================
# ECG U-Net Encoder
# =============================================================================
class ECGEncoder(nn.Module):
    """
    Strided convolutional encoder producing skip connections
    at multiple temporal resolutions.

    Input:  [B, 1, 10240]
    Skips:  s1 [B, 32, 5120]
            s2 [B, 64, 1280]
            s3 [B, 128, 320]
    Output: [B, 128, 80]  bottleneck (matches IMU feature map)
    """

    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.e1 = nn.Sequential(
            ConvBlock(1,   32, kernel=15, stride=2, padding=7),
            ResConvBlock(32),
        )  # → [B, 32, 5120]

        self.e2 = nn.Sequential(
            ConvBlock(32,  64, kernel=9, stride=4, padding=4),
            ResConvBlock(64),
        )  # → [B, 64, 1280]

        self.e3 = nn.Sequential(
            ConvBlock(64, 128, kernel=7, stride=4, padding=3),
            ResConvBlock(128),
        )  # → [B, 128, 320]

        self.e4 = nn.Sequential(
            ConvBlock(128, embed_dim, kernel=5, stride=4, padding=2),
            ResConvBlock(embed_dim),
        )  # → [B, 128, 80]

    def forward(self, x: torch.Tensor):
        s1 = self.e1(x)    # [B, 32,  5120]
        s2 = self.e2(s1)   # [B, 64,  1280]
        s3 = self.e3(s2)   # [B, 128,  320]
        bt = self.e4(s3)   # [B, 128,   80]  bottleneck
        return bt, [s1, s2, s3]


# =============================================================================
# Cross-Attention Bottleneck
# =============================================================================
class CrossAttention1d(nn.Module):
    """
    Multi-head cross-attention over temporal positions.

    ECG features (Q) attend to IMU features (K, V).
    At each of the 80 temporal positions, the ECG feature
    queries which IMU motion pattern is occurring — and uses
    that context to guide motion-artifact suppression.

    Residual connection preserves the ECG signal path;
    the attention output modulates rather than replaces it.
    """

    def __init__(self, embed_dim: int = EMBED_DIM, n_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        self.attn    = nn.MultiheadAttention(embed_dim, n_heads,
                                             dropout=dropout,
                                             batch_first=True)
        self.norm_q  = nn.LayerNorm(embed_dim)
        self.norm_out= nn.LayerNorm(embed_dim)
        self.gate    = nn.Parameter(torch.zeros(1))  # learnable gate

    def forward(self, ecg_feat: torch.Tensor,
                imu_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ecg_feat: [B, C, T]  ECG bottleneck features
            imu_feat: [B, C, T]  IMU feature map (frozen encoder)
        Returns:
            out: [B, C, T]  IMU-conditioned ECG features
        """
        # [B, C, T] → [B, T, C] for MultiheadAttention
        q = ecg_feat.permute(0, 2, 1)   # [B, 80, 128]
        k = imu_feat.permute(0, 2, 1)   # [B, 80, 128]
        v = imu_feat.permute(0, 2, 1)   # [B, 80, 128]

        q = self.norm_q(q)
        attn_out, _ = self.attn(q, k, v)  # [B, 80, 128]

        # Gated residual — sigmoid bounds IMU contribution to [0, 1]
        # tanh was allowing gate to exceed 1 and flip sign, causing
        # val spikes when IMU features over-dominated reconstruction.
        out = q + torch.sigmoid(self.gate) * attn_out
        out = self.norm_out(out)

        return out.permute(0, 2, 1)      # [B, 128, 80]


# =============================================================================
# ECG U-Net Decoder
# =============================================================================
class ECGDecoder(nn.Module):
    """
    Upsampling decoder with skip connections from ECG encoder.

    Skip connections preserve fine-grained ECG morphology
    (QRS timing, P-wave shape, ST segment) that the encoder
    compresses but must not distort.

    Input:  [B, 128, 80]   IMU-conditioned bottleneck
    Skips:  s3 [B, 128, 320]
            s2 [B, 64,  1280]
            s1 [B, 32,  5120]
    Output: [B, 1, 10240]  denoised ECG
    """

    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.d4 = UpBlock(embed_dim, 128, 128, scale=4)  # 80→320
        self.d3 = UpBlock(128, 64,  64,  scale=4)         # 320→1280
        self.d2 = UpBlock(64,  32,  32,  scale=4)         # 1280→5120
        self.d1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            nn.Conv1d(32, 1, kernel_size=7, padding=3),  # 5120→10240
        )

    def forward(self, x: torch.Tensor,
                skips: List[torch.Tensor]) -> torch.Tensor:
        s1, s2, s3 = skips
        x = self.d4(x, s3)   # [B, 128, 320]
        x = self.d3(x, s2)   # [B, 64,  1280]
        x = self.d2(x, s1)   # [B, 32,  5120]
        x = self.d1(x)        # [B, 1,   10240]
        return x


# =============================================================================
# Complete Cross-Attention U-Net
# =============================================================================
class CrossAttentionUNet(nn.Module):
    """
    Stage 3: IMU-conditioned ECG denoiser.

    The IMU encoder is loaded from the Stage 2 checkpoint and
    frozen — its weights encode the temporal correspondence
    between chest motion and ECG artifacts learned in Stage 2.

    Only the ECG encoder, cross-attention, and decoder are trained.
    """

    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.ecg_encoder   = ECGEncoder(embed_dim)
        self.cross_attn    = CrossAttention1d(embed_dim, n_heads=8)
        self.ecg_decoder   = ECGDecoder(embed_dim)

        # IMU encoder — loaded separately from Stage 2 checkpoint
        # Registered as a module so it moves to device with .to(device)
        # but weights are frozen after loading
        self.imu_encoder   = None   # set by load_imu_encoder()
        self.context_head  = None

    def load_imu_encoder(self, stage2_ckpt_path: str, device: torch.device):
        """
        Load frozen IMU encoder from Stage 2 siamese checkpoint.
        Only the IMUEncoder weights are extracted —
        the projection head and logit_scale are discarded.
        """
        siamese = SiameseAlignmentModule()
        ckpt    = _safe_torch_load(stage2_ckpt_path, map_location='cpu')
        # Stage 2 checkpoint uses 'model_state' key (set in train_stage2.py)
        state   = (ckpt.get('model_state') or
                   ckpt.get('model_state_dict') or
                   ckpt.get('state_dict') or
                   ckpt)   # fallback: raw state dict
        missing, unexpected = siamese.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "Stage 2 checkpoint key mismatch while loading frozen IMU encoder.\n"
                f"  missing keys: {sorted(missing)[:10]}\n"
                f"  unexpected keys: {sorted(unexpected)[:10]}"
            )

        self.imu_encoder = siamese.imu_encoder.to(device)
        self.context_head = siamese.context_head.to(device)
        # Freeze all IMU encoder parameters
        for p in self.imu_encoder.parameters():
            p.requires_grad = False
        for p in self.context_head.parameters():
            p.requires_grad = False
        self.imu_encoder.eval()
        self.context_head.eval()

        print(f"  IMU encoder loaded from {stage2_ckpt_path}")
        n_frozen = sum(p.numel() for p in self.imu_encoder.parameters())
        print(f"  Frozen params: {n_frozen:,}")

    def encode_imu_context(self, imu: torch.Tensor):
        with torch.no_grad():
            z_imu, fm_imu = self.imu_encoder(imu)
            context = self.context_head(z_imu) if self.context_head is not None else {}
        return z_imu, fm_imu, context

    def forward(self, ecg: torch.Tensor,
                imu: torch.Tensor,
                return_context: bool = False) -> torch.Tensor:
        """
        Args:
            ecg: [B, 1, 10240]  noisy ECG (+ augmentation noise during training)
            imu: [B, 6,  640]   concurrent IMU signal
        Returns:
            out: [B, 1, 10240]  denoised ECG
        """
        # Encode ECG with skip connections
        bottleneck, skips = self.ecg_encoder(ecg)  # [B,128,80], [s1,s2,s3]

        # Extract IMU feature map (frozen — no gradients)
        _, fm_imu, context = self.encode_imu_context(imu)

        # Cross-attention: ECG queries IMU for motion context
        conditioned = self.cross_attn(bottleneck, fm_imu)  # [B, 128, 80]

        # Decode with skip connections
        out = self.ecg_decoder(conditioned, skips)   # [B, 1, 10240]
        if return_context:
            return out, context
        return out


class ResidualArtifactRemover(CrossAttentionUNet):
    """
    IMU-conditioned residual artifact remover.

    The network predicts only the removable artifact residual. The cleaned ECG is
    computed as input ECG minus a bounded, motion-gated residual. This prevents
    the model from freely rewriting cardiac morphology when no clean target is
    available.
    """

    def __init__(self, embed_dim: int = EMBED_DIM, max_residual: float = 0.35):
        super().__init__(embed_dim)
        self.max_residual = float(max_residual)

    @staticmethod
    def motion_gate(mai: torch.Tensor,
                    noise_annotated: torch.Tensor = None,
                    low: float = 0.02,
                    high: float = 0.12) -> torch.Tensor:
        gate = ((mai.float() - low) / max(high - low, 1e-6)).clamp(0.0, 1.0)
        if noise_annotated is not None:
            gate = torch.maximum(gate, noise_annotated.float().clamp(0.0, 1.0))
        return gate.view(-1, 1, 1)

    @staticmethod
    def context_gate(context: dict,
                     noise_annotated: torch.Tensor = None,
                     floor: float = 0.05) -> torch.Tensor:
        gates = []
        if context:
            if "emg_logit" in context:
                gates.append(torch.sigmoid(context["emg_logit"]))
            if "noise_logit" in context:
                gates.append(torch.sigmoid(context["noise_logit"]))
            if "emg_ratio_pred" in context:
                emg_ratio = torch.expm1(context["emg_ratio_pred"].float()).clamp_min(0.0)
                gates.append((emg_ratio / 0.35).clamp(0.0, 1.0))
            if "mai_pred" in context:
                mai = torch.expm1(context["mai_pred"].float()).clamp_min(0.0)
                gates.append(((mai - 0.02) / 0.10).clamp(0.0, 1.0))
        if noise_annotated is not None:
            gates.append(noise_annotated.float().clamp(0.0, 1.0))
        if not gates:
            return None
        gate = torch.stack(gates, dim=0).amax(dim=0).clamp(floor, 1.0)
        return gate.view(-1, 1, 1)

    def forward(self, ecg: torch.Tensor, imu: torch.Tensor,
                mai: torch.Tensor = None,
                noise_annotated: torch.Tensor = None,
                return_residual: bool = False):
        raw_residual, context = super().forward(ecg, imu, return_context=True)
        residual = self.max_residual * torch.tanh(raw_residual)
        ctx_gate = self.context_gate(context, noise_annotated)
        if ctx_gate is not None:
            residual = residual * ctx_gate
        if mai is not None:
            residual = residual * self.motion_gate(mai, noise_annotated)
        clean = ecg - residual
        if return_residual:
            return clean, residual
        return clean


def _stft_log_bandpower(x: torch.Tensor, fs: int, flo: float, fhi: float,
                        n_fft: int = 512, hop_length: int = 128):
    sig = x.float().squeeze(1)
    window = torch.hann_window(n_fft, device=x.device, dtype=sig.dtype)
    spec = torch.stft(
        sig,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        return_complex=True,
    )
    freqs = torch.fft.rfftfreq(n_fft, d=1.0 / fs).to(x.device)
    mask = (freqs >= flo) & (freqs <= fhi)
    power = spec.abs().square()[:, mask, :].mean(dim=(1, 2))
    return torch.log(power.clamp_min(1e-8))


def generate_pseudo_target(
    ecg: torch.Tensor,
    fs: int = ECG_FS,
    baseline_cut: float = 0.5,
    hf_cut: float = 45.0,
    baseline_edge: float = 0.20,
    hf_edge: float = 5.0,
    baseline_scale: float = 0.25,
    hf_scale: float = 0.10,
) -> torch.Tensor:
    """
    Conservative differentiable residual-teacher pseudo-clean target.

    Instead of asking the network to copy a fully bandpassed ECG, estimate only
    the most defensible artifact residuals: slow baseline drift and extreme
    high-frequency noise. The ECG middle band is left untouched.
    """
    sig = ecg.float().squeeze(1)
    n = sig.shape[-1]
    freqs = torch.fft.rfftfreq(n, d=1.0 / fs).to(ecg.device)

    fft = torch.fft.rfft(sig, dim=-1)

    baseline_mask = torch.sigmoid(
        (baseline_cut - freqs) / max(baseline_edge, 1e-6)
    ).to(sig.dtype)
    hf_mask = torch.sigmoid(
        (freqs - hf_cut) / max(hf_edge, 1e-6)
    ).to(sig.dtype)

    baseline = torch.fft.irfft(fft * baseline_mask.view(1, -1), n=n, dim=-1)
    hf_noise = torch.fft.irfft(fft * hf_mask.view(1, -1), n=n, dim=-1)
    pseudo = sig - baseline_scale * baseline - hf_scale * hf_noise
    return pseudo.unsqueeze(1).to(dtype=ecg.dtype)


def gated_teacher_denoise(
    ecg: torch.Tensor,
    mai: torch.Tensor,
    noise_annotated: torch.Tensor = None,
    fs: int = ECG_FS,
    gate_low: float = 0.02,
    gate_high: float = 0.12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Official Stage 3 gated-teacher inference baseline.

    Low-risk windows remain identity. High-risk windows are moved toward the
    conservative pseudo-clean target according to the same MAI/noise gate used
    by the residual model.
    """
    pseudo = generate_pseudo_target(ecg, fs=fs)
    gate = ResidualArtifactRemover.motion_gate(
        mai, noise_annotated, low=gate_low, high=gate_high
    )
    clean = (1.0 - gate) * ecg + gate * pseudo
    residual = ecg - clean
    return clean, residual


def residual_artifact_loss(
    ecg: torch.Tensor,
    clean: torch.Tensor,
    residual: torch.Tensor,
    mai: torch.Tensor,
    noise_annotated: torch.Tensor,
    rpeaks_reliable: torch.Tensor,
    ecg_fs: int = ECG_FS,
    lambda_identity: float = 4.0,
    lambda_residual: float = 0.5,
    lambda_smooth: float = 0.2,
    lambda_qrs: float = 1.0,
    lambda_motion: float = 0.5,
) -> Tuple[torch.Tensor, dict]:
    """
    Self-supervised loss for residual artifact removal without clean targets.

    Low-motion, reliable windows anchor the identity mapping. High-motion/noise
    windows permit removal by pulling the prediction toward a conservative
    residual teacher. The residual remains bounded, smooth, and protected by a
    QRS-band preservation term.
    """
    mai = mai.float()
    noise = noise_annotated.float()
    reliable = rpeaks_reliable.float()

    low_motion = ((mai < 0.02) & (noise < 0.5) & (reliable > 0.5)).float()
    high_motion = ((mai > 0.08) | (noise > 0.5)).float()

    residual_energy = residual.square().mean(dim=[1, 2])
    clean_delta = (clean - ecg).abs().mean(dim=[1, 2])

    l_identity = ((residual_energy + clean_delta) * low_motion).sum()
    l_identity = l_identity / low_motion.sum().clamp_min(1.0)

    allowed = ResidualArtifactRemover.motion_gate(mai, noise).squeeze(-1).squeeze(-1)
    l_residual = (residual_energy * (1.0 - allowed)).mean()

    diff = residual[..., 1:] - residual[..., :-1]
    l_smooth = diff.abs().mean()

    ecg_qrs = _stft_log_bandpower(ecg, ecg_fs, 5.0, 15.0)
    clean_qrs = _stft_log_bandpower(clean, ecg_fs, 5.0, 15.0)
    qrs_err = (clean_qrs - ecg_qrs).abs()
    preserve_weight = (1.0 - 0.5 * high_motion).clamp(0.5, 1.0)
    l_qrs = (qrs_err * preserve_weight).mean()

    teacher_gate = high_motion
    pseudo_clean = generate_pseudo_target(ecg, fs=ecg_fs)
    #teacher_residual = ecg - pseudo_clean
    #target_residual = teacher_gate.view(-1, 1, 1) * teacher_residual
    teacher_err = (clean - pseudo_clean).square().mean(dim=[1, 2])
    l_motion = (teacher_err * teacher_gate).sum()
    l_motion = l_motion / teacher_gate.sum().clamp_min(1.0)

    total = (
        lambda_identity * l_identity
        + lambda_residual * l_residual
        + lambda_smooth * l_smooth
        + lambda_qrs * l_qrs
        + lambda_motion * l_motion
    )

    return total, {
        "L_total": total.item(),
        "L_identity": l_identity.item(),
        "L_residual": l_residual.item(),
        "L_smooth": l_smooth.item(),
        "L_qrs": l_qrs.item(),
        "L_motion": l_motion.item(),
        "L_teacher": l_motion.item(),
        "residual_rms": residual.square().mean().sqrt().item(),
        "residual_rms_low": (
            (residual_energy * low_motion).sum()
            / low_motion.sum().clamp_min(1.0)
        ).sqrt().item(),
        "residual_rms_high": (
            (residual_energy * high_motion).sum()
            / high_motion.sum().clamp_min(1.0)
        ).sqrt().item(),
        "low_motion_pct": float(low_motion.mean().item() * 100.0),
        "high_motion_pct": float(high_motion.mean().item() * 100.0),
    }


# =============================================================================
# Loss Function
# =============================================================================
def denoising_loss(
    output:     torch.Tensor,   # [B, 1, 10240]  model output
    target:     torch.Tensor,   # [B, 1, 10240]  clean target
    weights:    torch.Tensor,   # [B]             per-window loss weight
    ecg_fs:     int   = ECG_FS,
    lambda_mse: float = 1.0,
    lambda_freq:float = 1.0,  # balanced: log-magnitude keeps freq in ~1-3 range
) -> Tuple[torch.Tensor, dict]:
    """
    Weighted MSE + spectral L1 loss.

    L_mse:  mean squared error in time domain
    L_freq: L1 error in the 5-40 Hz band
            This band contains both QRS signal AND motion artifact.
            The loss penalises spectral distortion, pushing the model
            to preserve cardiac spectral content while suppressing
            the IMU-correlated artifact component.

    weights: from loss_weight field in Stage 1 HDF5
             clean=1.0, EMG-flagged=0.4, noise=0.1
    """
    B = output.shape[0]

    # ── Time-domain MSE ───────────────────────────────────────────────────────
    mse_per_sample = ((output - target) ** 2).mean(dim=[1, 2])   # [B]
    l_mse = (mse_per_sample * weights).mean()

    # ── Log-magnitude spectral L1 in 5-40 Hz ────────────────────────────────
    # Cast to float32 — cuFFT in FP16 requires power-of-2 signal lengths.
    # 10240 = 10 × 1024 is not a power of 2, so FP16 FFT raises RuntimeError.
    #
    # Log-magnitude compresses dynamic range so the spectral loss stays
    # on a stable scale throughout training regardless of signal amplitude.
    # Raw magnitude FFT error varies from ~8 (epoch 1) to ~0.7 (convergence),
    # making any fixed lambda unstable. log(1+|FFT|) stays in range ~1-3.
    N      = output.shape[-1]
    freqs  = torch.fft.rfftfreq(N, d=1.0 / ecg_fs).to(output.device)
    mask   = (freqs >= 5.0) & (freqs <= 40.0)

    O_fft  = torch.fft.rfft(output.float().squeeze(1), dim=-1)   # [B, N//2+1]
    T_fft  = torch.fft.rfft(target.float().squeeze(1), dim=-1)

    # Log-magnitude: log(1 + |X|) for stability; difference in log space
    O_log  = torch.log1p(O_fft.abs())
    T_log  = torch.log1p(T_fft.abs())
    freq_err = (O_log - T_log)[..., mask].abs().mean(dim=-1)   # [B]
    l_freq   = (freq_err * weights).mean()

    total = lambda_mse * l_mse + lambda_freq * l_freq

    return total, {
        "L_total": total.item(),
        "L_mse":   l_mse.item(),
        "L_freq":  l_freq.item(),
    }


# =============================================================================
# Smoke Test
# =============================================================================
if __name__ == "__main__":
    print("── Stage 3 denoiser smoke test ──────────────────────────────")
    B   = 4
    ecg = torch.randn(B, 1, 10240)
    imu = torch.randn(B, 6,  640)
    w   = torch.ones(B)

    model = CrossAttentionUNet()

    # Simulate frozen IMU encoder without checkpoint
    from siamese_alignment import IMUEncoder
    model.imu_encoder = IMUEncoder(EMBED_DIM)
    for p in model.imu_encoder.parameters():
        p.requires_grad = False

    out  = model(ecg, imu)
    print(f"  Input ECG : {ecg.shape}")
    print(f"  Output ECG: {out.shape}")

    loss, info = denoising_loss(out, ecg, w)
    print(f"  L_total : {info['L_total']:.4f}")
    print(f"  L_mse   : {info['L_mse']:.4f}")
    print(f"  L_freq  : {info['L_freq']:.4f}")

    n_train  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in model.imu_encoder.parameters())
    print(f"\n  Trainable params : {n_train:,}")
    print(f"  Frozen IMU params: {n_frozen:,}")
    print("\n  Smoke test passed ✓")
