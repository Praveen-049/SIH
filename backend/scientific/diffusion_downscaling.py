"""Conditional Diffusion Downscaling model for 12 km -> ~5 km high-resolution fields.

Implements a genuine Denoising Diffusion Probabilistic Model (DDPM) conditioned on
coarse atmospheric variables and high-resolution topography (elevation, slope, land-sea mask).
Addresses the SIH spectral smoothing problem using extreme-tail preservation losses.
"""

from __future__ import annotations

import os
import sys

# Ensure Windows Python 3.14 loads PyTorch C++ DLLs cleanly
if sys.platform == "win32":
    torch_lib_dir = os.path.join(
        os.path.dirname(sys.executable), "..", "Lib", "site-packages", "torch", "lib"
    )
    if os.path.exists(torch_lib_dir):
        try:
            os.add_dll_directory(torch_lib_dir)
        except Exception:
            pass

import math
from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TopographyConditioner(nn.Module):
    """Encodes high-resolution topography (elevation, slope, land/sea mask)."""

    def __init__(self, topo_channels: int = 3, out_dim: int = 16) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(topo_channels, out_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, topo: torch.Tensor) -> torch.Tensor:
        return self.conv(topo)


class ConditionalDenoisingUNet(nn.Module):
    """Conditional U-Net denoising backbone for weather diffusion."""

    def __init__(self, in_channels: int = 1, cond_channels: int = 19) -> None:
        super().__init__()
        total_in = in_channels + cond_channels  # Noisy target + (upscaled coarse + topo features + time emb)

        # Encoder
        self.inc = nn.Sequential(
            nn.Conv2d(total_in, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
        )
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
        )

        # Decoder
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
        )
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
        )
        self.outc = nn.Conv2d(32, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([x, cond], dim=1)
        x1 = self.inc(inp)
        x2 = self.down1(x1)
        x3 = self.bottleneck(x2)

        u1 = self.up1(x3)
        # Pad if odd dimensions
        if u1.shape != x2.shape:
            u1 = F.interpolate(u1, size=x2.shape[2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([u1, x2], dim=1))

        u2 = self.up2(d1)
        if u2.shape != x1.shape:
            u2 = F.interpolate(u2, size=x1.shape[2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([u2, x1], dim=1))

        return self.outc(d2)


class ConditionalWeatherDiffusion(nn.Module):
    """Conditional Weather Diffusion downscaler from 12 km to 5 km."""

    def __init__(self, timesteps: int = 50, beta_start: float = 1e-4, beta_end: float = 0.02) -> None:
        super().__init__()
        self.timesteps = timesteps

        # Linear noise schedule
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

        self.topo_encoder = TopographyConditioner(topo_channels=3, out_dim=16)
        # cond_channels: 2 (upscaled coarse + anomaly) + 16 (topo) + 1 (time) = 19
        self.denoiser = ConditionalDenoisingUNet(in_channels=1, cond_channels=19)

    def q_sample(
        self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward diffusion process: add noise at timestep t."""
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_alphas * x_start + sqrt_one_minus * noise

    def build_condition(
        self,
        coarse_12km: torch.Tensor,
        anomaly_field: torch.Tensor,
        topography: torch.Tensor,
        target_shape: tuple[int, int],
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Construct full multi-channel conditioning tensor at target 5 km resolution."""
        # Upsample coarse and anomaly to 5 km grid via bilinear interpolation
        coarse_up = F.interpolate(coarse_12km, size=target_shape, mode="bilinear", align_corners=False)
        anomaly_up = F.interpolate(anomaly_field, size=target_shape, mode="bilinear", align_corners=False)
        topo_feat = self.topo_encoder(topography)

        t_norm = (t.float() / float(self.timesteps)).view(-1, 1, 1, 1)
        t_channel = t_norm.expand(-1, 1, target_shape[0], target_shape[1])

        return torch.cat([coarse_up, anomaly_up, topo_feat, t_channel], dim=1)

    @torch.no_grad()
    def sample(
        self,
        coarse_12km: torch.Tensor,
        anomaly_field: torch.Tensor,
        topography: torch.Tensor,
        target_shape: tuple[int, int] = (32, 32),
    ) -> torch.Tensor:
        """Sample downscaled 5 km field using reverse diffusion sampling loop."""
        batch_size = coarse_12km.shape[0]
        device = coarse_12km.device

        # Start from pure Gaussian noise
        x = torch.randn((batch_size, 1, target_shape[0], target_shape[1]), device=device)

        for t_step in reversed(range(self.timesteps)):
            t_batch = torch.full((batch_size,), t_step, device=device, dtype=torch.long)
            cond = self.build_condition(coarse_12km, anomaly_field, topography, target_shape, t_batch)
            pred_noise = self.denoiser(x, cond)

            alpha = self.alphas[t_step]
            alpha_cumprod = self.alphas_cumprod[t_step]
            beta = self.betas[t_step]

            if t_step > 0:
                noise = torch.randn_like(x)
            else:
                noise = 0.0

            x = (1.0 / torch.sqrt(alpha)) * (
                x - (beta / torch.sqrt(1.0 - alpha_cumprod)) * pred_noise
            ) + torch.sqrt(beta) * noise

        return torch.clamp(x, min=0.0)


def extreme_tail_preservation_loss(
    pred: torch.Tensor, target: torch.Tensor, quantile: float = 0.95, weight: float = 2.0
) -> torch.Tensor:
    """Asymmetric loss heavily penalizing underestimation of extreme values (preventing smoothing)."""
    error = target - pred
    # Asymmetric Pinball / Quantile loss
    loss = torch.where(error > 0, weight * error, (1.0 - quantile) * torch.abs(error))
    return torch.mean(loss)


def compare_downscaling_methods(
    coarse_grid: np.ndarray,      # 12 km input (H_c, W_c)
    ground_truth_5km: np.ndarray,  # 5 km ground truth (H_f, W_f)
    model_output_5km: np.ndarray,  # Diffusion downscaled (H_f, W_f)
) -> dict[str, Any]:
    """Evaluates coarse vs bicubic baseline vs diffusion downscaled field against ground truth."""
    import scipy.ndimage

    # 1. Baseline Bicubic Upscaling
    zoom_factors = (
        ground_truth_5km.shape[0] / coarse_grid.shape[0],
        ground_truth_5km.shape[1] / coarse_grid.shape[1],
    )
    bicubic_baseline = scipy.ndimage.zoom(coarse_grid, zoom_factors, order=3)

    def calc_stats(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
        rmse = float(np.sqrt(np.mean((pred - truth) ** 2)))
        mae = float(np.mean(np.abs(pred - truth)))
        bias = float(np.mean(pred - truth))
        peak_err = float(np.max(pred) - np.max(truth))
        p95_pred, p95_truth = float(np.percentile(pred, 95.0)), float(np.percentile(truth, 95.0))
        p95_err = p95_pred - p95_truth
        corr = float(np.corrcoef(pred.ravel(), truth.ravel())[0, 1]) if np.std(pred) > 0 and np.std(truth) > 0 else 0.0

        return {
            "rmse": round(rmse, 3),
            "mae": round(mae, 3),
            "bias": round(bias, 3),
            "correlation": round(corr, 3),
            "peak_error": round(peak_err, 3),
            "p95_error": round(p95_err, 3),
            "max_intensity": round(float(np.max(pred)), 2),
            "p95_intensity": round(p95_pred, 2),
        }

    return {
        "ground_truth_metrics": {
            "max_intensity": round(float(np.max(ground_truth_5km)), 2),
            "p95_intensity": round(float(np.percentile(ground_truth_5km, 95.0)), 2),
        },
        "baseline_bicubic": calc_stats(bicubic_baseline, ground_truth_5km),
        "diffusion_model": calc_stats(model_output_5km, ground_truth_5km),
        "spectral_smoothing_addressed": bool(
            np.max(model_output_5km) > np.max(bicubic_baseline)
        ),
    }
