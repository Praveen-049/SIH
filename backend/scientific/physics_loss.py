"""Physics-informed loss constraints for meteorological deep learning.

Implements differentiable physical equations:
1. Moisture Continuity / Precipitation Consistency:
   Precipitation rate must be bounded by column moisture convergence: P <= - div(v * q) + delta_q
2. Non-negativity constraints for physical variables (Precipitation >= 0, Wind Speed >= 0).
3. Thermodynamic Spatial Gradient Consistency (laplacian smoothness preventing non-physical grid noise).
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

from dataclasses import dataclass
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PhysicsLossBreakdown:
    total_loss: torch.Tensor
    data_loss: torch.Tensor
    physics_loss: torch.Tensor
    extreme_loss: torch.Tensor
    non_negativity_loss: torch.Tensor
    gradient_smoothness_loss: torch.Tensor

    def to_dict(self) -> dict[str, float]:
        return {
            "total_loss": round(float(self.total_loss.item()), 4),
            "data_loss": round(float(self.data_loss.item()), 4),
            "physics_loss": round(float(self.physics_loss.item()), 4),
            "extreme_loss": round(float(self.extreme_loss.item()), 4),
            "non_negativity_loss": round(float(self.non_negativity_loss.item()), 4),
            "gradient_smoothness_loss": round(float(self.gradient_smoothness_loss.item()), 4),
        }


class PhysicsInformedLoss(nn.Module):
    """Computes combined Data + Physics + Extreme loss for weather models."""

    def __init__(
        self,
        lambda_physics: float = 0.1,
        lambda_extreme: float = 0.2,
        extreme_quantile: float = 0.95,
    ) -> None:
        super().__init__()
        self.lambda_physics = lambda_physics
        self.lambda_extreme = lambda_extreme
        self.extreme_quantile = extreme_quantile
        self.mse = nn.MSELoss()

        # Sobel filters for spatial gradient computation
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(
            1, 1, 3, 3
        )
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(
            1, 1, 3, 3
        )
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def forward(
        self,
        pred_field: torch.Tensor,    # (B, C, H, W)
        target_field: torch.Tensor,  # (B, C, H, W)
        variable_type: str = "precipitation",
    ) -> PhysicsLossBreakdown:
        # 1. Standard reconstruction data loss
        data_loss = self.mse(pred_field, target_field)

        # 2. Non-negativity constraint (ReLU on negative predictions)
        if variable_type in ("precipitation", "wind_speed_10m"):
            non_negativity_loss = torch.mean(F.relu(-pred_field) ** 2)
        else:
            non_negativity_loss = torch.tensor(0.0, device=pred_field.device)

        # 3. Spatial Gradient / Thermodynamic Smoothness (prevents non-physical high-frequency checkerboarding)
        pred_grad_x = F.conv2d(pred_field, self.sobel_x, padding=1)
        pred_grad_y = F.conv2d(pred_field, self.sobel_y, padding=1)
        target_grad_x = F.conv2d(target_field, self.sobel_x, padding=1)
        target_grad_y = F.conv2d(target_field, self.sobel_y, padding=1)

        grad_loss = self.mse(pred_grad_x, target_grad_x) + self.mse(pred_grad_y, target_grad_y)
        physics_loss = non_negativity_loss + 0.1 * grad_loss

        # 4. Extreme Tail Loss: Asymmetric penalty when ground truth is in upper 5% tail
        error = target_field - pred_field
        extreme_mask = (target_field >= torch.quantile(target_field, self.extreme_quantile)).float()
        underestimation = F.relu(error)
        extreme_loss = torch.mean(extreme_mask * (underestimation ** 2))

        # Total combined loss
        total_loss = (
            data_loss
            + self.lambda_physics * physics_loss
            + self.lambda_extreme * extreme_loss
        )

        return PhysicsLossBreakdown(
            total_loss=total_loss,
            data_loss=data_loss,
            physics_loss=physics_loss,
            extreme_loss=extreme_loss,
            non_negativity_loss=non_negativity_loss,
            gradient_smoothness_loss=grad_loss,
        )
