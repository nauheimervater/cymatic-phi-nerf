"""
Core positional encodings, volumetric NeRF architecture, 3D Nyquist bandwidth
diagnostics, and held-out generalization benchmarks for Cymatic Phi-NeRF.

Author: Thomas Nauheimer (2026)
License: MIT
"""

from __future__ import annotations

import math
import time
from typing import Callable, Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PHI: float = (1.0 + math.sqrt(5.0)) / 2.0


def compute_3d_nyquist_frequency(num_samples: int, box_side_length: float = 2.0) -> Dict[str, float]:
    """
    Compute the mean linear sampling spacing Delta_x and 3D Nyquist angular frequency
    omega_Nyquist = pi / Delta_x for N uniform volumetric samples in [-L/2, L/2]^3.
    """
    samples_per_axis = float(num_samples) ** (1.0 / 3.0)
    delta_x = float(box_side_length) / samples_per_axis
    f_nyquist_cycles = 1.0 / (2.0 * delta_x)
    omega_nyquist_rad = 2.0 * math.pi * f_nyquist_cycles
    return {
        "num_samples": float(num_samples),
        "samples_per_axis": samples_per_axis,
        "mean_spacing_delta_x": delta_x,
        "nyquist_cycles_per_unit": f_nyquist_cycles,
        "nyquist_omega_over_pi": omega_nyquist_rad / math.pi,
    }


def diophantine_phase_collision_metric(base: float, num_freqs: int = 8, max_harmonic: int = 8) -> float:
    """
    Quantify low-order integer harmonic commensurability across frequency bands b^k
    (0 <= j < k < num_freqs) by measuring the mean inverse distance of ratios b^(k-j)
    to the nearest integer 1 <= m <= max_harmonic:
        C(b) = mean_{0 <= j < k < L} 1 / (min_{1 <= m <= M} |b^(k-j) - m| + 1e-3).
    Dyadic bases (b = 2) have exact integer ratios 2^(k-j) in Z, causing maximal harmonic
    phase locking, whereas the Golden Ratio (b = Phi) minimizes integer collisions via its
    continued fraction [1; 1, 1, ...].
    """
    scores = []
    for j in range(num_freqs):
        for k in range(j + 1, num_freqs):
            ratio = float(base) ** (k - j)
            min_dist = min(abs(ratio - float(m)) for m in range(1, max_harmonic + 1))
            scores.append(1.0 / (min_dist + 1e-3))
    return float(np.mean(scores))


class StandardNeRFEncoding(nn.Module):
    """
    Standard dyadic NeRF positional encoding (Mildenhall et al., ECCV 2020):
        gamma(p) = [p, sin(2^0 pi p), cos(2^0 pi p), ..., sin(2^(L-1) pi p), cos(2^(L-1) pi p)].
    """

    def __init__(self, in_dims: int = 3, num_freqs: int = 8) -> None:
        super().__init__()
        self.in_dims = in_dims
        self.num_freqs = num_freqs
        freq_bands = 2.0 ** torch.arange(num_freqs).float()
        self.register_buffer("freq_bands", freq_bands)
        self.out_dims = in_dims + 2 * in_dims * num_freqs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = [x]
        for freq in self.freq_bands:
            out.append(torch.sin(freq * math.pi * x))
            out.append(torch.cos(freq * math.pi * x))
        return torch.cat(out, dim=-1)


class CymaticPhiEncoding(nn.Module):
    """
    Cymatic Golden-Ratio (Phi^k) Positional Encoding:
      1. Replaces dyadic 2^k steps with quasi-periodic Golden-Ratio powers Phi^k
         (log_2(Phi) = 0.6942 octaves/band), increasing band density below the 3D
         Nyquist limit and breaking integer harmonic commensurability.
      2. Optionally modulates high-frequency bands with a polar standing-wave envelope
         w_m(theta) = 0.8 + 0.2 * |sin(m * theta)| (m_harmonics > 0).
    """

    def __init__(self, in_dims: int = 3, num_freqs: int = 8, m_harmonics: int = 4) -> None:
        super().__init__()
        self.in_dims = in_dims
        self.num_freqs = num_freqs
        self.m_harmonics = m_harmonics
        freq_bands = (PHI ** torch.arange(num_freqs).float())
        self.register_buffer("freq_bands", freq_bands)
        self.out_dims = in_dims + 2 * in_dims * num_freqs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = [x]
        if self.m_harmonics > 0:
            r = torch.norm(x, dim=-1, keepdim=True) + 1e-8
            theta = torch.acos(torch.clamp(x[..., 2:3] / r, -1.0, 1.0))
            resonance = 0.8 + 0.2 * torch.abs(torch.sin(float(self.m_harmonics) * theta))
        else:
            resonance = 1.0

        for freq in self.freq_bands:
            out.append(torch.sin(freq * math.pi * x) * resonance)
            out.append(torch.cos(freq * math.pi * x) * resonance)
        return torch.cat(out, dim=-1)


class GeneralGeometricEncoding(nn.Module):
    """Geometric progression encoding b^k pi for arbitrary base b > 1."""

    def __init__(self, in_dims: int = 3, num_freqs: int = 8, base: float = PHI) -> None:
        super().__init__()
        self.in_dims = in_dims
        self.num_freqs = num_freqs
        self.base = float(base)
        freq_bands = (self.base ** torch.arange(num_freqs).float())
        self.register_buffer("freq_bands", freq_bands)
        self.out_dims = in_dims + 2 * in_dims * num_freqs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = [x]
        for freq in self.freq_bands:
            out.append(torch.sin(freq * math.pi * x))
            out.append(torch.cos(freq * math.pi * x))
        return torch.cat(out, dim=-1)


class NeRFModel(nn.Module):
    """4-layer coordinate MLP predicting RGB radiance and volumetric opacity sigma."""

    def __init__(self, encoder: nn.Module, hidden_dim: int = 128) -> None:
        super().__init__()
        self.encoder = encoder
        in_dim = int(encoder.out_dims)
        self.pts_linears = nn.ModuleList([
            nn.Linear(in_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        ])
        self.density_layer = nn.Linear(hidden_dim, 1)
        self.rgb_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        for layer in self.pts_linears:
            h = F.relu(layer(h))
        sigma = F.relu(self.density_layer(h))
        rgb = self.rgb_layer(h)
        return rgb, sigma


def scene_phi_shell(pts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Spherical shell (r = 0.7) with high-frequency radial and angular harmonics."""
    x, y, z = pts[..., 0], pts[..., 1], pts[..., 2]
    r = torch.sqrt(x**2 + y**2 + z**2)
    theta = torch.atan2(y, x)
    phi = torch.acos(torch.clamp(z / (r + 1e-8), -1.0, 1.0))

    density = torch.exp(-((r - 0.7) ** 2) / 0.02)
    pattern = torch.sin(6.0 * theta + 4.0 * phi) * torch.cos(PHI * 8.0 * r)
    density = density * (0.5 + 0.5 * pattern)

    r_c = 0.5 + 0.5 * torch.sin(5.0 * theta)
    g_c = 0.5 + 0.5 * torch.cos(5.0 * phi)
    b_c = 0.5 + 0.5 * torch.sin(PHI * 10.0 * r)
    rgb = torch.stack([r_c, g_c, b_c], dim=-1)
    return rgb, density.unsqueeze(-1)


def scene_gyroid_neutral(pts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Neutral 3D Schoen Gyroid implicit minimal surface with integer-wavenumber RGB field
    (contains zero dependence on PHI in the ground-truth scene).
    """
    x, y, z = pts[..., 0], pts[..., 1], pts[..., 2]
    k = 3.0 * math.pi
    gyroid = (
        torch.sin(k * x) * torch.cos(k * y)
        + torch.sin(k * y) * torch.cos(k * z)
        + torch.sin(k * z) * torch.cos(k * x)
    )
    density = torch.exp(-(gyroid ** 2) / 0.15)
    r_c = 0.5 + 0.5 * torch.sin(3.0 * math.pi * x) * torch.cos(2.0 * math.pi * y)
    g_c = 0.5 + 0.5 * torch.sin(3.0 * math.pi * y) * torch.cos(2.0 * math.pi * z)
    b_c = 0.5 + 0.5 * torch.sin(3.0 * math.pi * z) * torch.cos(2.0 * math.pi * x)
    rgb = torch.stack([r_c, g_c, b_c], dim=-1)
    return rgb, density.unsqueeze(-1)


def train_and_evaluate_nerf(
    encoder: nn.Module,
    pts_train: torch.Tensor,
    gt_rgb_train: torch.Tensor,
    gt_sigma_train: torch.Tensor,
    pts_test: torch.Tensor | None = None,
    gt_rgb_test: torch.Tensor | None = None,
    gt_sigma_test: torch.Tensor | None = None,
    steps: int = 300,
    lr: float = 2e-3,
    hidden_dim: int = 128,
) -> Dict[str, float]:
    """
    Train a NeRFModel for `steps` iterations and evaluate both training and held-out
    test reconstruction metrics (total loss, RGB MSE, and PSNR in dB).
    """
    model = NeRFModel(encoder, hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)

    t0 = time.perf_counter()
    with torch.no_grad():
        h_train = model.encoder(pts_train)
        h_test = model.encoder(pts_test) if pts_test is not None else None

    for _ in range(steps):
        model.train()
        h = h_train
        for layer in model.pts_linears:
            h = F.relu(layer(h))
        pred_sigma = F.relu(model.density_layer(h))
        pred_rgb = model.rgb_layer(h)

        loss_rgb = F.mse_loss(pred_rgb, gt_rgb_train)
        loss_sigma = F.mse_loss(pred_sigma, gt_sigma_train)
        loss = loss_rgb + 2.0 * loss_sigma

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    elapsed = time.perf_counter() - t0
    model.eval()
    with torch.no_grad():
        h = h_train
        for layer in model.pts_linears:
            h = F.relu(layer(h))
        tr_sigma = F.relu(model.density_layer(h))
        tr_rgb = model.rgb_layer(h)
        tr_rgb_mse = F.mse_loss(tr_rgb, gt_rgb_train).item()
        tr_loss = (tr_rgb_mse + 2.0 * F.mse_loss(tr_sigma, gt_sigma_train).item())
        tr_psnr = -10.0 * math.log10(max(tr_rgb_mse, 1e-10))

        if h_test is not None and gt_rgb_test is not None and gt_sigma_test is not None:
            ht = h_test
            for layer in model.pts_linears:
                ht = F.relu(layer(ht))
            te_sigma = F.relu(model.density_layer(ht))
            te_rgb = model.rgb_layer(ht)
            te_rgb_mse = F.mse_loss(te_rgb, gt_rgb_test).item()
            te_loss = (te_rgb_mse + 2.0 * F.mse_loss(te_sigma, gt_sigma_test).item())
            te_psnr = -10.0 * math.log10(max(te_rgb_mse, 1e-10))
        else:
            te_rgb_mse = tr_rgb_mse
            te_loss = tr_loss
            te_psnr = tr_psnr

    return {
        "train_loss": float(tr_loss),
        "train_rgb_mse": float(tr_rgb_mse),
        "train_psnr_db": float(tr_psnr),
        "test_loss": float(te_loss),
        "test_rgb_mse": float(te_rgb_mse),
        "test_psnr_db": float(te_psnr),
        "elapsed_s": float(elapsed),
    }
