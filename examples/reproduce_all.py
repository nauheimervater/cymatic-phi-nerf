"""
Reproduce all theoretical diagnostics and volumetric NeRF benchmarks for:
  "Cymatic Phi-NeRF: Quasi-Periodic Golden-Ratio Positional Encoding and 3D Nyquist
   Anti-Aliasing in Neural Radiance Fields"
  Author: Thomas Nauheimer (2026)
"""

from __future__ import annotations

import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cymatic_nerf import (
    PHI,
    StandardNeRFEncoding,
    CymaticPhiEncoding,
    GeneralGeometricEncoding,
    scene_phi_shell,
    scene_gyroid_neutral,
    compute_3d_nyquist_frequency,
    diophantine_phase_collision_metric,
    train_and_evaluate_nerf,
)


def main() -> None:
    print("=" * 84)
    print("CYMATIC PHI-NERF: 3D NYQUIST DIAGNOSTICS & HELD-OUT GENERALIZATION BENCHMARK")
    print("=" * 84)

    nyq = compute_3d_nyquist_frequency(num_samples=8192, box_side_length=2.0)
    print("\n[1] 3D Shannon-Nyquist & Diophantine Harmonic Analysis (N = 8,192 in [-1, 1]^3)")
    print(f"    Mean linear spacing Delta_x  : {nyq['mean_spacing_delta_x']:.6f} units")
    print(f"    3D Nyquist cutoff omega_Nyq  : {nyq['nyquist_omega_over_pi']:.4f} * pi rad/unit")
    for name, b in [
        ("Standard Dyadic (b = 2.0000)", 2.0),
        ("Golden Ratio    (b = 1.6180)", PHI),
        ("Nyquist-Matched (b = 1.3911)", nyq["nyquist_omega_over_pi"] ** (1.0 / 7.0)),
    ]:
        bands = [b**k for k in range(8)]
        below = sum(1 for f in bands if f <= nyq["nyquist_omega_over_pi"] + 1e-6)
        coll = diophantine_phase_collision_metric(b, num_freqs=8, max_harmonic=8)
        print(
            f"    {name} | max band = {bands[-1]:6.2f}*pi | "
            f"bands <= omega_Nyq: {below}/8 | Collision index C(b) = {coll:6.2f}"
        )

    print("\n[2] Multi-Seed Held-Out Generalization Ablation (300 Steps, N_train=8192, N_test=4096)")
    b_nyq = nyq["nyquist_omega_over_pi"] ** (1.0 / 7.0)
    configs = [
        ("Standard Dyadic (2^k)", lambda: StandardNeRFEncoding(num_freqs=8)),
        ("Pure Golden Ratio (Phi^k)", lambda: CymaticPhiEncoding(num_freqs=8, m_harmonics=0)),
        ("Cymatic Phi (Phi^k + sin m*theta)", lambda: CymaticPhiEncoding(num_freqs=8, m_harmonics=4)),
        ("Nyquist-Matched (1.39^k, max=10pi)", lambda: GeneralGeometricEncoding(num_freqs=8, base=b_nyq)),
    ]

    for scene_name, scene_fn in [
        ("Scene 1: Spherical Shell (Phi-Spiral)", scene_phi_shell),
        ("Scene 2: Neutral 3D Schoen Gyroid (Zero Phi in Ground Truth)", scene_gyroid_neutral),
    ]:
        print(f"\n  --- {scene_name} (Seeds: 42, 43, 44) ---")
        for cfg_name, enc_fn in configs:
            tr_psnrs, te_psnrs, te_losses = [], [], []
            for seed in [42, 43, 44]:
                torch.manual_seed(seed)
                pts_tr = (torch.rand(8192, 3) * 2.0) - 1.0
                pts_te = (torch.rand(4096, 3) * 2.0) - 1.0
                rgb_tr, sig_tr = scene_fn(pts_tr)
                rgb_te, sig_te = scene_fn(pts_te)

                torch.manual_seed(seed + 100)
                res = train_and_evaluate_nerf(
                    enc_fn(), pts_tr, rgb_tr, sig_tr, pts_te, rgb_te, sig_te, steps=300
                )
                tr_psnrs.append(res["train_psnr_db"])
                te_psnrs.append(res["test_psnr_db"])
                te_losses.append(res["test_loss"])

            print(
                f"    {cfg_name:36s} | Train PSNR: {np.mean(tr_psnrs):5.2f}+/-{np.std(tr_psnrs):4.2f} dB | "
                f"Test PSNR: {np.mean(te_psnrs):5.2f}+/-{np.std(te_psnrs):4.2f} dB | "
                f"Test Loss: {np.mean(te_losses):.5f}"
            )


if __name__ == "__main__":
    main()
