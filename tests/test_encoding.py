"""
Unit tests for cymatic_nerf: 3D Nyquist frequency diagnostics, Diophantine
phase-collision metric, positional encodings, and volumetric NeRF generalization.
"""

import math
import unittest
import torch

from cymatic_nerf import (
    PHI,
    StandardNeRFEncoding,
    CymaticPhiEncoding,
    GeneralGeometricEncoding,
    NeRFModel,
    scene_phi_shell,
    scene_gyroid_neutral,
    compute_3d_nyquist_frequency,
    diophantine_phase_collision_metric,
    train_and_evaluate_nerf,
)


class TestCymaticPhiNeRF(unittest.TestCase):
    def test_3d_nyquist_frequency(self) -> None:
        info = compute_3d_nyquist_frequency(num_samples=8192, box_side_length=2.0)
        self.assertAlmostEqual(info["samples_per_axis"], 8192.0 ** (1.0 / 3.0), places=5)
        self.assertAlmostEqual(info["mean_spacing_delta_x"], 0.0992125657, places=5)
        self.assertAlmostEqual(info["nyquist_omega_over_pi"], 10.079368399, places=5)
        # Dyadic (2^k) has 4 bands <= 10.08*pi (k=0,1,2,3 -> 1,2,4,8) and 4 above (16,32,64,128)
        dyadic_below = sum(1 for k in range(8) if (2.0**k) <= info["nyquist_omega_over_pi"])
        phi_below = sum(1 for k in range(8) if (PHI**k) <= info["nyquist_omega_over_pi"])
        phi_near = sum(1 for k in range(8) if (PHI**k) <= 1.101 * info["nyquist_omega_over_pi"])
        self.assertEqual(dyadic_below, 4)
        self.assertEqual(phi_below, 5)
        self.assertEqual(phi_near, 6)

    def test_diophantine_phase_collision_metric(self) -> None:
        # Dyadic b=2 has exact integer powers 2^(k-j), maximizing integer harmonic collisions
        c_dyadic = diophantine_phase_collision_metric(2.0, num_freqs=8, max_harmonic=8)
        c_phi = diophantine_phase_collision_metric(PHI, num_freqs=8, max_harmonic=8)
        self.assertAlmostEqual(c_dyadic, 642.88, places=1)
        self.assertAlmostEqual(c_phi, 2.98, places=1)
        c_nyq = diophantine_phase_collision_metric(10.079368399 ** (1.0 / 7.0), num_freqs=8, max_harmonic=8)
        self.assertAlmostEqual(c_nyq, 5.84, places=1)

    def test_encoding_shapes_and_finiteness(self) -> None:
        pts = torch.randn(64, 3)
        enc_std = StandardNeRFEncoding(in_dims=3, num_freqs=8)
        enc_cym = CymaticPhiEncoding(in_dims=3, num_freqs=8, m_harmonics=4)
        enc_pure = CymaticPhiEncoding(in_dims=3, num_freqs=8, m_harmonics=0)
        enc_gen = GeneralGeometricEncoding(in_dims=3, num_freqs=8, base=1.391)

        for enc in (enc_std, enc_cym, enc_pure, enc_gen):
            out = enc(pts)
            self.assertEqual(out.shape, (64, 51))
            self.assertTrue(torch.isfinite(out).all())

    def test_scenes_and_short_training(self) -> None:
        torch.manual_seed(42)
        pts_tr = (torch.rand(256, 3) * 2.0) - 1.0
        pts_te = (torch.rand(128, 3) * 2.0) - 1.0

        for scene_fn in (scene_phi_shell, scene_gyroid_neutral):
            rgb_tr, sig_tr = scene_fn(pts_tr)
            rgb_te, sig_te = scene_fn(pts_te)
            self.assertEqual(rgb_tr.shape, (256, 3))
            self.assertEqual(sig_tr.shape, (256, 1))

            res = train_and_evaluate_nerf(
                CymaticPhiEncoding(num_freqs=8, m_harmonics=4),
                pts_tr,
                rgb_tr,
                sig_tr,
                pts_te,
                rgb_te,
                sig_te,
                steps=25,
                hidden_dim=64,
            )
            self.assertGreater(res["train_psnr_db"], 8.0)
            self.assertGreater(res["test_psnr_db"], 8.0)


if __name__ == "__main__":
    unittest.main()
