"""
cymatic_nerf: Quasi-Periodic Golden-Ratio (Phi^k) and Cymatic Standing-Wave
Positional Encodings for 3D Neural Radiance Fields (NeRF) and Implicit Volumetric
Representations.
"""

from .encoding import (
    PHI,
    StandardNeRFEncoding,
    CymaticPhiEncoding,
    GeneralGeometricEncoding,
    NeRFModel,
    compute_3d_nyquist_frequency,
    diophantine_phase_collision_metric,
    scene_phi_shell,
    scene_gyroid_neutral,
    train_and_evaluate_nerf,
)

__all__ = [
    "PHI",
    "StandardNeRFEncoding",
    "CymaticPhiEncoding",
    "GeneralGeometricEncoding",
    "NeRFModel",
    "compute_3d_nyquist_frequency",
    "diophantine_phase_collision_metric",
    "scene_phi_shell",
    "scene_gyroid_neutral",
    "train_and_evaluate_nerf",
]
