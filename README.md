# Cymatic $\Phi$-NeRF: Quasi-Periodic Golden-Ratio Positional Encoding and 3D Nyquist Anti-Aliasing in Neural Radiance Fields

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)
[![Release v1.0.0](https://img.shields.io/badge/Release-v1.0.0-orange.svg)](https://github.com/nauheimervater/cymatic-phi-nerf/releases/tag/v1.0.0)

**Author:** Thomas Nauheimer, Independent Researcher (`nauheimer.t@gmail.com`)  
**Paper:** [`paper/Cymatic_Phi_NeRF.pdf`](paper/Cymatic_Phi_NeRF.pdf) · [`paper/Cymatic_Phi_NeRF.md`](paper/Cymatic_Phi_NeRF.md) · [`paper/Cymatic_Phi_NeRF.tex`](paper/Cymatic_Phi_NeRF.tex)

---

## Overview

Coordinate-based Multi-Layer Perceptrons (MLPs) in Neural Radiance Fields (NeRF) rely on sinusoidal positional encodings $\gamma_b(\mathbf{p}) = [\mathbf{p}, \sin(b^k \pi \mathbf{p}), \cos(b^k \pi \mathbf{p})]_{k=0}^{L-1}$ to overcome spectral bias. While standard NeRF uses dyadic octave scaling ($b = 2$), finite volumetric sample budgets $N$ in $[-1, 1]^3$ impose a strict **3D Shannon–Nyquist angular cutoff**:

$$\omega_{\text{Nyq}} = \frac{\pi}{\Delta x} = \frac{\pi N^{1/3}}{2R} \approx 10.08\,\pi\text{ rad/unit} \qquad (N = 8,192,\; R = 1)$$

This repository proves and benchmarks two fundamental properties of positional encoding bases:
1. **Sub-Nyquist Band Packing (Proposition 1):** Standard dyadic encoding ($2^k \pi$, $L = 8$) places **4 of its 8 bands** ($16\pi, 32\pi, 64\pi, 128\pi$) strictly above $\omega_{\text{Nyq}}$, causing severe high-frequency overfitting between sampled points. Replacing $b = 2$ with the **Golden Ratio** $\Phi = \frac{1+\sqrt{5}}{2} \approx 1.618034$ ($\log_2 \Phi \approx 0.6942\text{ octaves/band}$) packs **6 of 8 bands** inside/at the Nyquist limit ($\le 11.09\pi$), while the exact **Nyquist-Matched base** $b_{\text{Nyq}} = \omega_{\text{Nyq}}^{1/(L-1)} \approx 1.3911$ packs **all 8 of 8 bands** inside the unaliased passband.
2. **Diophantine Incommensurability (Proposition 2):** Because $\Phi = [1; 1, 1, \dots]$ is the Hurwitz most-irrational number, Golden-Ratio progressions reduce low-order integer harmonic phase collisions $C(b)$ by **$45\times$** compared to dyadic octaves ($C(\Phi) = 14.23$ vs. $C(2) = 642.86$).

---

## Key Benchmark Results (3-Seed Held-Out Generalization, $N_{\text{train}}=8,192$, $N_{\text{test}}=4,096$)

| Scene & Encoding Variant | Train PSNR ($\text{dB}$) | Held-Out Test PSNR ($\text{dB}$) | Held-Out Test Loss | Gain vs. Dyadic (Test) |
| :--- | :---: | :---: | :---: | :---: |
| **Scene 1: Spherical Shell ($\Phi$-Spiral)** | | | | |
| Standard Dyadic ($2^k$) | $19.57 \pm 0.72$ | $16.68 \pm 0.74$ | $0.04847$ | Baseline |
| Pure Golden Ratio ($\Phi^k$, $m=0$) | $20.96 \pm 0.67$ | **$18.71 \pm 0.28$** | $0.03034$ | **$+2.03\text{ dB}$** |
| Cymatic $\Phi$-NeRF ($\Phi^k$, $m=4$) | $20.86 \pm 1.04$ | **$18.52 \pm 0.38$** | $0.03087$ | **$+1.84\text{ dB}$** |
| Nyquist-Matched ($1.391^k$) | $22.74 \pm 0.86$ | **$20.76 \pm 0.50$** | $0.01596$ | **$+4.08\text{ dB}$** |
| **Scene 2: Neutral 3D Schoen Gyroid (Zero $\Phi$)** | | | | |
| Standard Dyadic ($2^k$) | $12.49 \pm 0.60$ | $11.99 \pm 0.06$ | $0.47195$ | Baseline |
| Pure Golden Ratio ($\Phi^k$, $m=0$) | $13.61 \pm 1.54$ | **$13.03 \pm 0.95$** | $0.44478$ | **$+1.04\text{ dB}$** |
| Cymatic $\Phi$-NeRF ($\Phi^k$, $m=4$) | $13.45 \pm 1.35$ | **$12.99 \pm 0.92$** | $0.44498$ | **$+1.00\text{ dB}$** |
| Nyquist-Matched ($1.391^k$) | $15.24 \pm 2.65$ | **$14.74 \pm 2.24$** | $0.35356$ | **$+2.75\text{ dB}$** |

---

## Repository Structure

```text
cymatic-phi-nerf/
├── cymatic_nerf/
│   ├── __init__.py
│   └── encoding.py            # StandardNeRFEncoding, CymaticPhiEncoding, 3D Nyquist & Diophantine metrics
├── examples/
│   └── reproduce_all.py       # Reproduces Table 1 & Table 2 across Scene 1 and Scene 2
├── tests/
│   └── test_encoding.py       # Unit tests for Nyquist cutoff, collision metric, and NeRF training
├── paper/
│   ├── Cymatic_Phi_NeRF.pdf   # Publication-ready PDF paper
│   ├── Cymatic_Phi_NeRF.md    # Full Markdown manuscript
│   ├── Cymatic_Phi_NeRF.tex   # LaTeX source manuscript
│   └── generate_paper_pdf.py  # ReportLab PDF builder
├── CITATION.cff
├── .zenodo.json
├── LICENSE
└── requirements.txt
```

## Quickstart & Verification

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
python examples/reproduce_all.py
```

## Citation

If you use this work in your research, please cite:

```bibtex
@article{nauheimer2026cymaticnerf,
  author  = {Nauheimer, Thomas},
  title   = {Cymatic Phi-NeRF: Quasi-Periodic Golden-Ratio Positional Encoding and 3D Nyquist Anti-Aliasing in Neural Radiance Fields},
  year    = {2026},
  month   = {11},
  url     = {https://github.com/nauheimervater/cymatic-phi-nerf}
}
```
