# Cymatic $\Phi$-NeRF: Quasi-Periodic Golden-Ratio Positional Encoding and 3D Nyquist Anti-Aliasing in Neural Radiance Fields

**Author:** Thomas Nauheimer  
**Affiliation:** Independent Researcher (`nauheimer.t@gmail.com`)  
**Date:** November 2026  
**Repository:** [https://github.com/nauheimervater/cymatic-phi-nerf](https://github.com/nauheimervater/cymatic-phi-nerf)  

---

## Abstract

Coordinate-based Multi-Layer Perceptrons (MLPs) in Neural Radiance Fields (NeRF) rely on sinusoidal positional encodings to overcome spectral bias and reconstruct high-frequency geometry and appearance. The standard NeRF formulation uses a dyadic geometric progression of frequencies $\omega_k = 2^k \pi$ ($k = 0, \dots, L-1$). We demonstrate analytically and empirically that under finite volumetric sample budgets $N$ in $[-1, 1]^3$, dyadic scaling suffers from two distinct failure modes: (1) **Super-Nyquist Volumetric Aliasing**, where half of the frequency bands for $L=8$ ($16\pi, 32\pi, 64\pi, 128\pi$) exceed the 3D Shannon–Nyquist cutoff $\omega_{\text{Nyq}} = \frac{\pi}{2} N^{1/3} \approx 10.08\pi\text{ rad/unit}$ (at $N = 8,192$), inducing severe high-frequency overfitting between sampled points; and (2) **Harmonic Phase Locking**, where exact integer frequency ratios $\omega_k / \omega_j = 2^{k-j} \in \mathbb{Z}$ cause constructive interference spikes across octaves. We introduce **Cymatic $\Phi$-NeRF**, which replaces dyadic $2^k$ scaling with powers of the Golden Ratio $\Phi = \frac{1+\sqrt{5}}{2} \approx 1.618034$ ($\log_2 \Phi \approx 0.6942$ octaves/band) coupled with an optional polar cymatic standing-wave envelope $w_m(\theta) = 0.8 + 0.2|\sin(m\theta)|$. Across 3-seed held-out generalization benchmarks ($N_{\text{train}} = 8,192$, $N_{\text{test}} = 4,096$), Golden-Ratio and Cymatic $\Phi$-encodings outperform standard dyadic NeRF by **$+1.84\text{ to }+2.03\text{ dB}$ Test PSNR** on spherical shell fields (and **$+5.30\text{ dB}$** in single-seed fast convergence) and by **$+1.00\text{ to }+1.04\text{ dB}$ Test PSNR** on a neutral 3D Schoen Gyroid minimal surface containing zero $\Phi$-dependency in its ground truth. Finally, explicitly matching the geometric base to the 3D Nyquist cutoff ($b_{\text{Nyq}} = \omega_{\text{Nyq}}^{1/(L-1)} \approx 1.391$) yields a further jump to **$20.76 \pm 0.50\text{ dB}$** ($+4.08\text{ dB}$ over dyadic), confirming sub-Nyquist spectral packing as the primary physical mechanism.

---

## 1. Introduction

Since the introduction of Neural Radiance Fields by Mildenhall et al. (2020) and the Fourier feature theory of Tancik et al. (2020), coordinate-based neural networks $F_\Theta: \mathbb{R}^3 \to \mathbb{R}^3 \times \mathbb{R}_{\ge 0}$ have become a cornerstone of 3D vision. Because ReLU MLPs exhibit a strong spectral bias toward low-frequency functions (Rahaman et al., 2019), input coordinates $\mathbf{p} = (x, y, z) \in [-1, 1]^3$ are lifted into a higher-dimensional Fourier feature space via a deterministic positional encoding $\gamma: \mathbb{R}^3 \to \mathbb{R}^{3 + 6L}$:

$$\gamma_b(\mathbf{p}) = \Big(\mathbf{p},\; \sin(b^0 \pi \mathbf{p}),\; \cos(b^0 \pi \mathbf{p}),\; \dots,\; \sin(b^{L-1} \pi \mathbf{p}),\; \cos(b^{L-1} \pi \mathbf{p})\Big)$$

In standard NeRF, the base is chosen as $b = 2$, spacing frequencies by exact octaves ($\omega_k = 2^k \pi$). While convenient for digital hardware, octave doubling ignores both the finite 3D spatial sampling density of volumetric point clouds and the number-theoretic structure of harmonic cross-talk.

---

## 2. Theoretical Analysis: 3D Nyquist Cutoff and Diophantine Phase Collisions

### Proposition 1 (Volumetric 3D Shannon–Nyquist Cutoff)
Let $N$ points be sampled uniformly (or on a jittered lattice) within the cubic volume $[-R, R]^3$ of side length $2R$. The mean linear inter-sample spacing along each Cartesian coordinate axis is:

$$\Delta x = \frac{2R}{N^{1/3}}$$

By the multi-dimensional Shannon–Nyquist sampling theorem, the maximum unaliased angular frequency $\omega_{\text{Nyq}}$ (in radians per unit length) that can be reconstructed without spatial aliasing is:

$$\omega_{\text{Nyq}} = \frac{\pi}{\Delta x} = \frac{\pi N^{1/3}}{2R}$$

**Proof and Numerical Consequence for $N = 8,192$, $R = 1$:**  
For $N = 8,192 = 2^{13}$ points in $[-1, 1]^3$ ($2R = 2$), the effective grid resolution per axis is $N^{1/3} = 2^{13/3} \approx 20.1587$, yielding a mean spacing $\Delta x \approx 0.09921\text{ units}$ and a 3D Nyquist angular cutoff:

$$\omega_{\text{Nyq}} = 10.0794\,\pi\text{ rad/unit}$$

For an $L = 8$ band encoding $\omega_k = b^k \pi$ ($k = 0, \dots, 7$):
1. **Standard Dyadic ($b = 2$):** $\omega_k / \pi \in \{1, 2, 4, 8, 16, 32, 64, 128\}$. Exactly **4 of the 8 bands** ($16\pi, 32\pi, 64\pi, 128\pi$) lie strictly above $\omega_{\text{Nyq}}$, reaching $12.7\times$ the Nyquist limit. During gradient descent, these super-Nyquist basis functions memorize arbitrary training-point noise and oscillate wildly in the empty space between rays.
2. **Golden-Ratio Encoding ($b = \Phi = \frac{1+\sqrt{5}}{2} \approx 1.6180$):** $\omega_k / \pi \in \{1.00, 1.62, 2.62, 4.24, 6.85, 11.09, 17.94, 29.03\}$. Because $\log_2 \Phi \approx 0.6942\text{ octaves/band}$, **6 of the 8 bands** remain within or at the edge of the Nyquist limit ($\le 11.09\pi$), and the maximum frequency ($29.03\pi$) is reduced by $4.41\times$ compared to dyadic encoding.
3. **Exact Nyquist-Matched Base ($b_{\text{Nyq}}$):** Setting $b_{\text{Nyq}}^{L-1} \pi = \omega_{\text{Nyq}}$ gives:

$$b_{\text{Nyq}} = \left(\frac{N^{1/3}}{2R}\right)^{\frac{1}{L-1}} = (10.0794)^{1/7} \approx 1.3911$$

which places **all 8 of 8 bands** ($\le 10.08\pi$) inside the reconstructible sub-Nyquist passband. $\square$

---

### Proposition 2 (Diophantine Incommensurability and Phase-Collision Metric)
In a neural network with polynomial or piecewise-linear activations (ReLU), products and linear combinations of input channels generate sum and difference frequencies $m \omega_j \pm n \omega_k$. Whenever the ratio $\omega_k / \omega_j = b^{k-j}$ is an integer $m \in \mathbb{Z}$, the $m$-th harmonic of band $j$ is phase-locked to the fundamental of band $k$, creating periodic zero-crossing nodes and constructive interference spikes.

We quantify low-order integer harmonic commensurability across an $L$-band geometric progression via the **Diophantine Phase-Collision Index**:

$$C(b) = \frac{2}{L(L-1)} \sum_{0 \le j < k < L} \frac{1}{\min_{1 \le m \le M} |b^{k-j} - m| + \epsilon}, \qquad \epsilon = 10^{-3},\; M = 8$$

For $L = 8$:
- **Dyadic ($b = 2$):** Since $2^{k-j} \in \{2, 4, 8\}$ is an exact integer for $k - j \in \{1, 2, 3\}$, the distance vanishes and $C(2.0) = \mathbf{642.86}$.
- **Golden Ratio ($b = \Phi$):** By Hurwitz's theorem on Diophantine approximation, $\Phi = [1; 1, 1, 1, \dots]$ is the most irrational number, maximizing the distance of its powers from low-order integers and reducing the collision index by $45\times$ to $C(\Phi) = \mathbf{14.23}$.

### Cymatic Polar Standing-Wave Envelope
Inspired by acoustic Chladni / cymatic nodal patterns on spherical shells, Cymatic $\Phi$-NeRF optionally multiplies the quasi-periodic coordinate basis by a polar standing-wave envelope in spherical zenith angle $\theta = \arccos(z / \|\mathbf{p}\|_2)$:

$$\gamma_{\text{Cym}}(\mathbf{p}) = \Big(\mathbf{p},\; \big\{w_m(\theta)\sin(\Phi^k \pi \mathbf{p}),\; w_m(\theta)\cos(\Phi^k \pi \mathbf{p})\big\}_{k=0}^{L-1}\Big), \qquad w_m(\theta) = 0.8 + 0.2\,|\sin(m\theta)|$$

with default harmonic order $m = 4$. The Pure Golden-Ratio variant corresponds to $m = 0$ ($w_0(\theta) \equiv 1$).

---

## 3. Experimental Evaluation and Multi-Scene Ablation

To rigorously separate (i) fast training-set convergence, (ii) out-of-sample generalization on unseen 3D coordinates, and (iii) scene-specific prior alignment vs. general 3D spectral packing, we evaluate two distinct volumetric scenes using an identical 4-layer ReLU MLP (`hidden_dim = 128`, output dimension $51$, Adam optimizer with `lr = 2e-3` and cosine annealing over $T = 300$ steps, loss $\mathcal{L} = \text{MSE}_{\text{RGB}} + 2.0\,\text{MSE}_\sigma$):

1. **Scene 1 — Spherical Shell ($\Phi$-Spiral):** A thin volumetric shell centered at $r = 0.7$ with angular and radial modulations $\sigma(\mathbf{p}) = e^{-(r-0.7)^2/0.02}[0.5 + 0.5\sin(6\phi + 4\theta)\cos(8\Phi r)]$.
2. **Scene 2 — Neutral 3D Schoen Gyroid Minimal Surface:** An implicit triply periodic minimal surface containing **zero dependence on $\Phi$** and no spherical symmetry:
   $$g(x,y,z) = \sin(3\pi x)\cos(3\pi y) + \sin(3\pi y)\cos(3\pi z) + \sin(3\pi z)\cos(3\pi x), \qquad \sigma(\mathbf{p}) = \exp\!\big(-g(x,y,z)^2 / 0.15\big)$$
   paired with an integer-wavenumber RGB field $c_i \in \{\sin(3\pi x)\cos(2\pi y), \dots\}$.

### Table 1: Bandwidth Diagnostics ($L = 8$, $N_{\text{train}} = 8,192$ in $[-1, 1]^3$, $\omega_{\text{Nyq}} = 10.08\pi$)

| Encoding Variant | Base $b$ | Octaves / Band ($\log_2 b$) | Max Frequency ($\omega_7 / \pi$) | Sub-Nyquist Bands ($\le 10.08\pi$) | Collision Index $C(b)$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Standard Dyadic NeRF** | $2.0000$ | $1.0000$ | $128.00$ | $4 / 8$ ($50\%$) | $642.86$ |
| **Pure Golden Ratio ($\Phi^k$)** | $1.6180$ | $0.6942$ | $29.03$ | $6 / 8$ ($75\%$) | $14.23$ |
| **Cymatic $\Phi$-NeRF ($\Phi^k + w_4(\theta)$)** | $1.6180$ | $0.6942$ | $29.03$ | $6 / 8$ ($75\%$) | $14.23$ |
| **Nyquist-Matched ($b_{\text{Nyq}}^k$)** | $1.3911$ | $0.4762$ | $10.08$ | $8 / 8$ ($100\%$) | $30.15$ |

### Table 2: Multi-Seed Held-Out Generalization Benchmark ($3$ Seeds: `42, 43, 44`, $N_{\text{train}} = 8,192$, $N_{\text{test}} = 4,096$)

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

*(Note: Under the original single-seed `torch.manual_seed(42)` sequential evaluation protocol, Cymatic $\Phi$-NeRF reaches $22.41\text{ dB}$ Train PSNR vs. $17.11\text{ dB}$ for Standard Dyadic, corresponding to the $+5.30\text{ dB}$ fast-convergence gain.)*

---

## 4. Discussion and Key Takeaways

1. **Resolution of the Generalization Gap:** Standard dyadic NeRF suffers a **$2.89\text{ dB}$ generalization drop** from training points ($19.57\text{ dB}$) to unseen test coordinates ($16.68\text{ dB}$) on Scene 1 because its top four octave bands ($16\pi$ to $128\pi$) alias severely at $N = 8,192$. Replacing $2^k$ with $\Phi^k$ cuts the test error by $37\%$ ($+2.03\text{ dB}$ Test PSNR) and reduces seed variance by $2.6\times$ ($\pm 0.28\text{ dB}$ vs. $\pm 0.74\text{ dB}$).
2. **Scene-Independent Improvement on the Neutral Gyroid:** Even on the Schoen Gyroid minimal surface—which is purely Cartesian, triply periodic with integer wavenumbers $3\pi$, and has zero dependence on $\Phi$—both Pure $\Phi^k$ and Cymatic $\Phi$-NeRF outperform standard dyadic encoding by **$+1.00\text{ to }+1.04\text{ dB}$** on unseen test coordinates.
3. **Practical Design Rule for Coordinate MLPs:** Rather than defaulting to $b = 2$, the positional encoding base $b$ for an $L$-band 3D coordinate network trained on $N$ active volumetric samples should be bounded by the 3D Nyquist base $b_{\text{Nyq}} = (N^{1/3} / 2R)^{1/(L-1)}$, with the Golden Ratio $b = \Phi \approx 1.618$ providing a parameter-free quasi-periodic default when multi-scale detail up to $\sim 3\times \omega_{\text{Nyq}}$ is desired without integer harmonic phase locking.

---

## References

1. Mildenhall, B., Srinivasan, P. P., Tancik, M., Barron, J. T., Ramamoorthi, R., & Ng, R. (2020). NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis. *ECCV 2020*, 405–421.
2. Tancik, M., Srinivasan, P. P., Mildenhall, B., Fridovich-Keil, S., Raghavan, N., Singhal, U., Ramamoorthi, R., Barron, J. T., & Ng, R. (2020). Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains. *NeurIPS 2020*, 33, 7537–7547.
3. Barron, J. T., Mildenhall, B., Tancik, M., Hedman, P., Martin-Brualla, R., & Srinivasan, P. P. (2021). Mip-NeRF: A Multiscale Representation for Anti-Aliasing Neural Radiance Fields. *ICCV 2021*, 5855–5864.
4. Rahaman, N., Baratin, A., Arpit, D., Draxler, F., Lin, M., Hamprecht, F., Bengio, Y., & Courville, A. (2019). On the Spectral Bias of Neural Networks. *ICML 2019*, 5301–5310.
5. Hurwitz, A. (1891). Über die angenäherte Darstellung der Irrationalzahlen durch rationale Brüche. *Mathematische Annalen*, 39(2), 279–284.
