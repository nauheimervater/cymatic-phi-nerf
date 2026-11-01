"""
Generate the publication-grade PDF for:
  "Cymatic Phi-NeRF: Quasi-Periodic Golden-Ratio Positional Encoding and 3D Nyquist
   Anti-Aliasing in Neural Radiance Fields"
  Author: Thomas Nauheimer (2026)
"""

from __future__ import annotations

import os
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def build_pdf(output_path: str) -> None:
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=2.2 * cm,
        rightMargin=2.2 * cm,
        topMargin=2.2 * cm,
        bottomMargin=2.2 * cm,
        title="Cymatic Phi-NeRF: Quasi-Periodic Golden-Ratio Positional Encoding and 3D Nyquist Anti-Aliasing in Neural Radiance Fields",
        author="Thomas Nauheimer",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "PaperTitle",
        parent=styles["Heading1"],
        fontName="Times-Bold",
        fontSize=15,
        leading=19,
        alignment=TA_CENTER,
        spaceAfter=8,
    )
    author_style = ParagraphStyle(
        "PaperAuthor",
        parent=styles["Normal"],
        fontName="Times-Roman",
        fontSize=10.5,
        leading=13,
        alignment=TA_CENTER,
        spaceAfter=14,
    )
    abstract_style = ParagraphStyle(
        "PaperAbstract",
        parent=styles["Normal"],
        fontName="Times-Italic",
        fontSize=9.5,
        leading=12.5,
        alignment=TA_JUSTIFY,
        leftIndent=18,
        rightIndent=18,
        spaceAfter=12,
    )
    h1_style = ParagraphStyle(
        "SectionH1",
        parent=styles["Heading2"],
        fontName="Times-Bold",
        fontSize=12,
        leading=15,
        spaceBefore=10,
        spaceAfter=5,
    )
    body_style = ParagraphStyle(
        "PaperBody",
        parent=styles["Normal"],
        fontName="Times-Roman",
        fontSize=10,
        leading=13.5,
        alignment=TA_JUSTIFY,
        spaceAfter=7,
    )
    eq_style = ParagraphStyle(
        "PaperEq",
        parent=styles["Normal"],
        fontName="Times-Italic",
        fontSize=10,
        leading=14,
        alignment=TA_CENTER,
        spaceBefore=4,
        spaceAfter=6,
    )
    caption_style = ParagraphStyle(
        "TableCaption",
        parent=styles["Normal"],
        fontName="Times-Bold",
        fontSize=9.5,
        leading=12,
        alignment=TA_LEFT,
        spaceBefore=6,
        spaceAfter=4,
    )

    story = []
    story.append(
        Paragraph(
            "Cymatic &Phi;-NeRF: Quasi-Periodic Golden-Ratio Positional Encoding and 3D Nyquist Anti-Aliasing in Neural Radiance Fields",
            title_style,
        )
    )
    story.append(
        Paragraph(
            "<b>Thomas Nauheimer</b><br/>Independent Researcher (nauheimer.t@gmail.com)<br/>November 2026 (v1.0.1) &middot; Repository: https://github.com/nauheimervater/cymatic-phi-nerf",
            author_style,
        )
    )

    story.append(
        Paragraph(
            "<b>Abstract.</b> Coordinate-based Multi-Layer Perceptrons (MLPs) in Neural Radiance Fields (NeRF) rely on sinusoidal positional encodings to overcome spectral bias. The standard NeRF formulation uses a dyadic geometric progression &omega;<sub>k</sub> = 2<sup>k</sup>&pi; (k = 0, ..., L&minus;1). We demonstrate analytically and empirically that under finite volumetric sample budgets N in [&minus;1, 1]<sup>3</sup>, dyadic scaling suffers from two distinct failure modes: (1) <b>Super-Nyquist Volumetric Aliasing</b>, where half of the frequency bands for L=8 (16&pi;, 32&pi;, 64&pi;, 128&pi;) exceed the 3D Shannon&ndash;Nyquist cutoff &omega;<sub>Nyq</sub> = (&pi;/2) N<sup>1/3</sup> &approx; 10.08&pi; rad/unit (at N = 8,192), inducing severe high-frequency overfitting between sampled points; and (2) <b>Harmonic Phase Locking</b>, where exact integer ratios &omega;<sub>k</sub>/&omega;<sub>j</sub> = 2<sup>k&minus;j</sup> &isin; Z cause constructive interference spikes across octaves. We introduce <b>Cymatic &Phi;-NeRF</b>, replacing dyadic 2<sup>k</sup> scaling with powers of the Golden Ratio &Phi; = (1+&radic;5)/2 &approx; 1.618034 (log<sub>2</sub>&Phi; &approx; 0.6942 octaves/band) coupled with an optional polar cymatic standing-wave envelope w<sub>m</sub>(&theta;) = 0.8 + 0.2|sin(m&theta;)|. Across 3-seed held-out generalization benchmarks (N<sub>train</sub> = 8,192, N<sub>test</sub> = 4,096), Golden-Ratio and Cymatic &Phi;-encodings outperform standard dyadic NeRF by <b>+1.84 to +2.03 dB Test PSNR</b> on spherical shell fields (and <b>+5.30 dB</b> in single-seed fast convergence) and by <b>+1.00 to +1.04 dB Test PSNR</b> on a neutral 3D Schoen Gyroid minimal surface containing zero &Phi;-dependency in its ground truth. Explicitly matching the geometric base to the 3D Nyquist cutoff (b<sub>Nyq</sub> = &omega;<sub>Nyq</sub><sup>1/(L&minus;1)</sup> &approx; 1.391) yields <b>20.76 &plusmn; 0.50 dB</b> (+4.08 dB over dyadic), confirming sub-Nyquist spectral packing as the primary physical mechanism.",
            abstract_style,
        )
    )

    story.append(Paragraph("1. Introduction & Volumetric Nyquist Analysis", h1_style))
    story.append(
        Paragraph(
            "Coordinate MLPs map 3D spatial points p = (x, y, z) &isin; [&minus;1, 1]<sup>3</sup> into Fourier features via &gamma;<sub>b</sub>(p) = [p, sin(b<sup>0</sup>&pi;p), cos(b<sup>0</sup>&pi;p), ..., sin(b<sup>L&minus;1</sup>&pi;p), cos(b<sup>L&minus;1</sup>&pi;p)]. For N uniform volumetric points in [&minus;R, R]<sup>3</sup>, the mean linear spacing per coordinate axis is &Delta;x = 2R / N<sup>1/3</sup>, yielding the 3D Shannon&ndash;Nyquist angular cutoff:",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "&omega;<sub>Nyq</sub> = &pi; / &Delta;x = (&pi; N<sup>1/3</sup>) / (2R) &approx; 10.0794 &pi; rad/unit   (for N = 8,192, R = 1).",
            eq_style,
        )
    )
    story.append(
        Paragraph(
            "<b>Proposition 1 (Sub-Nyquist Band Packing).</b> Under L = 8 bands, standard dyadic scaling (b = 2) places 4 of 8 bands (16&pi;, 32&pi;, 64&pi;, 128&pi;) strictly above &omega;<sub>Nyq</sub>, reaching 12.7&times; the Nyquist limit and causing severe inter-ray aliasing. By contrast, Golden-Ratio scaling (b = &Phi; &approx; 1.6180) places 5 of 8 bands below the Nyquist limit (a sixth, 11.09&pi;, lies within 10% above it), while the Nyquist-matched base b<sub>Nyq</sub> = (10.0794)<sup>1/7</sup> &approx; 1.3911 places all 8 of 8 bands inside the unaliased passband.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "<b>Proposition 2 (Diophantine Phase-Collision Metric).</b> Measuring low-order integer harmonic commensurability via C(b) = mean<sub>0&le;j&lt;k&lt;L</sub> [min<sub>1&le;m&le;8</sub> |b<sup>k&minus;j</sup> &minus; m| + 10<sup>&minus;3</sup>]<sup>&minus;1</sup> shows that dyadic octaves produce C(2.0) = 642.88 due to exact integer powers 2<sup>k&minus;j</sup> &isin; Z, whereas &Phi; = [1; 1, 1, ...] gives C(&Phi;) = 2.98 and the Nyquist-matched base C(1.3911) = 5.84. Because the best-performing base has the higher collision index, the gains are attributed primarily to sub-Nyquist band placement. <i>(Erratum v1.0.1: v1.0.0 stated 14.23 / 30.15 and 6 of 8 sub-Nyquist &Phi; bands.)</i>",
            body_style,
        )
    )

    story.append(Paragraph("2. Multi-Scene Held-Out Generalization Benchmarks", h1_style))
    story.append(
        Paragraph(
            "Table 1 summarizes bandwidth diagnostics and Table 2 reports 3-seed held-out test generalization (seeds 42, 43, 44; N<sub>train</sub> = 8,192, N<sub>test</sub> = 4,096; 300 Adam steps, hidden_dim = 128) across Scene 1 (Spherical Shell) and Scene 2 (Neutral 3D Schoen Gyroid minimal surface sin(3&pi;x)cos(3&pi;y) + sin(3&pi;y)cos(3&pi;z) + sin(3&pi;z)cos(3&pi;x), which contains zero &Phi;-dependency).",
            body_style,
        )
    )

    story.append(Paragraph("Table 1: 3D Nyquist Bandwidth & Diophantine Diagnostics (L = 8, N = 8,192)", caption_style))
    t1_data = [
        ["Encoding Variant", "Base b", "log2(b)", "Max Band (rad)", "Sub-Nyquist Bands", "Collision C(b)"],
        ["Standard Dyadic (2^k)", "2.0000", "1.0000", "128.00 pi", "4 / 8 (50%)", "642.86"],
        ["Pure Golden Ratio (Phi^k)", "1.6180", "0.6942", "29.03 pi", "5 / 8 (62.5%)", "2.98"],
        ["Cymatic Phi (Phi^k + w_4)", "1.6180", "0.6942", "29.03 pi", "5 / 8 (62.5%)", "2.98"],
        ["Nyquist-Matched (1.391^k)", "1.3911", "0.4762", "10.08 pi", "8 / 8 (100%)", "5.84"],
    ]
    t1 = Table(t1_data, colWidths=[4.3 * cm, 1.8 * cm, 1.8 * cm, 2.5 * cm, 3.1 * cm, 2.6 * cm])
    t1.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Times-Roman"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.8),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9ca3af")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f3f4f6")]),
            ]
        )
    )
    story.append(t1)
    story.append(Spacer(1, 8))

    story.append(
        Paragraph(
            "Table 2: 3-Seed Held-Out Generalization Benchmark (N_train = 8,192, N_test = 4,096, 300 Steps)",
            caption_style,
        )
    )
    t2_data = [
        ["Scene & Encoding Variant", "Train PSNR (dB)", "Test PSNR (dB)", "Test Loss", "Gain vs. Dyadic"],
        ["Scene 1: Spherical Shell (Phi-Spiral)", "", "", "", ""],
        ["  Standard Dyadic (2^k)", "19.57 +/- 0.72", "16.68 +/- 0.74", "0.04847", "Baseline"],
        ["  Pure Golden Ratio (Phi^k, m=0)", "20.96 +/- 0.67", "18.71 +/- 0.28", "0.03034", "+2.03 dB"],
        ["  Cymatic Phi-NeRF (Phi^k, m=4)", "20.86 +/- 1.04", "18.52 +/- 0.38", "0.03087", "+1.84 dB"],
        ["  Nyquist-Matched (1.391^k)", "22.74 +/- 0.86", "20.76 +/- 0.50", "0.01596", "+4.08 dB"],
        ["Scene 2: Neutral 3D Schoen Gyroid", "", "", "", ""],
        ["  Standard Dyadic (2^k)", "12.49 +/- 0.60", "11.99 +/- 0.06", "0.47195", "Baseline"],
        ["  Pure Golden Ratio (Phi^k, m=0)", "13.61 +/- 1.54", "13.03 +/- 0.95", "0.44478", "+1.04 dB"],
        ["  Cymatic Phi-NeRF (Phi^k, m=4)", "13.45 +/- 1.35", "12.99 +/- 0.92", "0.44498", "+1.00 dB"],
        ["  Nyquist-Matched (1.391^k)", "15.24 +/- 2.65", "14.74 +/- 2.24", "0.35356", "+2.75 dB"],
    ]
    t2 = Table(t2_data, colWidths=[5.6 * cm, 3.0 * cm, 3.0 * cm, 2.2 * cm, 2.5 * cm])
    t2.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
                ("FONTNAME", (0, 1), (-1, 1), "Times-Bold"),
                ("FONTNAME", (0, 6), (-1, 6), "Times-Bold"),
                ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#e5e7eb")),
                ("BACKGROUND", (0, 6), (-1, 6), colors.HexColor("#e5e7eb")),
                ("FONTNAME", (0, 2), (-1, 5), "Times-Roman"),
                ("FONTNAME", (0, 7), (-1, 10), "Times-Roman"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.8),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9ca3af")),
            ]
        )
    )
    story.append(t2)
    story.append(Spacer(1, 8))

    story.append(Paragraph("3. Conclusion & References", h1_style))
    story.append(
        Paragraph(
            "Replacing dyadic 2<sup>k</sup> scaling with quasi-periodic Golden-Ratio (&Phi;<sup>k</sup>) or 3D Nyquist-matched geometric progressions substantially suppresses inter-ray overfitting on unseen 3D coordinates across both spherical and neutral Cartesian minimal-surface geometries.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "1. Mildenhall et al. (2020), <i>NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis</i>, ECCV 2020.<br/>"
            "2. Tancik et al. (2020), <i>Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains</i>, NeurIPS 2020.<br/>"
            "3. Barron et al. (2021), <i>Mip-NeRF: A Multiscale Representation for Anti-Aliasing Neural Radiance Fields</i>, ICCV 2021.",
            body_style,
        )
    )

    doc.build(story)


if __name__ == "__main__":
    out_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Cymatic_Phi_NeRF.pdf")
    build_pdf(out_file)
    print(f"Generated PDF: {out_file} ({os.path.getsize(out_file)} bytes)")
