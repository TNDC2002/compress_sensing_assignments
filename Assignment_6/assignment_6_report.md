# Assignment 6 Report: Sparse Representation Classification for Face Recognition

**COMP5340 — Spring 2026**

## Abstract

We implement Sparse Representation Classification (SRC) on the cropped Extended Yale Face Database B, extend it to handle sparse pixel/patch corruptions via robust SRC, and evaluate preprocessing plus outlier rejection on personal selfies. On a stratified 50/50 split, baseline SRC achieves **91.12%** accuracy. Robust SRC helps under patch occlusion; a residual-based gate rejects out-of-gallery selfies while accepting an in-database “friend” probe.

---

## Dataset and protocol

| Item | Value |
|------|-------|
| Database | `material/CroppedYale_96_84_2414_subset.mat` |
| Images | 2414 faces, size **96 × 84**, 38 subjects |
| Split | Stratified 50/50 per subject, `seed = 0` |
| Training | 1209 images (dictionary only) |
| Testing | 1205 images (unless noted) |
| Sparse coding | LASSO + ISTA, λ = 0.01 |

**Code:** `assignment_6_src_common.py`, `assignment_6_src_faces.py`, `assignment_6_issue2_robust_outliers.py`, `assignment_6_issue3_selfies.py`

---

## Issue 1: Baseline SRC

### Method

1. Vectorize each face to \(y \in \mathbb{R}^{8064}\); build dictionary \(A\) from training faces (column \(\ell_2\)-normalized).
2. Solve \(\min_\alpha \|\alpha\|_1 + \frac{\lambda}{2}\|y - A\alpha\|_2^2\) (exact \(A\alpha = y\) is infeasible when pixel dimension exceeds training count).
3. Predict \(\arg\min_k \|y - A_k \delta_k(\alpha)\|_2\) (Wright et al., PAMI 2009).

### Results

| Metric | Value |
|--------|------:|
| **Accuracy** | **91.12%** |
| Runtime | 578 s (~0.48 s / image) |

```bash
python Assignment_6/assignment_6_src_faces.py
```

---

## Issue 2: Robust SRC under sparse outliers

### Corruption

- **Random pixels:** fraction \(p\) of entries → uniform noise in \([0,255]\).
- **Random patches:** \(24\times24\) blocks until \(\approx p\) of pixels are corrupted.

### Robust model

\(y = A\alpha + e\) with sparse \(e\):

\[
\min_{\alpha,e}\ \|\alpha\|_1 + \|e\|_1 + \frac{\lambda}{2}\|y - A\alpha - e\|_2^2
\]

Classify from \(\alpha\) using the same SRC residual rule.

### Results

![Random pixel corruption](accuracy_vs_corruption_pixel.png)

![Random patch corruption](accuracy_vs_corruption_patch.png)

### Observations

- **Pixels:** both methods degrade gradually; performance is similar.
- **Patches:** more harmful; robust SRC can beat standard SRC at moderate \(p\) (explicit \(e\) models localized outliers).
- **High \(p\):** both collapse when corruption dominates the signal.
- Robust SRC is ~4–5× slower per image.

```bash
python Assignment_6/assignment_6_issue2_robust_outliers.py
```

---

## Issue 3: Selfies and outlier rejection

### Preprocessing

Eight selfies in `selfy/`: grayscale, face crop (Haar + fallback), resize to **96×84**, match Yale mean/std.

![Preprocessed selfies](preprocessed_selfies_grid.png)

### Can SRC recognize you?

**No.** Raw SRC forces a Yale label (often class 38) with poor fit (\(r_1 \approx 0.92\)–\(0.95\), \(r_2/r_1 \approx 1\)). You are not among the 38 training identities.

### Outlier-rejection gate

Reject as **unknown** if any holds (thresholds from 200 Yale test faces, 95th percentile):

| Test | Meaning |
|------|---------|
| \(r_{(1)} > \tau_{\mathrm{res}}\) | No class fits well |
| \(r_{(2)}/r_{(1)} < \tau_{\mathrm{ratio}}\) | Ambiguous winner |
| \(\|y - A\alpha\|_2 > \tau_{\mathrm{total}}\) | Poor reconstruction |
| concentration \(< \tau_{\mathrm{conc}}\) | Coefficients not focused on one class |

### Results

| Probe | Count | Accepted | Rejected |
|-------|------:|---------:|---------:|
| Selfies (unknown) | 8 | 0 | **8** |
| Yale friend (in DB) | 1 | **1** | 0 (true class 20) |
| Yale validation | 200 | 173 | 27 (86.5%) |

Combined: **100%** selfie rejection; friend correctly accepted.

```bash
python Assignment_6/assignment_6_issue3_selfies.py --calibration-max 200
```

---

## LaTeX report

Compile the full report (with figures) from this folder:

```bash
cd Assignment_6
pdflatex assignment_6_report.tex
```

---

## Reference

J. Wright, A. Y. Yang, A. Ganesh, S. S. Sastry, and Y. Ma, “Robust face recognition via sparse representation,” *IEEE TPAMI*, 31(2):210–227, 2009.
