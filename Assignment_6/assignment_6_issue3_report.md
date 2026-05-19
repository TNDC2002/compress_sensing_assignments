# Assignment 6 Report — Issue 3: Selfies and Outlier Rejection

## Goal

1. Preprocess personal selfies to match cropped Yale B format (`96 × 84` grayscale).
2. Test whether SRC recognizes you after preprocessing.
3. Design and demonstrate an **outlier-rejection** rule (“person not in database”).
4. Evaluate on an **extended database**: Yale B training gallery + appended selfie probes (unknown identities).

Selfies used: `Assignment_6/selfy/*.jpg` (8 images).

## Preprocessing

Pipeline (`preprocess_selfie_to_yale` in `assignment_6_src_common.py`):

1. Grayscale + EXIF orientation correction.
2. **Face detection** (OpenCV Haar cascade) with padding; fallback to centered upper-body crop.
3. Resize to **96 × 84** (Lanczos).
4. Match first-order statistics to Yale B: scale to database mean/std (`≈ 69.5`, `≈ 62.7`).

Outputs:

- Per-image chips: `preprocessed_selfies/*_yale.png`
- Montage: `preprocessed_selfies_grid.png`

## Recognition without outlier rejection

Training: standard SRC dictionary from **1209** Yale training faces (38 subjects), same split as Issue 1 (`seed = 0`, 50/50 per class).

| Observation | Result |
|-------------|--------|
| Raw SRC label on selfies | Almost always **forced** to some Yale ID (often class 38) |
| True identity in gallery? | **No** — you are not among the 38 subjects |
| Meaningful “recognition”? | **No** — low-quality fit to the nearest Yale subject |

Even after preprocessing, selfies do **not** correspond to a subject in the database. SRC still returns a label because it must choose the smallest class residual among the 38 candidates.

Typical selfie SRC features (all 8 probes):

- Minimum class residual `r₁ ≈ 0.92–0.95` (poor fit).
- Residual ratio `r₂/r₁ ≈ 1.00–1.01` (two classes fit equally poorly → ambiguous).
- Low sparsity concentration (`≈ 0.04–0.06`): coefficients spread across many training identities.

## Outlier-rejection strategy

After sparse coding, compute per-class residuals `r_k` (standard SRC) and global fit statistics. **Reject** (declare *unknown*) if any of the following holds:

| Test | Intuition |
|------|-----------|
| `r₁ > τ_res` | No class reconstructs the face well. |
| `r₂/r₁ < τ_ratio` | Winner is not clearly better than runner-up. |
| `‖y − Aα‖₂ > τ_total` | Overall reconstruction is poor. |
| sparsity concentration `< τ_conc` | L1 mass not focused on one identity. |

Thresholds `{τ_res, τ_ratio, τ_total, τ_conc}` are calibrated on **in-database** Yale test faces (200 images, 95th-percentile envelopes) so most true subjects are accepted.

Implementation: `calibrate_outlier_thresholds`, `is_outlier` in `assignment_6_src_common.py`.

## Demonstration on extended database

**Setup**

- Gallery: Yale B train split (38 classes).
- Unknown probes: 8 preprocessed selfies (label 39 conceptually — not used in training).
- In-database probe (“friend”): one **held-out Yale test** image (`yale_test_1200`, true class **20**) simulating a friend who *is* in the database.

**Calibrated thresholds (95th percentile, 200 validation faces)**

- `r₁ ≤ 0.927`
- `r₂/r₁ ≥ 1.003`
- `‖y − Aα‖₂ ≤ 0.409`
- concentration `≥ 0.054`

**Gate results**

| Probe type | Count | Accepted | Rejected | Notes |
|------------|------:|---------:|---------:|-------|
| Selfies (unknown) | 8 | 0 | **8** | All declared **UNKNOWN** |
| Yale friend (in DB) | 1 | **1** | 0 | Predicted class **20** (correct) |
| Yale validation (in DB) | 200 | 173 | 27 | **86.5%** acceptance |

Combined gate: **100%** unknown rejection on selfies, friend accepted, overall gate accuracy **87.1%** on the mixed evaluation set.

## Observations

1. **Preprocessing is necessary but not sufficient.** Selfies must be grayscale, aligned, and resized; without this, SRC inputs are out of distribution. Even with preprocessing, you are still not a Yale subject.
2. **SRC alone always assigns a label.** Minimum-residual classification has no “none of the above” option; unknown users are mislabeled to the nearest Yale identity.
3. **Outliers are detectable from SRC statistics.** Unknown faces show **high** `r₁`, **low** `r₂/r₁`, and **low** concentration — different from in-database test faces (friend probe: `r₁ ≈ 0.77`, `r₂/r₁ ≈ 1.21`, concentration `≈ 0.18`).
4. **Patch corruptions vs identity outliers.** Issue 2 corruption hurts accuracy but faces remain in-gallery; Issue 3 selfies are **structurally out-of-database** and are better handled by rejection than by forcing a class label.

## Reproduce

```bash
cd Assignment_6
python assignment_6_issue3_selfies.py --calibration-max 200
```

Outputs: `issue3_selfie_probe_results.csv`, `issue3_outlier_gate_summary.csv`, preprocessed images under `preprocessed_selfies/`.
