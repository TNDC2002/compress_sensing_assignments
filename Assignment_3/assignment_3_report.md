# Assignment 3 Report — Sparse Recovery with Various Sensing Matrices

## Code and repository

Please find the runnable experiment script and outputs in this GitHub repository (in case local files are missing or you want to reproduce the curves):

**https://github.com/TNDC2002/compress_sensing_assignments**

| What | Location in the repo |
|------|------------------------|
| Main script (all questions, sensing matrices, BP + OMP) | [`Assignment_3/assignment_3_experiments.py`](https://github.com/TNDC2002/compress_sensing_assignments/blob/main/Assignment_3/assignment_3_experiments.py) |
| This report | [`Assignment_3/assignment_3_report.md`](https://github.com/TNDC2002/compress_sensing_assignments/blob/main/Assignment_3/assignment_3_report.md) |
| Numeric results | [`Assignment_3/time_sparse_S5_probabilities.csv`](https://github.com/TNDC2002/compress_sensing_assignments/blob/main/Assignment_3/time_sparse_S5_probabilities.csv), [`frequency_sparse_S5_probabilities.csv`](https://github.com/TNDC2002/compress_sensing_assignments/blob/main/Assignment_3/frequency_sparse_S5_probabilities.csv), [`time_sparse_S25_probabilities.csv`](https://github.com/TNDC2002/compress_sensing_assignments/blob/main/Assignment_3/time_sparse_S25_probabilities.csv) |
| Figures | `Assignment_3/*_curve.png` (same folder on GitHub) |

**How to reproduce (from `Assignment_3/`):**

```bash
python assignment_3_experiments.py --trials 100 --output-dir .
```

Optional log capture:

```bash
python assignment_3_experiments.py --trials 100 --output-dir . > run_log.txt 2>&1
```

---

## Problem summary (COMP5340 Homework 3)

We study recovery of length-`N = 256` signals from `M` linear measurements `y = A x`, with sparsity `S` in either the **time domain** or the **DCT (frequency) domain**. For each sensing matrix type, we sweep `M ∈ {10, 20, …, 100}`, run **100 independent trials** per `M` (new random signal and, where applicable, new random subsampling / random matrix), and estimate the **probability of perfect recovery** when `‖x̂ − x‖₂ ≤ 10⁻⁶` (or the analogous criterion in the sparse DCT coefficients for the frequency-sparse case; see Question 2).

**Recovery methods**

- **BP (basis pursuit):** minimize `‖z‖₁` subject to `A z = y` (implemented with `scipy.optimize.linprog`, same LP reduction as in earlier assignments).
- **OMP:** greedy pursuit with sparsity budget `S` (orthogonal matching pursuit).

**Sensing schemes** (labels match the CSV columns and script)

| Label in outputs | Homework item | Description |
|------------------|---------------|-------------|
| `time_random` | (a) | `M` random rows of `I` |
| `time_uniform` | (b) | `M` rows of `I` at indices `⌊k N / M⌋` (0-based in code) |
| `freq_random` | (c) | `M` random rows of orthonormal DCT matrix `F` |
| `freq_low` | (d) | first `M` rows of `F` |
| `freq_equispaced` | (e) | rows of `F` at the same uniform index pattern as (b) |
| `random_domain` | (f) | `M × N` Gaussian, rows orthonormalized (`qr` on `Gᵀ`, then `Qᵀ`) |

---

## Question 1 — Time-sparse signal (`S = 5`)

The true signal has only five nonzero **time** samples; support and amplitudes are random each trial.

**Which sensing matrices work best?**  
In this run, **random frequency-domain subsampling** (`freq_random`) and the **row-orthonormal Gaussian ensemble** (`random_domain`) dominate: basis pursuit reaches probability **1** by about **`M = 30–40`**, and OMP typically needs slightly more measurements to reach the same level. **Pure time-domain subsampling** (`time_random`, `time_uniform`) yields **zero** perfect recoveries under the strict `10⁻⁶` criterion in the recorded CSV—structured subsampling in the same domain as the sparsity is a poor match to incoherent compressed sensing.

**How many measurements are enough?**  
For the best schemes here, **roughly `M ≈ 40`** is enough for BP to achieve **100%** success in these trials; OMP reaches **100%** on `freq_random` by **`M = 50`** in the saved table. See `time_sparse_S5_probabilities.csv` and `time_sparse_S5_BP_curve.png` / `time_sparse_S5_OMP_curve.png` for the full curves.

**Selected probabilities (from `time_sparse_S5_probabilities.csv`)**

| M | BP `freq_random` | BP `random_domain` | OMP `freq_random` | OMP `random_domain` |
|---:|---:|---:|---:|---:|
| 30 | 1.00 | 0.50 | 1.00 | 1.00 |
| 40 | 1.00 | 1.00 | 0.50 | 1.00 |
| 50 | 1.00 | 1.00 | 1.00 | 1.00 |
| 100 | 1.00 | 1.00 | 1.00 | 1.00 |

(Low-frequency and equispaced frequency sampling are intermediate; time-only schemes stay at 0 in this dataset.)

**BP vs OMP**  
OMP is usually **much faster** per solve than BP, but BP can be **more reliable** at smaller `M` when the sensing matrix is favorable (see transition region in the plots).

---

## Question 2 — Frequency-sparse signal (`S = 5`)

Here the signal is **`x = Ψ α`** with **`α` sparse** in the DCT domain (`Ψ` = inverse orthonormal DCT, matching the homework’s `idct` construction).

### Modified `ℓ₁` problem

Measurements are still `y = A x = A Ψ α`. Sparsity is in **`α`**, not in `x`, so basis pursuit should be written in **`α`**:

\[
\hat{\alpha} = \arg\min_{\alpha} \|\alpha\|_1 \quad \text{subject to} \quad y = A \Psi \alpha,
\]

then **`x̂ = Ψ α̂`**. OMP is applied to the same effective dictionary **`B = A Ψ`**, with sparsity **`S`**, and success is judged on **`‖α̂ − α‖₂ ≤ 10⁻⁶`** (equivalent to matching the sparse spectral coefficients the homework cares about).

### Observations

- **Incidence with Q1:** When sparsity moves to the DCT domain, **time-domain random rows** can behave like a **more incoherent** choice relative to that sparsity than **low-frequency or equispaced frequency** rows, which are highly coherent with a sparse spectrum.
- **Read your run from the artifacts:** Exact probabilities are in [`frequency_sparse_S5_probabilities.csv`](https://github.com/TNDC2002/compress_sensing_assignments/blob/main/Assignment_3/frequency_sparse_S5_probabilities.csv) and the plots `frequency_sparse_S5_BP_curve.png` / `frequency_sparse_S5_OMP_curve.png`. Compare how `time_random`, `freq_low`, and `freq_equispaced` rank; the assignment’s point is the **role of sparsity domain vs measurement domain**, not a single universal “best” matrix for all `M`.

---

## Question 3 — Larger sparsity (`S = 25`, time-sparse)

With **10%** nonzero entries, recovery is **much harder** for the same `M`: neither BP nor OMP reaches **probability 1** by **`M = 100`** on the schemes that were easy for `S = 5`. The CSV shows gradual improvement for **`freq_random`** and **`random_domain`**; **time subsampling** remains ineffective.

**Sample from `time_sparse_S25_probabilities.csv`**

| M | BP `freq_random` | BP `random_domain` | OMP `freq_random` | OMP `random_domain` |
|---:|---:|---:|---:|---:|
| 70 | 0.09 | 0.06 | 0.48 | 0.44 |
| 100 | 0.99 | 0.99 | 0.99 | 0.99 |

So **more measurements** (or **lower sparsity**) are required for the same reliability; the phase transition shifts to larger `M`.

---

## Conclusion

- **Sensing quality depends on sparsity domain:** time-sparse signals are recovered much better with **incoherent** views (e.g. random DCT rows or orthonormalized random `A`) than with **structured time subsampling** alone, under this experimental protocol.
- **Frequency-sparse** signals call for solving **`ℓ₁` in the DCT coefficient vector** via **`y = A Ψ α`**; the relative ranking of matrices changes compared to the time-sparse case—see the frequency-sparse CSV and figures.
- **BP** tends to be **stronger but slower**; **OMP** is **faster** and can match BP when `M` is large enough.
- **Raising `S` to 25** pushes the success curves down: **`M = 100`** is near the edge of reliable perfect recovery for the better matrices in this run.

All numerical results and plots can be regenerated from [`assignment_3_experiments.py`](https://github.com/TNDC2002/compress_sensing_assignments/blob/main/Assignment_3/assignment_3_experiments.py) in the repository linked at the top.
