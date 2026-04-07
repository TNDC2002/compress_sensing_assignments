# Assignment 2 Report: Greedy Sparse Recovery and Runtime Comparison

## Problem Statement

Recover the sparse signal `x` from Assignment I using four greedy algorithms (`MP`, `OMP`, `SP`, and `CoSaMP`), compare their recovery results, and time all sparse recovery methods implemented so far to evaluate computational cost differences.

## Experimental Configuration

- **Dataset:** `../Assignment_1/COMP5340HW1.mat`
- **Systems evaluated:** `(Af, yf)` and `(Ar, yr)`
- **Assumed sparsity:** `S = 3`
- **Methods timed:**
  - `L0 exhaustive` (brute-force support search)
  - `LP (L1 / basis pursuit)`
  - `MP`
  - `OMP`
  - `SP`
  - `CoSaMP`
- **Execution command:**  
  `python greedy_recovery_comparison.py ../Assignment_1/COMP5340HW1.mat --sparsity 3`

## Q1. Can `x` be recovered with the four greedy algorithms?

Yes. All four greedy algorithms identified the same top-3 support on both sensing systems:

- **Recovered support (0-based):** `(2, 15, 25)`

In particular:

- `OMP`, `SP`, and `CoSaMP` achieved near machine-precision reconstruction residuals, indicating highly accurate recovery of both support and amplitudes.
- `MP` recovered the correct support, but with noticeably larger residuals, indicating weaker coefficient (amplitude) estimation.

### Reconstruction residuals `||Ax - y||_2`

| Method | `(Af, yf)` | `(Ar, yr)` |
|---|---:|---:|
| `L0 exhaustive` | `5.93e-15` | `7.58e-15` |
| `LP (L1)` | `9.82e-14` | `1.69e-13` |
| `MP` | `9.82e-01` | `5.47e-01` |
| `OMP` | `2.97e-15` | `1.47e-14` |
| `SP` | `5.93e-15` | `7.58e-15` |
| `CoSaMP` | `5.93e-15` | `7.58e-15` |

## Q2. Any difference in the recovery results?

Yes, there are meaningful differences in coefficient accuracy:

- `SP` and `CoSaMP` matched `L0 exhaustive` exactly in this run.
- `OMP` was effectively identical to `L0` (differences on the order of `1e-14`).
- `LP (L1)` was also very close (small numerical differences on the order of `1e-13` to `1e-12`).
- `MP` produced substantially different coefficient values (L2 differences of order `1` relative to the others), despite recovering the same support indices.

Therefore, while support recovery is consistent across methods in this case, amplitude recovery quality is method-dependent, with `OMP`, `SP`, and `CoSaMP` outperforming `MP`.

## Q3. Any difference in computational cost?

Yes. The timing differences are large.

### Typical runtime per solve (same experiment run)

- `L0 exhaustive`: approximately `7.6 - 10.9 s`
- `LP (L1)`: approximately `7 - 15 ms`
- `MP`: approximately `0.13 - 0.20 ms`
- `OMP`: approximately `0.23 - 0.27 ms`
- `SP`: approximately `0.23 - 0.27 ms`
- `CoSaMP`: approximately `0.26 - 0.32 ms`

### Interpretation

- `L0 exhaustive` is computationally prohibitive due to combinatorial support search.
- `LP (L1)` is far faster than exhaustive `L0`, but still slower than greedy methods.
- Greedy methods are orders of magnitude faster; among them, `MP` is fastest, while `OMP`, `SP`, and `CoSaMP` remain very fast and substantially more accurate in this dataset.

## Conclusion

`x` can be successfully recovered from Assignment I using all four greedy algorithms in terms of support identification. However, reconstruction quality differs: `OMP`, `SP`, and `CoSaMP` provide near-exact recovery, whereas `MP` is less accurate in coefficients. In computational cost, the gap is significant: exhaustive `L0` is by far the slowest, while greedy algorithms deliver high accuracy at millisecond or sub-millisecond runtime.

---

## Additional Experiment: Synthetic Signal (`N=256`, `S=5`)

### Problem Setup

To evaluate recovery behavior more systematically, a synthetic sparse-recovery experiment was conducted with:

- Signal length `N = 256`
- Sparsity `S = 5`
- Measurement counts `M = {5, 10, 15, ..., 100}`
- `100` random trials per `M`
- New random sparse signal and new random Gaussian sensing matrix in every trial
- Perfect recovery criterion: `||x_hat - x||_2 <= 1e-6`

Algorithms tested:

- `L1` minimization (basis pursuit)
- `MP`, `OMP`, `SP`, `CoSaMP`

Implementation and outputs:

- Script: `synthetic_recovery_probability_curve.py`
- Curve plot: `synthetic_recovery_probability_curve.png`
- Numerical table: `synthetic_recovery_probability_results.csv`

Execution command:

`python synthetic_recovery_probability_curve.py --trials 100 --m-start 5 --m-stop 100 --m-step 5`

### Results Summary

- For small `M` (`5` to `10`), all methods had near-zero recovery probability.
- Recovery probability increased with `M` for `L1`, `OMP`, `SP`, and `CoSaMP`.
- `SP` and `CoSaMP` showed stronger performance in the transition region (`M = 20` to `35`) than `OMP`.
- `L1` became highly reliable around `M >= 35` and reached `1.00` by `M = 40`.
- `OMP`, `SP`, and `CoSaMP` all approached `1.00` by about `M = 45` to `60`.
- `MP` remained at `0.00` perfect-recovery probability across all tested `M` under this strict criterion.

Selected points from the 100-trial run:

| `M` | `L1` | `MP` | `OMP` | `SP` | `CoSaMP` |
|---:|---:|---:|---:|---:|---:|
| 20 | 0.13 | 0.00 | 0.16 | 0.27 | 0.30 |
| 30 | 0.75 | 0.00 | 0.75 | 0.81 | 0.81 |
| 40 | 1.00 | 0.00 | 0.94 | 0.99 | 0.98 |
| 50 | 1.00 | 0.00 | 1.00 | 1.00 | 0.99 |
| 100 | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 |

### Computational Cost (Synthetic Experiment)

Average runtime per solve (from the full 100-trial sweep):

- `L1`: `44.932 ms`
- `MP`: `0.398 ms`
- `OMP`: `0.504 ms`
- `SP`: `0.597 ms`
- `CoSaMP`: `1.897 ms`

### Interpretation

- There is a clear trade-off between runtime and robustness.
- `MP` is fastest but fails to achieve perfect recovery reliably under `||x_hat - x||_2 <= 1e-6`.
- `OMP`, `SP`, and `CoSaMP` are still very fast and achieve high recovery probability as `M` grows.
- `L1` is slower than greedy methods but highly reliable, reaching perfect recovery probability at moderate-to-high measurement counts.

