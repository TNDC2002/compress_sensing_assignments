# Assignment 3 Report: Sparse Recovery with Various Sensing Matrices

## Repository and reproducibility

Code and outputs are available at:

- [https://github.com/TNDC2002/compress_sensing_assignments](https://github.com/TNDC2002/compress_sensing_assignments)
- Main script: `Assignment_3/assignment_3_experiments.py`
- This report: `Assignment_3/assignment_3_report.md`

To reproduce the full experiment from `Assignment_3`:

```bash
python assignment_3_experiments.py --trials 100 --output-dir .
```

## Experimental setup

- Signal dimension: `N = 256`
- Measurements: `M in {10, 20, ..., 100}`
- Trials per `M`: `100`
- Perfect recovery criterion: `||x_hat - x||_2 <= 1e-6`
- Recovery methods:
  - `BP` (basis pursuit / `l1` minimization)
  - `OMP` (greedy sparse recovery)
- Sensing matrix schemes:
  - `time_random`, `time_uniform`
  - `freq_random`, `freq_low`, `freq_equispaced`
  - `random_domain` (row-orthonormalized Gaussian matrix)

---

## Q1. Time-sparse signal (`S = 5`)

For time-sparse signals, the best-performing sensing matrices are the incoherent/random ones in transformed/random domains.

- `BP`: strongest schemes are `freq_random` and `random_domain`; both reach probability `1.0` by `M = 40`.
- `OMP`: strongest scheme is `freq_random`; reaches probability `1.0` by `M = 50` (also `random_domain` reaches `1.0` by `M = 50`).
- `time_random` and `time_uniform` remain near zero across all `M`.

Selected values from `time_sparse_S5_probabilities.csv`:

| M | BP freq_random | BP random_domain | OMP freq_random | OMP random_domain |
|---:|---:|---:|---:|---:|
| 20 | 0.16 | 0.10 | 0.50 | 0.33 |
| 30 | 0.89 | 0.81 | 0.91 | 0.80 |
| 40 | 1.00 | 0.99 | 0.99 | 0.98 |
| 50 | 1.00 | 1.00 | 1.00 | 1.00 |
| 100 | 1.00 | 1.00 | 1.00 | 1.00 |

---

## Q2. Frequency-sparse signal (`S = 5`)

Let `x = Psi alpha` where `alpha` is sparse in the DCT domain.

### Modified `l1` problem

Instead of solving for `x` directly, recover sparse coefficients `alpha`:

`alpha_hat = arg min ||alpha||_1  subject to  y = A Psi alpha`

Then reconstruct:

`x_hat = Psi alpha_hat`

In the implementation, `Psi` is generated consistently with DCT basis definition using `axis=0`.

### Observations

- `BP`: `time_random` and `random_domain` are best; both reach probability `1.0` at `M = 40`.
- `OMP`: same trend; `time_random` reaches `1.0` at `M = 40`, `random_domain` is `0.99` at `M = 40` and `1.0` at `M = 50`.
- Frequency-row sampling (`freq_random`, `freq_low`, `freq_equispaced`) stays near zero in this setting.

Selected values from `frequency_sparse_S5_probabilities.csv`:

| M | BP time_random | BP random_domain | OMP time_random | OMP random_domain |
|---:|---:|---:|---:|---:|
| 20 | 0.27 | 0.06 | 0.42 | 0.19 |
| 30 | 0.90 | 0.82 | 0.93 | 0.85 |
| 40 | 1.00 | 1.00 | 1.00 | 0.99 |
| 50 | 1.00 | 1.00 | 1.00 | 1.00 |
| 100 | 1.00 | 1.00 | 1.00 | 1.00 |

---

## Q3. Increased sparsity (`S = 25`, time-sparse)

As sparsity increases, recovery becomes significantly harder for the same number of measurements.

- Best schemes remain `freq_random` and `random_domain`.
- Neither `BP` nor `OMP` reaches full reliability early; both approach `0.99` only at `M = 100` for top schemes.
- `time_random` and `time_uniform` stay at `0.0`.

Selected values from `time_sparse_S25_probabilities.csv`:

| M | BP freq_random | BP random_domain | OMP freq_random | OMP random_domain |
|---:|---:|---:|---:|---:|
| 70 | 0.05 | 0.06 | 0.55 | 0.44 |
| 80 | 0.41 | 0.38 | 0.85 | 0.74 |
| 90 | 0.87 | 0.83 | 0.96 | 0.85 |
| 100 | 0.99 | 0.99 | 0.99 | 0.99 |

---

## Efficiency discussion

- `OMP` is computationally lighter and usually faster per recovery.
- `BP` often provides stronger robustness in challenging regions, but depends on sensing design and sparsity level.

## Final conclusion

- Recovery performance is strongly determined by the match between **sparsity domain** and **sensing incoherence**.
- For this assignment run:
  - Time-sparse signals are best recovered with `freq_random` / `random_domain`.
  - Frequency-sparse signals are best recovered with `time_random` / `random_domain`.
- Increasing sparsity from `S=5` to `S=25` shifts the recovery phase transition to much larger `M`, confirming the expected compressive sensing behavior.
