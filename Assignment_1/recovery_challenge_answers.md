# Recovery Challenge Answers

## Please find my scripts at https://github.com/TNDC2002/compress_sensing_assignments in case .py files are corrupted or not uploadable.

## Problem Setup

Given:
- Signal length: `N = 100`
- Sparsity bound: `S <= 3` (for the provided dataset)
- Two sensing systems from the `.mat` file:
  - `(Af, yf)`
  - `(Ar, yr)`

Goal: recover sparse signal `x` from measurements `y = A x`.

---

## 1) Exhaustive-search `l0` minimization

Yes, `x` can be recovered by classic exhaustive search over supports of size `S=3`.

Method:
1. Enumerate all 3-element supports `Omega` from `{1,...,100}` (equivalent to `nchoosek(1:N,3)` in MATLAB).
2. For each support, build submatrix `A_Omega` (columns indexed by `Omega`).
3. Estimate coefficients on that support by least squares / pseudoinverse:
   `x_Omega = arg min ||A_Omega x_Omega - y||_2`.
4. Form full vector `x` (zeros outside `Omega`) and compute residual `||A x - y||_2`.
5. Choose the support with minimum residual.

Recovered result (both sensing systems):
- Support (1-based): **[3, 16, 26]**
- Nonzero values: **[5, 3, 40]**
- Residual:
  - `(Af, yf)`: `~1.07e-14`
  - `(Ar, yr)`: `~8.25e-15`

So exhaustive `l0` recovery successfully reconstructs the same sparse signal from both systems.

---

## 2) Linear programming approach and comparison

To solve via LP, use basis pursuit (`l1` minimization):

`min ||x||_1  subject to  A x = y`.

Standard LP form (using auxiliary variable `t`):
- Minimize `sum_i t_i`
- Subject to:
  - `A x = y`
  - `-t <= x <= t`
  - `t >= 0`

In Python, this is solved with `scipy.optimize.linprog`.

### Difference in recovered signal

For both `(Af, yf)` and `(Ar, yr)`, LP recovered the same support and values as exhaustive `l0`:
- Support (1-based): `[3, 16, 26]`
- Values: `[5, 3, 40]`
- Difference between methods is only numerical precision noise (`~1e-12` to `1e-13` level).

### Timing difference

Measured on this dataset (`M=25, N=100, S=3`):
- Exhaustive `l0`: about **8.6 to 9.2 seconds**
- LP (`linprog`): about **0.01 to 0.02 seconds**

Conclusion: recovery result is effectively the same, but LP is dramatically faster.

---

## 3) Observations on sensing matrices and recovered signals

1. `Af` and `Ar` are different sensing matrices, but both recover the same sparse `x`.
2. Residuals are near machine precision (`~1e-14`), so both models satisfy `A x ~= y` very accurately.
3. This indicates both matrices are suitable for sparse recovery at low sparsity (`S=3`) in this problem.
4. Equivalent recovery from both systems increases confidence that the recovered support and values are correct (not an artifact of one matrix).

---

## 4) Increasing sparsity from `S=3` to `S=10` (expected behavior and observations)

For custom experiments (random sparse `x`, generate `y = A x`, then recover):

### Practical observation about algorithms

- Exhaustive `l0` search scales combinatorially with `S` because it must test `C(100,S)` supports:
  - `C(100,3) = 161,700` (manageable)
  - `C(100,10) ~ 1.73e13` (computationally impractical)
- LP (`l1`) remains computationally tractable and is the practical method as `S` grows.

### Recovery behavior as `S` increases

- For small `S` (like 3), LP and exhaustive `l0` usually agree and recover exactly.
- As `S` increases toward 10 with fixed `M=25`, recovery becomes harder:
  - exact support recovery rate decreases,
  - reconstruction error tends to increase,
  - differences between matrix quality (`Af` vs `Ar`) become more visible.

### Summary for Q4

- Exact exhaustive `l0` is useful as a ground-truth benchmark at small `S`.
- LP is the scalable approach and should be preferred for larger sparsity levels.
- The sparsity-vs-recovery tradeoff is the key phenomenon: higher sparsity with fixed measurements gives lower recoverability.

---

## Final Conclusion

The unknown 3-sparse signal was successfully recovered from both sensing systems:
- `x(3)=5`, `x(16)=3`, `x(26)=40` (1-based indexing), all other entries zero.

Both exhaustive `l0` and LP (`l1`) produced the same reconstruction on this dataset, but LP was orders of magnitude faster. This matches core compressed sensing theory: `l1` can act as an efficient surrogate for `l0` under favorable sensing conditions.
