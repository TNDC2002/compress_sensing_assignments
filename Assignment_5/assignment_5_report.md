# Assignment 5 Report: Why l1 Minimization Encourages Sparsity

## Statement

Let `A in R^(M x N)`, `y in R^M`, and consider

`(P1): min ||x||_1  subject to  Ax = y`.

If `(P1)` has a unique solution `x_1`, then

`||x_1||_0 <= M`.

## Proof

Assume for contradiction that the unique minimizer `x_1` has more than `M` nonzero entries:

`k := ||x_1||_0 > M`.

Let `S = supp(x_1)` with `|S| = k`, and let `A_S` be the submatrix of `A` formed by columns indexed by `S`.
Since `A_S` has `M` rows and `k > M` columns, its columns are linearly dependent. Therefore, there exists a nonzero vector `h_S in R^k` such that

`A_S h_S = 0`.

Extend `h_S` to `h in R^N` by setting entries outside `S` to zero. Then:

`Ah = 0`.

So for every scalar `t`, the vector

`x(t) := x_1 + t h`

is feasible, because

`A x(t) = A x_1 + t A h = y`.

Now inspect the l1 norm near `t = 0`. For `i notin S`, both `x_{1,i} = 0` and `h_i = 0`, so those coordinates do not change. For `i in S`, choose `|t|` small enough so `sign(x_{1,i} + t h_i) = sign(x_{1,i})`. Then

`|x_{1,i} + t h_i| = |x_{1,i}| + t sign(x_{1,i}) h_i`.

Hence, for sufficiently small `|t|`,

`||x_1 + t h||_1 = ||x_1||_1 + t sum_{i in S} sign(x_{1,i}) h_i`.

Define

`c := sum_{i in S} sign(x_{1,i}) h_i`.

Three cases:

1. If `c > 0`, take `t < 0` small. Then `||x_1 + t h||_1 < ||x_1||_1`.
2. If `c < 0`, take `t > 0` small. Then `||x_1 + t h||_1 < ||x_1||_1`.
3. If `c = 0`, then for all sufficiently small `t`,
   `||x_1 + t h||_1 = ||x_1||_1`, so there are infinitely many feasible points with the same objective value.

In all cases, uniqueness of `x_1` is contradicted (either a strictly better feasible point exists, or another minimizer exists).

Therefore the assumption `||x_1||_0 > M` is false, and we must have

`||x_1||_0 <= M`.

QED.
