#!/usr/bin/env python3
"""
Exhaustive-search L0 recovery for a 3-sparse signal x from y = A x.

Given A (M x N) and y (M,), this script checks all supports of size S=3:
    1) choose support indices (columns of A)
    2) estimate nonzeros by least squares / pseudoinverse:
           x_S = argmin ||A_S x_S - y||_2
    3) keep support with minimum residual norm ||A x - y||_2
"""

from __future__ import annotations

import argparse
from itertools import combinations
from typing import Dict, Iterable, Tuple

import numpy as np
import scipy.io as sio


def _pick_key(data: Dict[str, np.ndarray], *candidates: str) -> np.ndarray:
    """Pick the first existing key from a .mat dictionary."""
    for k in candidates:
        if k in data:
            return data[k]
    raise KeyError(f"None of these keys found: {candidates}")


def recover_l0_exhaustive(
    A: np.ndarray,
    y: np.ndarray,
    sparsity: int = 3,
) -> Tuple[np.ndarray, Tuple[int, ...], float]:
    """
    Brute-force L0 recovery by exhaustive support search.

    Returns:
        x_best: recovered full-length vector (N,)
        support_best: chosen support indices
        residual_best: minimum ||A x - y||_2
    """
    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    m, n = A.shape
    if y.shape[0] != m:
        raise ValueError(f"Shape mismatch: A is {A.shape}, y has length {y.shape[0]}")
    if not (1 <= sparsity <= n):
        raise ValueError(f"sparsity must be in [1, {n}]")

    residual_best = np.inf
    support_best: Tuple[int, ...] | None = None
    xs_best: np.ndarray | None = None

    for support in combinations(range(n), sparsity):
        As = A[:, support]  # (M x S)
        # Least-squares estimate == pseudoinverse solution for this support.
        xs, *_ = np.linalg.lstsq(As, y, rcond=None)
        res = np.linalg.norm(As @ xs - y, ord=2)

        if res < residual_best:
            residual_best = float(res)
            support_best = support
            xs_best = xs

    assert support_best is not None and xs_best is not None
    x_best = np.zeros(n, dtype=float)
    x_best[list(support_best)] = xs_best
    return x_best, support_best, residual_best


def print_solution(name: str, x: np.ndarray, support: Iterable[int], residual: float) -> None:
    support = tuple(support)
    values = x[list(support)]
    print(f"\n=== {name} ===")
    print(f"Best support (0-based): {support}")
    print(f"Best support (1-based): {[i + 1 for i in support]}")
    print(f"Estimated nonzero values: {values}")
    print(f"Residual ||A x - y||_2: {residual:.6e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Exhaustive-search L0 sparse recovery")
    parser.add_argument("mat_file", help="Path to .mat file")
    parser.add_argument(
        "--sparsity", type=int, default=3, help="Assumed number of nonzeros in x (default: 3)"
    )
    args = parser.parse_args()

    data = sio.loadmat(args.mat_file)

    Af = _pick_key(data, "Af")
    Ar = _pick_key(data, "Ar")
    yf = _pick_key(data, "yf", "Yf")
    yr = _pick_key(data, "yr", "Yr")

    x_f, supp_f, res_f = recover_l0_exhaustive(Af, yf, sparsity=args.sparsity)
    x_r, supp_r, res_r = recover_l0_exhaustive(Ar, yr, sparsity=args.sparsity)

    print_solution("Recovered from (Af, yf)", x_f, supp_f, res_f)
    print_solution("Recovered from (Ar, yr)", x_r, supp_r, res_r)

    # If both are equivalent sensing systems, the recovered x should be very close.
    diff = np.linalg.norm(x_f - x_r, ord=2)
    print(f"\n||x_f - x_r||_2: {diff:.6e}")


if __name__ == "__main__":
    main()
