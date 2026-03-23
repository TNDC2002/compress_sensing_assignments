#!/usr/bin/env python3
"""
Compare exhaustive L0 search vs LP-based sparse recovery.

Method 1 (from Q1):
    min ||x||_0  subject to A x = y
    solved by exhaustive support search (S=3)

Method 2 (Q2):
    min ||x||_1  subject to A x = y
    solved as a linear program with scipy.optimize.linprog
"""

from __future__ import annotations

import argparse
import time
from itertools import combinations
from typing import Dict, Iterable, Tuple

import numpy as np
import scipy.io as sio
from scipy.optimize import linprog


def _pick_key(data: Dict[str, np.ndarray], *candidates: str) -> np.ndarray:
    for k in candidates:
        if k in data:
            return data[k]
    raise KeyError(f"None of these keys found: {candidates}")


def recover_l0_exhaustive(
    A: np.ndarray, y: np.ndarray, sparsity: int = 3
) -> Tuple[np.ndarray, Tuple[int, ...], float]:
    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    m, n = A.shape

    best_res = np.inf
    best_supp: Tuple[int, ...] | None = None
    best_xs: np.ndarray | None = None

    for supp in combinations(range(n), sparsity):
        As = A[:, supp]
        xs, *_ = np.linalg.lstsq(As, y, rcond=None)
        res = np.linalg.norm(As @ xs - y, ord=2)
        if res < best_res:
            best_res = float(res)
            best_supp = supp
            best_xs = xs

    assert best_supp is not None and best_xs is not None
    x = np.zeros(n, dtype=float)
    x[list(best_supp)] = best_xs
    return x, best_supp, best_res


def recover_lp_l1(A: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Basis pursuit as LP:
        min sum(t_i)
        s.t. A x = y
             -t <= x <= t
             t >= 0
    Variables are z = [x, t] in R^(2N).
    """
    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    m, n = A.shape

    # Objective: minimize sum(t)
    c = np.concatenate([np.zeros(n), np.ones(n)])

    # Equality: A x = y  -> [A, 0] z = y
    A_eq = np.hstack([A, np.zeros((m, n))])
    b_eq = y

    # Inequalities for -t <= x <= t:
    # x - t <= 0
    # -x - t <= 0
    I = np.eye(n)
    A_ub = np.vstack(
        [
            np.hstack([I, -I]),
            np.hstack([-I, -I]),
        ]
    )
    b_ub = np.zeros(2 * n)

    # Bounds: x free, t >= 0
    bounds = [(None, None)] * n + [(0, None)] * n

    result = linprog(
        c=c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    if not result.success:
        raise RuntimeError(f"linprog failed: {result.message}")

    x = result.x[:n]
    residual = float(np.linalg.norm(A @ x - y, ord=2))
    return x, residual


def top_support(x: np.ndarray, s: int = 3) -> Tuple[int, ...]:
    idx = np.argsort(np.abs(x))[-s:]
    return tuple(sorted(idx.tolist()))


def print_result(title: str, x: np.ndarray, residual: float, s: int = 3) -> None:
    supp = top_support(x, s=s)
    vals = x[list(supp)]
    print(f"\n--- {title} ---")
    print(f"Top-{s} support (0-based): {supp}")
    print(f"Top-{s} support (1-based): {[i + 1 for i in supp]}")
    print(f"Values at support: {vals}")
    print(f"Residual ||A x - y||_2: {residual:.6e}")
    print(f"L1 norm ||x||_1: {np.linalg.norm(x, ord=1):.6e}")


def compare_one_system(name: str, A: np.ndarray, y: np.ndarray, sparsity: int = 3) -> None:
    print(f"\n================ {name} ================")
    print(f"A shape: {A.shape}, y shape: {np.asarray(y).shape}")

    t0 = time.perf_counter()
    x_l0, supp_l0, res_l0 = recover_l0_exhaustive(A, y, sparsity=sparsity)
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    x_lp, res_lp = recover_lp_l1(A, y)
    t3 = time.perf_counter()

    print_result("Exhaustive L0", x_l0, res_l0, s=sparsity)
    print(f"Exact L0 support (0-based): {supp_l0}")
    print_result("LP (L1 / basis pursuit)", x_lp, res_lp, s=sparsity)

    l2_diff = np.linalg.norm(x_l0 - x_lp, ord=2)
    print(f"\n||x_l0 - x_lp||_2: {l2_diff:.6e}")
    print(f"Time L0 exhaustive: {(t1 - t0):.6f} sec")
    print(f"Time LP (linprog):  {(t3 - t2):.6f} sec")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare exhaustive L0 and LP-based sparse recovery")
    parser.add_argument("mat_file", help="Path to .mat file")
    parser.add_argument("--sparsity", type=int, default=3, help="Known sparsity level (default: 3)")
    args = parser.parse_args()

    data = sio.loadmat(args.mat_file)
    Af = _pick_key(data, "Af")
    Ar = _pick_key(data, "Ar")
    yf = _pick_key(data, "yf", "Yf")
    yr = _pick_key(data, "yr", "Yr")

    compare_one_system("(Af, yf)", Af, yf, sparsity=args.sparsity)
    compare_one_system("(Ar, yr)", Ar, yr, sparsity=args.sparsity)


if __name__ == "__main__":
    main()
