#!/usr/bin/env python3
"""
Recover sparse x from Assignment I with:
  - L0 exhaustive search
  - LP (L1 / basis pursuit)
  - MP
  - OMP
  - SP
  - CoSaMP

Also reports per-method runtime for computational-cost comparison.
"""

from __future__ import annotations

import argparse
import time
from itertools import combinations
from typing import Callable, Dict, Tuple

import numpy as np
import scipy.io as sio
from scipy.optimize import linprog


Array = np.ndarray
RecoveryFn = Callable[[Array, Array, int, int, float], Array]


def _pick_key(data: Dict[str, Array], *candidates: str) -> Array:
    for k in candidates:
        if k in data:
            return data[k]
    raise KeyError(f"None of these keys found: {candidates}")


def _normalize_cols(A: Array) -> Tuple[Array, Array]:
    norms = np.linalg.norm(A, axis=0)
    norms[norms < 1e-12] = 1.0
    return A / norms, norms


def top_support(x: Array, s: int) -> Tuple[int, ...]:
    idx = np.argsort(np.abs(x))[-s:]
    return tuple(sorted(idx.tolist()))


def recover_l0_exhaustive(A: Array, y: Array, sparsity: int, *_unused: object) -> Array:
    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    _, n = A.shape
    best_res = np.inf
    best_x = np.zeros(n, dtype=float)

    for supp in combinations(range(n), sparsity):
        As = A[:, supp]
        xs, *_ = np.linalg.lstsq(As, y, rcond=None)
        res = np.linalg.norm(As @ xs - y)
        if res < best_res:
            best_res = float(res)
            best_x[:] = 0.0
            best_x[list(supp)] = xs
    return best_x


def recover_lp_l1(A: Array, y: Array, *_unused: object) -> Array:
    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    m, n = A.shape

    c = np.concatenate([np.zeros(n), np.ones(n)])
    A_eq = np.hstack([A, np.zeros((m, n))])
    b_eq = y
    I = np.eye(n)
    A_ub = np.vstack([np.hstack([I, -I]), np.hstack([-I, -I])])
    b_ub = np.zeros(2 * n)
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
    return result.x[:n]


def recover_mp(A: Array, y: Array, sparsity: int, max_iter: int, tol: float) -> Array:
    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    A_n, col_norms = _normalize_cols(A)
    _, n = A.shape

    x_scaled = np.zeros(n, dtype=float)
    r = y.copy()
    for _ in range(max_iter):
        corr = A_n.T @ r
        j = int(np.argmax(np.abs(corr)))
        x_scaled[j] += corr[j]
        r = y - A_n @ x_scaled
        if np.linalg.norm(r) <= tol or np.count_nonzero(np.abs(x_scaled) > 1e-12) >= sparsity:
            break
    return x_scaled / col_norms


def recover_omp(A: Array, y: Array, sparsity: int, max_iter: int, tol: float) -> Array:
    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    _, n = A.shape

    supp: list[int] = []
    r = y.copy()
    x = np.zeros(n, dtype=float)

    for _ in range(min(max_iter, sparsity)):
        corr = A.T @ r
        j = int(np.argmax(np.abs(corr)))
        if j not in supp:
            supp.append(j)
        As = A[:, supp]
        xs, *_ = np.linalg.lstsq(As, y, rcond=None)
        r = y - As @ xs
        if np.linalg.norm(r) <= tol:
            break

    if supp:
        x[supp] = xs
    return x


def recover_sp(A: Array, y: Array, sparsity: int, max_iter: int, tol: float) -> Array:
    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    _, n = A.shape

    supp = set(np.argsort(np.abs(A.T @ y))[-sparsity:].tolist())
    x = np.zeros(n, dtype=float)

    for _ in range(max_iter):
        supp_list = sorted(supp)
        As = A[:, supp_list]
        xs, *_ = np.linalg.lstsq(As, y, rcond=None)
        r = y - As @ xs
        if np.linalg.norm(r) <= tol:
            break

        merged = set(supp_list) | set(np.argsort(np.abs(A.T @ r))[-sparsity:].tolist())
        merged_list = sorted(merged)
        Amerged = A[:, merged_list]
        xmerged, *_ = np.linalg.lstsq(Amerged, y, rcond=None)
        keep_idx = np.argsort(np.abs(xmerged))[-sparsity:]
        new_supp = {merged_list[i] for i in keep_idx}
        if new_supp == supp:
            break
        supp = new_supp

    if supp:
        supp_list = sorted(supp)
        As = A[:, supp_list]
        xs, *_ = np.linalg.lstsq(As, y, rcond=None)
        x[supp_list] = xs
    return x


def recover_cosamp(A: Array, y: Array, sparsity: int, max_iter: int, tol: float) -> Array:
    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    _, n = A.shape

    supp: set[int] = set()
    x = np.zeros(n, dtype=float)
    r = y.copy()

    for _ in range(max_iter):
        proxy = A.T @ r
        omega = set(np.argsort(np.abs(proxy))[-2 * sparsity :].tolist())
        merged = sorted(supp | omega)
        Amerged = A[:, merged]
        b, *_ = np.linalg.lstsq(Amerged, y, rcond=None)
        keep_idx = np.argsort(np.abs(b))[-sparsity:]
        supp = {merged[i] for i in keep_idx}

        x[:] = 0.0
        supp_list = sorted(supp)
        As = A[:, supp_list]
        xs, *_ = np.linalg.lstsq(As, y, rcond=None)
        x[supp_list] = xs
        r = y - A @ x
        if np.linalg.norm(r) <= tol:
            break
    return x


def evaluate_system(
    name: str,
    A: Array,
    y: Array,
    sparsity: int,
    max_iter: int,
    tol: float,
) -> Dict[str, Array]:
    methods: Dict[str, RecoveryFn] = {
        "L0 exhaustive": recover_l0_exhaustive,
        "LP (L1)": recover_lp_l1,
        "MP": recover_mp,
        "OMP": recover_omp,
        "SP": recover_sp,
        "CoSaMP": recover_cosamp,
    }
    out: Dict[str, Array] = {}
    y = np.asarray(y, dtype=float).reshape(-1)

    print(f"\n===== {name} =====")
    print("Method         | Residual ||Ax-y||_2 | Top support (0-based) | Time (ms)")
    print("-" * 72)

    for method_name, fn in methods.items():
        t0 = time.perf_counter()
        x_hat = fn(A, y, sparsity, max_iter, tol)
        t1 = time.perf_counter()
        residual = np.linalg.norm(A @ x_hat - y)
        supp = top_support(x_hat, sparsity)
        elapsed_ms = 1000.0 * (t1 - t0)
        print(f"{method_name:14s} | {residual:18.6e} | {str(supp):22s} | {elapsed_ms:9.3f}")
        out[method_name] = x_hat
    return out


def print_pairwise_differences(results: Dict[str, Array], sparsity: int) -> None:
    names = list(results.keys())
    print("\nPairwise differences between recovered x (L2 norm):")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            d = np.linalg.norm(results[a] - results[b])
            same_supp = top_support(results[a], sparsity) == top_support(results[b], sparsity)
            print(f"- {a:14s} vs {b:14s}: ||dx||_2 = {d:.6e}, same top-{sparsity} support = {same_supp}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Greedy sparse recovery comparison")
    parser.add_argument("mat_file", help="Path to .mat file")
    parser.add_argument("--sparsity", type=int, default=3, help="Known sparsity level")
    parser.add_argument("--max-iter", type=int, default=50, help="Maximum greedy iterations")
    parser.add_argument("--tol", type=float, default=1e-9, help="Residual tolerance")
    args = parser.parse_args()

    data = sio.loadmat(args.mat_file)
    Af = _pick_key(data, "Af")
    Ar = _pick_key(data, "Ar")
    yf = _pick_key(data, "yf", "Yf")
    yr = _pick_key(data, "yr", "Yr")

    results_f = evaluate_system("(Af, yf)", Af, yf, args.sparsity, args.max_iter, args.tol)
    print_pairwise_differences(results_f, args.sparsity)

    results_r = evaluate_system("(Ar, yr)", Ar, yr, args.sparsity, args.max_iter, args.tol)
    print_pairwise_differences(results_r, args.sparsity)

    print("\nCross-system consistency (same method on Af/yf vs Ar/yr):")
    for name in results_f.keys():
        d = np.linalg.norm(results_f[name] - results_r[name])
        supp_same = top_support(results_f[name], args.sparsity) == top_support(
            results_r[name], args.sparsity
        )
        print(f"- {name:14s}: ||x_f - x_r||_2 = {d:.6e}, same top-{args.sparsity} support = {supp_same}")


if __name__ == "__main__":
    main()

