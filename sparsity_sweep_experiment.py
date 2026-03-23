#!/usr/bin/env python3
"""
Q4 experiment: generate random sparse x, form y=Ax, recover with:
  1) exhaustive support search (L0, with known sparsity S)
  2) LP / L1 (basis pursuit)
for S = 3..10 on both sensing matrices Af and Ar.
"""

from __future__ import annotations

import argparse
import math
import time
from itertools import combinations
from typing import Dict, Tuple

import numpy as np
import scipy.io as sio
from scipy.optimize import linprog


def _pick_key(data: Dict[str, np.ndarray], *candidates: str) -> np.ndarray:
    for k in candidates:
        if k in data:
            return data[k]
    raise KeyError(f"None of these keys found: {candidates}")


def recover_l0_exhaustive(A: np.ndarray, y: np.ndarray, sparsity: int) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    _, n = A.shape
    best_res = np.inf
    best_x = np.zeros(n, dtype=float)

    for supp in combinations(range(n), sparsity):
        As = A[:, supp]
        xs, *_ = np.linalg.lstsq(As, y, rcond=None)
        res = np.linalg.norm(As @ xs - y, ord=2)
        if res < best_res:
            best_res = float(res)
            best_x = np.zeros(n, dtype=float)
            best_x[list(supp)] = xs
    return best_x


def recover_lp_l1(A: np.ndarray, y: np.ndarray) -> np.ndarray:
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


def support(x: np.ndarray, s: int, tol: float = 1e-6) -> Tuple[int, ...]:
    nz = np.where(np.abs(x) > tol)[0]
    if len(nz) == s:
        return tuple(sorted(nz.tolist()))
    # Fallback: use largest magnitudes if LP returns tiny numerical leftovers.
    idx = np.argsort(np.abs(x))[-s:]
    return tuple(sorted(idx.tolist()))


def one_trial(
    A: np.ndarray,
    s: int,
    rng: np.random.Generator,
    amp_low: int,
    amp_high: int,
) -> Dict[str, float]:
    _, n = A.shape
    true_support = tuple(sorted(rng.choice(n, size=s, replace=False).tolist()))
    x_true = np.zeros(n, dtype=float)
    # Integer amplitudes, avoid zeros.
    vals = rng.integers(amp_low, amp_high + 1, size=s).astype(float)
    vals[vals == 0] = 1.0
    signs = rng.choice([-1.0, 1.0], size=s)
    x_true[list(true_support)] = vals * signs

    y = A @ x_true

    t0 = time.perf_counter()
    x_l0 = recover_l0_exhaustive(A, y, sparsity=s)
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    x_lp = recover_lp_l1(A, y)
    t3 = time.perf_counter()

    supp_l0 = support(x_l0, s)
    supp_lp = support(x_lp, s)

    succ_l0 = float(supp_l0 == true_support)
    succ_lp = float(supp_lp == true_support)
    relerr_l0 = float(np.linalg.norm(x_l0 - x_true) / (np.linalg.norm(x_true) + 1e-12))
    relerr_lp = float(np.linalg.norm(x_lp - x_true) / (np.linalg.norm(x_true) + 1e-12))

    return {
        "succ_l0": succ_l0,
        "succ_lp": succ_lp,
        "relerr_l0": relerr_l0,
        "relerr_lp": relerr_lp,
        "time_l0": (t1 - t0),
        "time_lp": (t3 - t2),
    }


def run_matrix_experiment(
    name: str,
    A: np.ndarray,
    s_min: int,
    s_max: int,
    trials: int,
    seed: int,
    amp_low: int,
    amp_high: int,
) -> None:
    rng = np.random.default_rng(seed)
    print(f"\n===== Matrix {name} shape={A.shape} =====")
    print("S | L0 succ | LP succ | mean relerr L0 | mean relerr LP | mean t_L0(s) | mean t_LP(s)")
    print("-" * 90)
    n = A.shape[1]
    for s in range(s_min, s_max + 1):
        # Exhaustive L0 is combinatorial; beyond this threshold it becomes impractical.
        max_l0_combinations = 5_000_000
        num_combos = int(math.comb(n, s))
        do_l0 = num_combos <= max_l0_combinations

        rows = [one_trial(A, s, rng, amp_low, amp_high) for _ in range(trials)] if do_l0 else []

        if do_l0:
            succ_l0 = np.mean([r["succ_l0"] for r in rows])
            rel_l0 = np.mean([r["relerr_l0"] for r in rows])
            t_l0 = np.mean([r["time_l0"] for r in rows])
        else:
            succ_l0 = np.nan
            rel_l0 = np.nan
            t_l0 = np.nan

        # Always report LP by running only LP part when L0 is skipped.
        lp_rows = []
        if do_l0:
            lp_rows = rows
        else:
            for _ in range(trials):
                _, ncols = A.shape
                true_support = tuple(sorted(rng.choice(ncols, size=s, replace=False).tolist()))
                x_true = np.zeros(ncols, dtype=float)
                vals = rng.integers(amp_low, amp_high + 1, size=s).astype(float)
                vals[vals == 0] = 1.0
                signs = rng.choice([-1.0, 1.0], size=s)
                x_true[list(true_support)] = vals * signs
                y = A @ x_true

                t2 = time.perf_counter()
                x_lp = recover_lp_l1(A, y)
                t3 = time.perf_counter()
                supp_lp = support(x_lp, s)
                lp_rows.append(
                    {
                        "succ_lp": float(supp_lp == true_support),
                        "relerr_lp": float(
                            np.linalg.norm(x_lp - x_true) / (np.linalg.norm(x_true) + 1e-12)
                        ),
                        "time_lp": (t3 - t2),
                    }
                )

        succ_lp = np.mean([r["succ_lp"] for r in lp_rows])
        rel_lp = np.mean([r["relerr_lp"] for r in lp_rows])
        t_lp = np.mean([r["time_lp"] for r in lp_rows])

        l0_succ_str = f"{succ_l0:7.2%}" if do_l0 else "  N/A  "
        l0_rel_str = f"{rel_l0:14.3e}" if do_l0 else "      N/A     "
        l0_time_str = f"{t_l0:11.4f}" if do_l0 else "    N/A    "
        print(
            f"{s:2d} | {l0_succ_str} | {succ_lp:7.2%} |"
            f" {l0_rel_str} | {rel_lp:14.3e} | {l0_time_str} | {t_lp:11.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Sparse recovery sweep S=3..10")
    parser.add_argument("mat_file", help="Path to .mat file")
    parser.add_argument("--s-min", type=int, default=3)
    parser.add_argument("--s-max", type=int, default=10)
    parser.add_argument("--trials", type=int, default=3, help="Random trials per sparsity")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--amp-low", type=int, default=1)
    parser.add_argument("--amp-high", type=int, default=9)
    args = parser.parse_args()

    data = sio.loadmat(args.mat_file)
    Af = _pick_key(data, "Af")
    Ar = _pick_key(data, "Ar")

    run_matrix_experiment(
        "Af", Af, args.s_min, args.s_max, args.trials, args.seed, args.amp_low, args.amp_high
    )
    run_matrix_experiment(
        "Ar", Ar, args.s_min, args.s_max, args.trials, args.seed + 1, args.amp_low, args.amp_high
    )


if __name__ == "__main__":
    main()
