#!/usr/bin/env python3
"""
Assignment 2: Sparse recovery probability curve on synthetic data.

Signal model:
    N = 256, S = 5 sparse signal with random support and amplitudes.

Measurement model:
    y = A x, where A is random Gaussian with shape (M, N).

For each M in {5, 10, ..., 100}:
    - Generate `trials` random instances of (x, A).
    - Recover with L1 minimization and 4 greedy methods: MP, OMP, SP, CoSaMP.
    - Mark perfect recovery if ||x_hat - x||_2 <= 1e-6.
    - Compute success probability for each method.

Outputs:
    - CSV table of probabilities
    - PNG performance curve (probability vs M)
"""

from __future__ import annotations

import argparse
import csv
import time
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog


Array = np.ndarray
RecoverFn = Callable[[Array, Array, int, int, float], Array]


def normalize_columns(A: Array) -> Tuple[Array, Array]:
    norms = np.linalg.norm(A, axis=0)
    norms[norms < 1e-12] = 1.0
    return A / norms, norms


def recover_l1_bp(A: Array, y: Array, *_unused: object) -> Array:
    """Basis pursuit via LP: min ||x||_1 s.t. A x = y."""
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
        return np.zeros(n, dtype=float)
    return result.x[:n]


def recover_mp(A: Array, y: Array, sparsity: int, max_iter: int, tol: float) -> Array:
    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    A_n, norms = normalize_columns(A)
    _, n = A.shape
    x_n = np.zeros(n, dtype=float)
    r = y.copy()

    for _ in range(max_iter):
        corr = A_n.T @ r
        j = int(np.argmax(np.abs(corr)))
        x_n[j] += corr[j]
        r = y - A_n @ x_n
        if np.linalg.norm(r) <= tol or np.count_nonzero(np.abs(x_n) > 1e-12) >= sparsity:
            break
    return x_n / norms


def recover_omp(A: Array, y: Array, sparsity: int, max_iter: int, tol: float) -> Array:
    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    _, n = A.shape
    supp: List[int] = []
    x = np.zeros(n, dtype=float)
    r = y.copy()

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

        merge = set(supp_list) | set(np.argsort(np.abs(A.T @ r))[-sparsity:].tolist())
        merge_list = sorted(merge)
        Amerge = A[:, merge_list]
        b, *_ = np.linalg.lstsq(Amerge, y, rcond=None)
        keep = np.argsort(np.abs(b))[-sparsity:]
        new_supp = {merge_list[i] for i in keep}
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
    x = np.zeros(n, dtype=float)
    r = y.copy()
    supp: set[int] = set()

    for _ in range(max_iter):
        proxy = A.T @ r
        omega = set(np.argsort(np.abs(proxy))[-2 * sparsity :].tolist())
        merge = sorted(supp | omega)
        Amerge = A[:, merge]
        b, *_ = np.linalg.lstsq(Amerge, y, rcond=None)
        keep = np.argsort(np.abs(b))[-sparsity:]
        supp = {merge[i] for i in keep}

        x[:] = 0.0
        supp_list = sorted(supp)
        As = A[:, supp_list]
        xs, *_ = np.linalg.lstsq(As, y, rcond=None)
        x[supp_list] = xs
        r = y - A @ x
        if np.linalg.norm(r) <= tol:
            break
    return x


def generate_sparse_signal(n: int, s: int, rng: np.random.Generator) -> Array:
    x = np.zeros(n, dtype=float)
    q = rng.permutation(n)
    support = q[:s]
    x[support] = rng.standard_normal(s)
    return x


def run_experiment(
    n: int,
    s: int,
    m_values: List[int],
    trials: int,
    threshold: float,
    max_iter: int,
    tol: float,
    seed: int,
) -> Tuple[Dict[str, List[float]], Dict[str, float]]:
    methods: Dict[str, RecoverFn] = {
        "L1": recover_l1_bp,
        "MP": recover_mp,
        "OMP": recover_omp,
        "SP": recover_sp,
        "CoSaMP": recover_cosamp,
    }
    probs = {name: [] for name in methods}
    avg_time_ms = {name: 0.0 for name in methods}
    call_count = 0

    rng = np.random.default_rng(seed)

    for M in m_values:
        success = {name: 0 for name in methods}
        for _ in range(trials):
            x_true = generate_sparse_signal(n, s, rng)
            A = rng.standard_normal((M, n))
            y = A @ x_true

            for name, fn in methods.items():
                t0 = time.perf_counter()
                x_hat = fn(A, y, s, max_iter, tol)
                t1 = time.perf_counter()
                err = np.linalg.norm(x_hat - x_true)
                if err <= threshold:
                    success[name] += 1
                avg_time_ms[name] += 1000.0 * (t1 - t0)
            call_count += 1

        for name in methods:
            probs[name].append(success[name] / trials)

        row = " | ".join([f"{name}: {probs[name][-1]:.2f}" for name in methods])
        print(f"M={M:3d} -> {row}")

    for name in methods:
        avg_time_ms[name] /= max(1, call_count)
    return probs, avg_time_ms


def write_csv(path: str, m_values: List[int], probs: Dict[str, List[float]]) -> None:
    names = list(probs.keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["M"] + names)
        for i, m in enumerate(m_values):
            writer.writerow([m] + [probs[name][i] for name in names])


def plot_curve(path: str, m_values: List[int], probs: Dict[str, List[float]]) -> None:
    plt.figure(figsize=(9, 6))
    for name, ys in probs.items():
        plt.plot(m_values, ys, marker="o", linewidth=1.8, markersize=4, label=name)
    plt.xlabel("Number of measurements M")
    plt.ylabel("Probability of perfect recovery")
    plt.title("Sparse Recovery Performance (N=256, S=5)")
    plt.ylim(-0.02, 1.02)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic sparse recovery probability curve")
    parser.add_argument("--n", type=int, default=256, help="Signal length N")
    parser.add_argument("--s", type=int, default=5, help="Sparsity S")
    parser.add_argument("--m-start", type=int, default=5)
    parser.add_argument("--m-stop", type=int, default=100)
    parser.add_argument("--m-step", type=int, default=5)
    parser.add_argument("--trials", type=int, default=100, help="Instances per M")
    parser.add_argument("--threshold", type=float, default=1e-6, help="Perfect recovery threshold")
    parser.add_argument("--max-iter", type=int, default=50, help="Max greedy iterations")
    parser.add_argument("--tol", type=float, default=1e-9, help="Residual tolerance for greedy methods")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--csv-out",
        default="synthetic_recovery_probability_results.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--plot-out",
        default="synthetic_recovery_probability_curve.png",
        help="Output plot path",
    )
    args = parser.parse_args()

    m_values = list(range(args.m_start, args.m_stop + 1, args.m_step))
    probs, avg_time_ms = run_experiment(
        n=args.n,
        s=args.s,
        m_values=m_values,
        trials=args.trials,
        threshold=args.threshold,
        max_iter=args.max_iter,
        tol=args.tol,
        seed=args.seed,
    )
    write_csv(args.csv_out, m_values, probs)
    plot_curve(args.plot_out, m_values, probs)

    print("\nAverage runtime per solve (ms):")
    for name, t in avg_time_ms.items():
        print(f"- {name:6s}: {t:10.3f}")
    print(f"\nSaved CSV:  {args.csv_out}")
    print(f"Saved plot: {args.plot_out}")


if __name__ == "__main__":
    main()

