#!/usr/bin/env python3
"""
COMP5340 Homework 3 (Spring 2026)

Sparse recovery with multiple sensing matrices for:
1) Time-sparse signals
2) Frequency-sparse signals
3) Increased sparsity level

Outputs:
- CSV tables with recovery probabilities
- PNG performance plots
- Markdown summary report
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import dct, idct
from scipy.optimize import linprog

Array = np.ndarray


@dataclass(frozen=True)
class ExperimentConfig:
    n: int
    s: int
    m_values: List[int]
    trials: int
    seed: int
    threshold: float
    max_iter: int
    tol: float


def dct_matrix(n: int) -> Array:
    # Match homework hint: DCT applied along columns (axis=0).
    return dct(np.eye(n), type=2, norm="ortho", axis=0)


def idct_matrix(n: int) -> Array:
    return idct(np.eye(n), type=2, norm="ortho", axis=0)


def recover_bp(A: Array, y: Array) -> Array:
    """Basis pursuit: min ||x||_1 s.t. A x = y."""
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


def recover_omp(A: Array, y: Array, sparsity: int, max_iter: int, tol: float) -> Array:
    """OMP for ||x||_0 <= sparsity."""
    _, n = A.shape
    support: List[int] = []
    x = np.zeros(n, dtype=float)
    r = y.copy()

    for _ in range(min(max_iter, sparsity)):
        corr = A.T @ r
        j = int(np.argmax(np.abs(corr)))
        if j not in support:
            support.append(j)

        As = A[:, support]
        xs, *_ = np.linalg.lstsq(As, y, rcond=None)
        r = y - As @ xs
        if np.linalg.norm(r) <= tol:
            break

    if support:
        x[support] = xs
    return x


def random_row_subsample(mat: Array, m: int, rng: np.random.Generator) -> Array:
    n = mat.shape[0]
    idx = rng.choice(n, size=m, replace=False)
    return mat[idx, :]


def uniform_indices(n: int, m: int) -> Array:
    # Following assignment style: floor(k * N / M), with 1-based in prompt.
    # Converted to 0-based indexing and clipped to valid range.
    idx = np.floor(np.arange(m) * n / m).astype(int)
    idx = np.clip(idx, 0, n - 1)
    return np.unique(idx)[:m] if len(np.unique(idx)) >= m else idx


def build_sensing_matrices(n: int, m: int, rng: np.random.Generator) -> Dict[str, Array]:
    I = np.eye(n)
    F = dct_matrix(n)
    idx_uni = uniform_indices(n, m)

    A_time_random = random_row_subsample(I, m, rng)
    A_time_uniform = I[idx_uni, :]
    A_freq_random = random_row_subsample(F, m, rng)
    A_freq_low = F[:m, :]
    A_freq_equispaced = F[idx_uni, :]

    # Random Gaussian + row orthonormalization.
    G = rng.standard_normal((m, n))
    Q, _ = np.linalg.qr(G.T)  # Q shape: (n, m), orthonormal columns
    A_random_domain = Q.T  # shape: (m, n), orthonormal rows

    return {
        "time_random": A_time_random,
        "time_uniform": A_time_uniform,
        "freq_random": A_freq_random,
        "freq_low": A_freq_low,
        "freq_equispaced": A_freq_equispaced,
        "random_domain": A_random_domain,
    }


def generate_time_sparse(n: int, s: int, rng: np.random.Generator) -> Array:
    x = np.zeros(n, dtype=float)
    supp = rng.choice(n, size=s, replace=False)
    x[supp] = rng.standard_normal(s)
    return x


def generate_freq_sparse(n: int, s: int, rng: np.random.Generator) -> Tuple[Array, Array]:
    alpha = np.zeros(n, dtype=float)
    supp = rng.choice(n, size=s, replace=False)
    alpha[supp] = rng.standard_normal(s)
    x = idct(alpha, type=2, norm="ortho", axis=0)
    return x, alpha


def run_single_setting(
    cfg: ExperimentConfig,
    signal_kind: str,
    output_dir: Path,
) -> Dict[str, Dict[str, List[float]]]:
    rng = np.random.default_rng(cfg.seed)
    psi = idct_matrix(cfg.n)  # x = psi @ alpha

    methods: Dict[str, Callable[..., Array]] = {
        "BP": recover_bp,
        "OMP": recover_omp,
    }
    scheme_names = [
        "time_random",
        "time_uniform",
        "freq_random",
        "freq_low",
        "freq_equispaced",
        "random_domain",
    ]
    probs = {method: {scheme: [] for scheme in scheme_names} for method in methods}

    for m in cfg.m_values:
        success = {method: {scheme: 0 for scheme in scheme_names} for method in methods}
        for _ in range(cfg.trials):
            matrices = build_sensing_matrices(cfg.n, m, rng)
            if signal_kind == "time_sparse":
                x_true = generate_time_sparse(cfg.n, cfg.s, rng)
                alpha_true = None
            else:
                x_true, alpha_true = generate_freq_sparse(cfg.n, cfg.s, rng)

            for scheme, A in matrices.items():
                y = A @ x_true
                if signal_kind == "time_sparse":
                    x_bp = recover_bp(A, y)
                    x_omp = recover_omp(A, y, cfg.s, cfg.max_iter, cfg.tol)
                    if np.linalg.norm(x_bp - x_true) <= cfg.threshold:
                        success["BP"][scheme] += 1
                    if np.linalg.norm(x_omp - x_true) <= cfg.threshold:
                        success["OMP"][scheme] += 1
                else:
                    # Frequency sparse: recover alpha in y = (A @ psi) alpha
                    B = A @ psi
                    alpha_bp = recover_bp(B, y)
                    alpha_omp = recover_omp(B, y, cfg.s, cfg.max_iter, cfg.tol)
                    if np.linalg.norm(alpha_bp - alpha_true) <= cfg.threshold:
                        success["BP"][scheme] += 1
                    if np.linalg.norm(alpha_omp - alpha_true) <= cfg.threshold:
                        success["OMP"][scheme] += 1

        for method in methods:
            for scheme in scheme_names:
                probs[method][scheme].append(success[method][scheme] / cfg.trials)

        print(f"{signal_kind}, S={cfg.s}, M={m}: done")

    save_probability_csv(output_dir / f"{signal_kind}_S{cfg.s}_probabilities.csv", cfg.m_values, probs)
    save_plots(output_dir, signal_kind, cfg.s, cfg.m_values, probs)
    return probs


def save_probability_csv(path: Path, m_values: List[int], probs: Dict[str, Dict[str, List[float]]]) -> None:
    schemes = list(next(iter(probs.values())).keys())
    methods = list(probs.keys())
    header = ["M"] + [f"{method}_{scheme}" for method in methods for scheme in schemes]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, m in enumerate(m_values):
            row = [m]
            for method in methods:
                for scheme in schemes:
                    row.append(probs[method][scheme][i])
            writer.writerow(row)


def save_plots(
    output_dir: Path,
    signal_kind: str,
    sparsity: int,
    m_values: List[int],
    probs: Dict[str, Dict[str, List[float]]],
) -> None:
    for method, by_scheme in probs.items():
        plt.figure(figsize=(9, 6))
        for scheme, ys in by_scheme.items():
            plt.plot(m_values, ys, marker="o", linewidth=1.5, markersize=4, label=scheme)
        plt.xlabel("Number of measurements M")
        plt.ylabel("Probability of perfect recovery")
        plt.title(f"{signal_kind} (S={sparsity}) - {method}")
        plt.ylim(-0.02, 1.02)
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        out_path = output_dir / f"{signal_kind}_S{sparsity}_{method}_curve.png"
        plt.savefig(out_path, dpi=180)
        plt.close()


def first_m_reaching_target(m_values: List[int], ys: List[float], target: float = 1.0) -> int | None:
    for m, y in zip(m_values, ys):
        if y >= target:
            return m
    return None


def pick_best_scheme(probs: Dict[str, List[float]]) -> str:
    # Highest average probability across all M.
    return max(probs.keys(), key=lambda k: float(np.mean(probs[k])))


def write_summary_report(
    path: Path,
    m_values: List[int],
    time_s5: Dict[str, Dict[str, List[float]]],
    freq_s5: Dict[str, Dict[str, List[float]]],
    time_s25: Dict[str, Dict[str, List[float]]],
) -> None:
    lines: List[str] = []
    lines.append("# Assignment 3 Report")
    lines.append("")
    lines.append("## Setup")
    lines.append("- N = 256")
    lines.append("- M in {10, 20, ..., 100}")
    lines.append("- 100 trials per M")
    lines.append("- Perfect recovery if L2 error <= 1e-6")
    lines.append("- Recovery methods: BP (L1 minimization), OMP (greedy)")
    lines.append("")

    def section(title: str, data: Dict[str, Dict[str, List[float]]]) -> None:
        lines.append(f"## {title}")
        for method, method_data in data.items():
            best = pick_best_scheme(method_data)
            m_full = first_m_reaching_target(m_values, method_data[best], target=1.0)
            lines.append(f"- {method}: best sensing scheme = `{best}`")
            if m_full is None:
                lines.append("-  Earliest M with probability 1.0: not reached up to M=100")
            else:
                lines.append(f"-  Earliest M with probability 1.0: M={m_full}")
        lines.append("")

    section("Q1: Time-sparse signal (S=5)", time_s5)
    section("Q2: Frequency-sparse signal (S=5)", freq_s5)
    section("Q3: Time-sparse signal with higher sparsity (S=25)", time_s25)

    lines.append("## Frequency-sparse L1 formulation")
    lines.append(
        "- For x = Psi * alpha where alpha is sparse in DCT domain, solve min ||alpha||_1 subject to y = A*Psi*alpha."
    )
    lines.append("- Then recover x_hat = Psi * alpha_hat.")
    lines.append("")

    lines.append("## Method efficiency")
    lines.append("- OMP is typically much faster than BP in this setup.")
    lines.append("- BP usually provides stronger recovery robustness when measurements are limited.")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Assignment 3 compressive sensing experiments")
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--m-start", type=int, default=10)
    parser.add_argument("--m-stop", type=int, default=100)
    parser.add_argument("--m-step", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=1e-6)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--tol", type=float, default=1e-9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    m_values = list(range(args.m_start, args.m_stop + 1, args.m_step))

    cfg_s5 = ExperimentConfig(
        n=args.n,
        s=5,
        m_values=m_values,
        trials=args.trials,
        seed=args.seed,
        threshold=args.threshold,
        max_iter=args.max_iter,
        tol=args.tol,
    )
    cfg_s25 = ExperimentConfig(
        n=args.n,
        s=25,
        m_values=m_values,
        trials=args.trials,
        seed=args.seed + 101,
        threshold=args.threshold,
        max_iter=args.max_iter,
        tol=args.tol,
    )

    print("Running Q1: time-sparse, S=5")
    time_s5 = run_single_setting(cfg_s5, "time_sparse", output_dir)
    print("Running Q2: frequency-sparse, S=5")
    freq_s5 = run_single_setting(cfg_s5, "frequency_sparse", output_dir)
    print("Running Q3: time-sparse, S=25")
    time_s25 = run_single_setting(cfg_s25, "time_sparse", output_dir)

    write_summary_report(
        output_dir / "assignment_3_report.md",
        m_values,
        time_s5=time_s5,
        freq_s5=freq_s5,
        time_s25=time_s25,
    )
    print(f"Done. Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
