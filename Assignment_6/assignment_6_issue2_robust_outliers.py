#!/usr/bin/env python3
"""
COMP5340 Homework 6 — Issue 2

Evaluate standard SRC vs robust SRC under sparse outliers on test faces:
  - random pixel corruptions
  - random patch corruptions

Plots classification accuracy vs corruption percentage and writes CSV summaries.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from assignment_6_src_common import (
    TrainingDictionary,
    build_dictionary,
    corrupt_random_patches,
    corrupt_random_pixels,
    default_mat_path,
    load_yale_database,
    robust_src_predict_one,
    src_predict_one,
    stratified_train_test_split,
    vectorize_faces,
)

Array = np.ndarray


def parse_corruption_list(text: str) -> List[float]:
    values = [float(x.strip()) for x in text.split(",") if x.strip()]
    for v in values:
        if v < 0.0 or v > 1.0:
            raise ValueError(f"Corruption fraction must be in [0, 1], got {v}")
    return values


def evaluate_accuracy(
    X: Array,
    test_idx: Array,
    test_labels: Array,
    dictionary: TrainingDictionary,
    corruption_frac: float,
    corruption_kind: str,
    image_shape: Tuple[int, int],
    patch_size: Tuple[int, int],
    robust: bool,
    lam: float,
    lam_e: float,
    max_iter: int,
    tol: float,
    rng: np.random.Generator,
    verbose_every: int = 0,
) -> Tuple[float, float]:
    n_test = len(test_idx)
    correct = 0
    t0 = time.perf_counter()

    for j, img_idx in enumerate(test_idx):
        y = X[:, img_idx]
        if corruption_kind == "pixel":
            y_use = corrupt_random_pixels(y, corruption_frac, rng)
        elif corruption_kind == "patch":
            y_use = corrupt_random_patches(y, image_shape, corruption_frac, patch_size, rng)
        else:
            raise ValueError(f"Unknown corruption kind: {corruption_kind}")

        if robust:
            pred = robust_src_predict_one(
                y_use, dictionary, lam_alpha=lam, lam_e=lam_e, max_iter=max_iter, tol=tol
            )
        else:
            pred = src_predict_one(y_use, dictionary, lam=lam, max_iter=max_iter, tol=tol)

        if pred == test_labels[j]:
            correct += 1

        if verbose_every > 0 and (j + 1) % verbose_every == 0:
            elapsed = time.perf_counter() - t0
            print(
                f"    {corruption_kind} p={corruption_frac:.2f} "
                f"{'robust' if robust else 'standard':8s} "
                f"{j + 1}/{n_test} acc={100.0 * correct / (j + 1):.1f}% ({elapsed:.1f}s)"
            )

    elapsed = time.perf_counter() - t0
    return 100.0 * correct / max(n_test, 1), elapsed


def run_sweep(
    X: Array,
    test_idx: Array,
    test_labels: Array,
    dictionary: TrainingDictionary,
    corruption_fracs: Sequence[float],
    corruption_kinds: Sequence[str],
    image_shape: Tuple[int, int],
    patch_size: Tuple[int, int],
    lam: float,
    lam_e: float,
    max_iter: int,
    tol: float,
    seed: int,
    verbose_every: int,
) -> List[dict]:
    rows: List[dict] = []
    for kind in corruption_kinds:
        for frac in corruption_fracs:
            for robust in (False, True):
                rng = np.random.default_rng(seed + int(1e4 * frac) + (1 if kind == "patch" else 0) + (100 if robust else 0))
                label = "robust_src" if robust else "standard_src"
                print(f"Running {kind} corruption {frac:.0%} — {label}")
                acc, elapsed = evaluate_accuracy(
                    X,
                    test_idx,
                    test_labels,
                    dictionary,
                    corruption_frac=frac,
                    corruption_kind=kind,
                    image_shape=image_shape,
                    patch_size=patch_size,
                    robust=robust,
                    lam=lam,
                    lam_e=lam_e,
                    max_iter=max_iter,
                    tol=tol,
                    rng=rng,
                    verbose_every=verbose_every,
                )
                rows.append(
                    {
                        "corruption_kind": kind,
                        "corruption_frac": frac,
                        "method": label,
                        "accuracy_percent": acc,
                        "elapsed_sec": elapsed,
                    }
                )
                print(f"  -> {acc:.2f}% in {elapsed:.1f}s")
    return rows


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["corruption_kind", "corruption_frac", "method", "accuracy_percent", "elapsed_sec"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "corruption_kind": row["corruption_kind"],
                    "corruption_frac": f"{row['corruption_frac']:.4f}",
                    "method": row["method"],
                    "accuracy_percent": f"{row['accuracy_percent']:.4f}",
                    "elapsed_sec": f"{row['elapsed_sec']:.3f}",
                }
            )


def plot_accuracy_curves(rows: Sequence[dict], out_dir: Path) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for kind in sorted({r["corruption_kind"] for r in rows}):
        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        for method, style in (("standard_src", "o--"), ("robust_src", "s-")):
            sub = sorted(
                [r for r in rows if r["corruption_kind"] == kind and r["method"] == method],
                key=lambda r: r["corruption_frac"],
            )
            xs = [100.0 * r["corruption_frac"] for r in sub]
            ys = [r["accuracy_percent"] for r in sub]
            label = "Standard SRC" if method == "standard_src" else "Robust SRC"
            ax.plot(xs, ys, style, linewidth=2.0, markersize=7, label=label)

        ax.set_xlabel("Corruption percentage (%)")
        ax.set_ylabel("Classification accuracy (%)")
        title_kind = "Random pixel" if kind == "pixel" else "Random patch"
        ax.set_title(f"{title_kind} corruptions")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        out_path = out_dir / f"accuracy_vs_corruption_{kind}.png"
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        paths.append(out_path)
    return paths


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Robust SRC outlier corruption experiments")
    parser.add_argument("--mat-path", type=Path, default=default_mat_path())
    parser.add_argument("--train-ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lambda", dest="lam", type=float, default=0.01)
    parser.add_argument("--lambda-e", dest="lam_e", type=float, default=0.01, help="L1 weight on outlier term e")
    parser.add_argument("--max-iter", type=int, default=250)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument(
        "--corruption-fracs",
        type=str,
        default="0,0.05,0.1,0.15,0.2,0.3,0.4,0.5",
        help="Comma-separated corruption fractions in [0, 1]",
    )
    parser.add_argument("--patch-height", type=int, default=24)
    parser.add_argument("--patch-width", type=int, default=24)
    parser.add_argument("--max-test", type=int, default=0)
    parser.add_argument("--verbose-every", type=int, default=0)
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=Path(__file__).resolve().parent / "robust_outlier_results.csv",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    rng = np.random.default_rng(args.seed)
    corruption_fracs = parse_corruption_list(args.corruption_fracs)
    patch_size = (args.patch_height, args.patch_width)

    print(f"Loading database: {args.mat_path}")
    db = load_yale_database(args.mat_path)
    X = vectorize_faces(db.faces)
    split = stratified_train_test_split(db.labels, args.train_ratio, rng)

    train_labels = db.labels[split.train_idx]
    test_labels = db.labels[split.test_idx]
    test_idx = split.test_idx
    if args.max_test > 0:
        test_idx = test_idx[: args.max_test]
        test_labels = test_labels[: args.max_test]

    dictionary = build_dictionary(X, split.train_idx, train_labels)
    print(
        f"Train={dictionary.A.shape[1]}, test={len(test_idx)}, "
        f"corruption levels={corruption_fracs}, patch={patch_size}"
    )

    rows = run_sweep(
        X,
        test_idx,
        test_labels,
        dictionary,
        corruption_fracs=corruption_fracs,
        corruption_kinds=("pixel", "patch"),
        image_shape=db.image_shape,
        patch_size=patch_size,
        lam=args.lam,
        lam_e=args.lam_e,
        max_iter=args.max_iter,
        tol=args.tol,
        seed=args.seed,
        verbose_every=args.verbose_every,
    )

    write_csv(args.csv_out, rows)
    plot_paths = plot_accuracy_curves(rows, args.plot_dir)
    print(f"\nWrote CSV: {args.csv_out}")
    for p in plot_paths:
        print(f"Wrote plot: {p}")


if __name__ == "__main__":
    main()
