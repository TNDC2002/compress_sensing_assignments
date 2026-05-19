#!/usr/bin/env python3
"""
COMP5340 Homework 6 (Spring 2026) — Issue 1

Sparse Representation Classification (SRC) on the cropped Extended Yale Face
Database B (96 x 84, 2414 images, 38 subjects).

Pipeline:
1. Random train/test split (stratified, default 50/50 per subject).
2. Build dictionary A from training faces only (columns = vectorized images).
3. For each test face y, solve an L1 sparse coding problem on A.
4. Classify by minimum class-wise reconstruction residual.

The lecture formulation min ||alpha||_1 s.t. A alpha = y is used when the
system is underdetermined. Here the pixel dimension (8064) exceeds the number
of training images (~1200), so y rarely lies exactly in col(A). We therefore
use the standard convex relaxation (basis pursuit denoising / LASSO):

    min ||alpha||_1 + (lambda / 2) ||y - A alpha||_2^2

which reduces to equality-constrained L1 minimization when a feasible exact
fit exists. Classification follows Wright et al. (PAMI 2009): predict the
class whose training atoms best reconstruct y.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from assignment_6_src_common import (
    build_dictionary,
    default_mat_path,
    load_yale_database,
    src_predict_one,
    stratified_train_test_split,
    vectorize_faces,
)


def src_classify(
    X: Array,
    test_idx: Array,
    test_labels: Array,
    dictionary: TrainingDictionary,
    lam: float,
    max_iter: int,
    tol: float,
    verbose_every: int = 0,
) -> Tuple[Array, float]:
    n_test = len(test_idx)
    predictions = np.empty(n_test, dtype=int)

    t0 = time.perf_counter()
    for j, img_idx in enumerate(test_idx):
        predictions[j] = src_predict_one(
            X[:, img_idx],
            dictionary,
            lam=lam,
            max_iter=max_iter,
            tol=tol,
        )
        if verbose_every > 0 and (j + 1) % verbose_every == 0:
            elapsed = time.perf_counter() - t0
            acc_so_far = 100.0 * np.mean(predictions[: j + 1] == test_labels[: j + 1])
            print(f"  processed {j + 1}/{n_test} ({acc_so_far:.2f}% so far, {elapsed:.1f}s)")

    elapsed = time.perf_counter() - t0
    return predictions, elapsed


def write_results_csv(
    path: Path,
    seed: int,
    train_ratio: float,
    lam: float,
    n_train: int,
    n_test: int,
    accuracy: float,
    elapsed_sec: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "seed",
                "train_ratio",
                "lambda",
                "n_train",
                "n_test",
                "accuracy_percent",
                "elapsed_sec",
            ]
        )
        writer.writerow(
            [
                seed,
                f"{train_ratio:.4f}",
                f"{lam:g}",
                n_train,
                n_test,
                f"{accuracy:.4f}",
                f"{elapsed_sec:.3f}",
            ]
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SRC face recognition on cropped Yale B database")
    parser.add_argument(
        "--mat-path",
        type=Path,
        default=default_mat_path(),
        help="Path to CroppedYale .mat file",
    )
    parser.add_argument("--train-ratio", type=float, default=0.5, help="Fraction of each class used for training")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for train/test split")
    parser.add_argument(
        "--lambda",
        dest="lam",
        type=float,
        default=0.01,
        help="LASSO regularization weight (L1 term)",
    )
    parser.add_argument("--max-iter", type=int, default=300, help="Maximum ISTA iterations per test image")
    parser.add_argument("--tol", type=float, default=1e-6, help="ISTA stopping tolerance")
    parser.add_argument(
        "--max-test",
        type=int,
        default=0,
        help="If > 0, classify only the first N test images (for quick checks)",
    )
    parser.add_argument(
        "--verbose-every",
        type=int,
        default=100,
        help="Print progress every N test images (0 to disable)",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=Path(__file__).resolve().parent / "src_results.csv",
        help="CSV file for summary metrics",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    rng = np.random.default_rng(args.seed)

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
    n_train, n_test = dictionary.A.shape[1], len(test_idx)

    print(f"Images: {db.faces.shape[0]}, pixels per face: {dictionary.A.shape[0]}")
    print(f"Training images: {n_train}, test images: {n_test}, classes: {len(np.unique(db.labels))}")
    print(f"SRC sparse coding: LASSO with lambda={args.lam:g}, ISTA max_iter={args.max_iter}")
    print("Classification rule: minimum class reconstruction residual")

    predictions, elapsed = src_classify(
        X,
        test_idx,
        test_labels,
        dictionary,
        lam=args.lam,
        max_iter=args.max_iter,
        tol=args.tol,
        verbose_every=args.verbose_every,
    )
    accuracy = 100.0 * float(np.mean(predictions == test_labels))

    print(f"\nClassification accuracy: {accuracy:.2f}%")
    print(f"Elapsed time: {elapsed:.1f} s ({elapsed / max(n_test, 1):.2f} s per test image)")
    write_results_csv(
        args.csv_out,
        args.seed,
        args.train_ratio,
        args.lam,
        n_train,
        n_test,
        accuracy,
        elapsed,
    )
    print(f"Wrote summary to {args.csv_out}")


if __name__ == "__main__":
    main()
