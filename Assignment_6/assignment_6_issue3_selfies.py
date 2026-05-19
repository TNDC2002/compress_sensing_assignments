#!/usr/bin/env python3
"""
COMP5340 Homework 6 — Issue 3 (full pipeline)

1. Train SRC on Yale only (me NOT in gallery). Test on my selfies — can it recognize me?
2. Develop / calibrate outlier-rejection gate; demonstrate on my selfies (unknown).
3. Add my selfies to the training gallery (new class 39); rebuild dictionary and train again.
4. Test again: my held-out selfies should be recognized; friend's selfies should still be rejected.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from assignment_6_src_common import (
    OutlierThresholds,
    SrcDecision,
    TrainingDictionary,
    build_dictionary,
    build_dictionary_from_matrix,
    calibrate_outlier_thresholds,
    default_mat_path,
    is_outlier,
    load_yale_database,
    preprocess_selfie_to_yale,
    src_decision_one,
    stratified_train_test_split,
    vectorize_faces,
)

Array = np.ndarray
MY_CLASS_ID = 39


@dataclass(frozen=True)
class SelfieChip:
    name: str
    chip: Array  # (H, W)


@dataclass(frozen=True)
class ProbeResult:
    phase: str
    name: str
    probe_kind: str
    in_gallery: bool
    true_class: int
    predicted_class: int
    min_residual: float
    residual_ratio: float
    sparsity_concentration: float
    rejected: bool
    correct_label: bool


def save_preprocessed_grid(chips: Sequence[Tuple[str, Array]], out_path: Path) -> None:
    if not chips:
        return
    tile_h, tile_w = chips[0][1].shape
    cols = min(4, len(chips))
    rows = int(np.ceil(len(chips) / cols))
    margin = 8
    canvas = Image.new("L", (cols * tile_w + (cols + 1) * margin, rows * tile_h + (rows + 1) * margin), 255)
    for i, (_, chip) in enumerate(chips):
        r, c = divmod(i, cols)
        tile = Image.fromarray(np.clip(chip, 0, 255).astype(np.uint8))
        x = margin + c * (tile_w + margin)
        y = margin + r * (tile_h + margin)
        canvas.paste(tile, (x, y))
    canvas.save(out_path)


def save_chip_png(chip: Array, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(chip, 0, 255).astype(np.uint8)).save(path)


def load_selfie_chips(selfie_dir: Path, yale_mean: float, yale_std: float) -> List[SelfieChip]:
    chips: List[SelfieChip] = []
    if not selfie_dir.is_dir():
        return chips
    for path in sorted(selfie_dir.glob("*")):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        chip = preprocess_selfie_to_yale(path, yale_mean=yale_mean, yale_std=yale_std)
        chips.append(SelfieChip(name=path.name, chip=chip))
    return chips


def split_selfies(
    selfies: Sequence[SelfieChip],
    train_ratio: float,
    rng: np.random.Generator,
) -> Tuple[List[SelfieChip], List[SelfieChip]]:
    idx = np.arange(len(selfies))
    rng.shuffle(idx)
    n_train = int(round(train_ratio * len(selfies)))
    n_train = min(max(n_train, 1), len(selfies) - 1)
    train = [selfies[i] for i in idx[:n_train]]
    test = [selfies[i] for i in idx[n_train:]]
    return train, test


def chips_to_columns(chips: Sequence[SelfieChip]) -> Array:
    return np.column_stack([s.chip.reshape(-1) for s in chips])


def build_extended_dictionary(
    X_yale: Array,
    yale_train_idx: Array,
    yale_train_labels: Array,
    extra_train: Sequence[SelfieChip],
    extra_class_id: int,
) -> TrainingDictionary:
    X_yale_train = X_yale[:, yale_train_idx]
    if extra_train:
        X_extra = chips_to_columns(extra_train)
        labels_extra = np.full(len(extra_train), extra_class_id, dtype=int)
        X_all = np.hstack([X_yale_train, X_extra])
        labels_all = np.concatenate([yale_train_labels, labels_extra])
    else:
        X_all = X_yale_train
        labels_all = yale_train_labels
    return build_dictionary_from_matrix(X_all, labels_all)


def evaluate_probe(
    phase: str,
    name: str,
    probe_kind: str,
    chip: Array,
    dictionary: TrainingDictionary,
    thresholds: OutlierThresholds,
    lam: float,
    max_iter: int,
    tol: float,
    in_gallery: bool,
    true_class: int,
) -> ProbeResult:
    decision = src_decision_one(chip.reshape(-1), dictionary, lam, max_iter, tol)
    rejected = is_outlier(decision, thresholds)
    correct_label = (not in_gallery and rejected) or (
        in_gallery and (not rejected) and decision.predicted_class == true_class
    )
    return ProbeResult(
        phase=phase,
        name=name,
        probe_kind=probe_kind,
        in_gallery=in_gallery,
        true_class=true_class,
        predicted_class=decision.predicted_class,
        min_residual=decision.min_residual,
        residual_ratio=decision.residual_ratio,
        sparsity_concentration=decision.sparsity_concentration,
        rejected=rejected,
        correct_label=correct_label,
    )


def calibrate_thresholds_on_inliers(
    X: Array,
    inlier_idx: Array,
    dictionary: TrainingDictionary,
    lam: float,
    max_iter: int,
    tol: float,
    max_calib: int,
) -> OutlierThresholds:
    idx = inlier_idx[:max_calib] if max_calib > 0 else inlier_idx
    decisions = [src_decision_one(X[:, i], dictionary, lam, max_iter, tol) for i in idx]
    return calibrate_outlier_thresholds(decisions, residual_quantile=0.95, total_quantile=0.95)


def print_probe_table(results: Sequence[ProbeResult]) -> None:
    print(f"{'phase':<12} {'name':<26} {'kind':<14} {'pred':>5} {'r1':>7} {'rej':>5} {'ok':>4}")
    for r in results:
        short = r.name if len(r.name) <= 26 else r.name[:23] + "..."
        print(
            f"{r.phase:<12} {short:<26} {r.probe_kind:<14} {r.predicted_class:>5} "
            f"{r.min_residual:7.4f} {int(r.rejected):>5} {int(r.correct_label):>4}"
        )


def write_results_csv(path: Path, results: Sequence[ProbeResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "phase",
                "name",
                "probe_kind",
                "in_gallery",
                "true_class",
                "predicted_class",
                "min_residual",
                "residual_ratio",
                "sparsity_concentration",
                "rejected",
                "correct_label",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "phase": r.phase,
                    "name": r.name,
                    "probe_kind": r.probe_kind,
                    "in_gallery": int(r.in_gallery),
                    "true_class": r.true_class,
                    "predicted_class": r.predicted_class,
                    "min_residual": f"{r.min_residual:.6f}",
                    "residual_ratio": f"{r.residual_ratio:.6f}",
                    "sparsity_concentration": f"{r.sparsity_concentration:.6f}",
                    "rejected": int(r.rejected),
                    "correct_label": int(r.correct_label),
                }
            )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Issue 3 full pipeline")
    parser.add_argument("--mat-path", type=Path, default=default_mat_path())
    parser.add_argument("--selfie-dir", type=Path, default=Path(__file__).resolve().parent / "selfy")
    parser.add_argument(
        "--friend-selfie-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "friend_selfies",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--yale-train-ratio", type=float, default=0.5)
    parser.add_argument("--my-train-ratio", type=float, default=0.625, help="Fraction of my selfies for training in phase 3")
    parser.add_argument("--lambda", dest="lam", type=float, default=0.01)
    parser.add_argument("--max-iter", type=int, default=250)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--calibration-max", type=int, default=200)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    out_dir = args.output_dir
    rng = np.random.default_rng(args.seed)

    db = load_yale_database(args.mat_path)
    yale_mean = float(db.faces.mean())
    yale_std = float(db.faces.std())
    X = vectorize_faces(db.faces)
    yale_split = stratified_train_test_split(db.labels, args.yale_train_ratio, rng)
    yale_train_labels = db.labels[yale_split.train_idx]

    print("Loading and preprocessing selfies...")
    my_selfies = load_selfie_chips(args.selfie_dir, yale_mean, yale_std)
    friend_selfies = load_selfie_chips(args.friend_selfie_dir, yale_mean, yale_std)
    if not my_selfies:
        raise FileNotFoundError(f"No images in {args.selfie_dir}")
    if not friend_selfies:
        raise FileNotFoundError(f"No images in {args.friend_selfie_dir}")

    my_train, my_test = split_selfies(my_selfies, args.my_train_ratio, rng)
    print(f"My selfies: {len(my_train)} train, {len(my_test)} test | Friend: {len(friend_selfies)} test only")

    pre_my = out_dir / "preprocessed_selfies"
    pre_friend = out_dir / "preprocessed_friend_selfies"
    for s in my_selfies:
        save_chip_png(s.chip, pre_my / f"{Path(s.name).stem}_yale.png")
    for s in friend_selfies:
        save_chip_png(s.chip, pre_friend / f"{Path(s.name).stem}_yale.png")
    save_preprocessed_grid([(s.name, s.chip) for s in my_selfies], out_dir / "preprocessed_selfies_grid.png")
    save_preprocessed_grid([(s.name, s.chip) for s in friend_selfies], out_dir / "preprocessed_friend_selfies_grid.png")

    dict_yale = build_dictionary(X, yale_split.train_idx, yale_train_labels)
    all_results: List[ProbeResult] = []
    no_gate = OutlierThresholds(999.0, 0.0, 999.0, 0.0)  # phase 1: SRC label only, no rejection

    # ------------------------------------------------------------------ Phase 1
    print("\n=== Phase 1: Yale-only gallery (me NOT in database) — recognition test ===")
    phase = "phase1"
    for s in my_selfies:
        r = evaluate_probe(
            phase, s.name, "my_selfie", s.chip, dict_yale,
            no_gate,
            args.lam, args.max_iter, args.tol,
            in_gallery=False, true_class=-1,
        )
        all_results.append(r)
    n_wrong_label = sum(r.predicted_class != MY_CLASS_ID for r in all_results if r.phase == phase)
    print(f"Me in gallery? No. SRC assigns Yale IDs (never class {MY_CLASS_ID}).")
    print(f"All {len(my_selfies)} probes get a forced Yale label (misrecognition if interpreted as 'me').")

    # ------------------------------------------------------------------ Phase 2
    print("\n=== Phase 2: Outlier-rejection gate (calibrated on Yale in-gallery test) ===")
    phase = "phase2"
    thresholds = calibrate_thresholds_on_inliers(
        X, yale_split.test_idx, dict_yale, args.lam, args.max_iter, args.tol, args.calibration_max,
    )
    print(
        f"Thresholds: r1<={thresholds.max_min_residual:.4f}, r2/r1>={thresholds.min_residual_ratio:.3f}, "
        f"total<={thresholds.max_total_residual:.4f}, conc>={thresholds.min_sparsity_concentration:.3f}"
    )
    for s in my_selfies:
        r = evaluate_probe(
            phase, s.name, "my_selfie", s.chip, dict_yale, thresholds,
            args.lam, args.max_iter, args.tol, in_gallery=False, true_class=-1,
        )
        all_results.append(r)
    for s in friend_selfies:
        r = evaluate_probe(
            phase, s.name, "friend_selfie", s.chip, dict_yale, thresholds,
            args.lam, args.max_iter, args.tol, in_gallery=False, true_class=-1,
        )
        all_results.append(r)
    p2_my = [r for r in all_results if r.phase == phase and r.probe_kind == "my_selfie"]
    p2_fr = [r for r in all_results if r.phase == phase and r.probe_kind == "friend_selfie"]
    print(f"My selfies rejected: {sum(r.rejected for r in p2_my)}/{len(p2_my)}")
    print(f"Friend selfies rejected: {sum(r.rejected for r in p2_fr)}/{len(p2_fr)}")

    # ------------------------------------------------------------------ Phase 3
    print("\n=== Phase 3: Extended gallery — add my training selfies as class 39; rebuild dictionary ===")
    dict_extended = build_extended_dictionary(
        X, yale_split.train_idx, yale_train_labels, my_train, MY_CLASS_ID,
    )
    print(
        f"Dictionary size: {dict_extended.A.shape[1]} atoms "
        f"({len(yale_split.train_idx)} Yale + {len(my_train)} mine), classes include {MY_CLASS_ID}"
    )

    # ------------------------------------------------------------------ Phase 4
    print("\n=== Phase 4: Test after I am in gallery ===")
    phase = "phase4"
    # Calibrate gate on Yale test + my test (both in-gallery)
    my_test_idx_for_calib = []  # use decisions directly
    calib_decisions = [
        src_decision_one(X[:, i], dict_extended, args.lam, args.max_iter, args.tol)
        for i in yale_split.test_idx[: args.calibration_max]
    ]
    calib_decisions += [
        src_decision_one(s.chip.reshape(-1), dict_extended, args.lam, args.max_iter, args.tol)
        for s in my_test
    ]
    thresholds_ext = calibrate_outlier_thresholds(
        calib_decisions, residual_quantile=0.95, total_quantile=0.95,
    )
    print(
        f"Thresholds (Yale test + my test): r1<={thresholds_ext.max_min_residual:.4f}, "
        f"r2/r1>={thresholds_ext.min_residual_ratio:.3f}"
    )

    for s in my_test:
        r = evaluate_probe(
            phase, s.name, "my_test", s.chip, dict_extended, thresholds_ext,
            args.lam, args.max_iter, args.tol, in_gallery=True, true_class=MY_CLASS_ID,
        )
        all_results.append(r)
    for s in my_train:
        r = evaluate_probe(
            phase, s.name, "my_train", s.chip, dict_extended, thresholds_ext,
            args.lam, args.max_iter, args.tol, in_gallery=True, true_class=MY_CLASS_ID,
        )
        all_results.append(r)
    for s in friend_selfies:
        r = evaluate_probe(
            phase, s.name, "friend_selfie", s.chip, dict_extended, thresholds_ext,
            args.lam, args.max_iter, args.tol, in_gallery=False, true_class=-1,
        )
        all_results.append(r)

    p4_my_test = [r for r in all_results if r.phase == phase and r.probe_kind == "my_test"]
    p4_friend = [r for r in all_results if r.phase == phase and r.probe_kind == "friend_selfie"]
    print(f"My test selfies: recognized class {MY_CLASS_ID}? "
          f"{sum(r.predicted_class == MY_CLASS_ID for r in p4_my_test)}/{len(p4_my_test)}, "
          f"accepted (not rejected)? {sum(not r.rejected for r in p4_my_test)}/{len(p4_my_test)}")
    print(f"Friend selfies still rejected? {sum(r.rejected for r in p4_friend)}/{len(p4_friend)}")

    print("\n=== Full probe table ===")
    print_probe_table(all_results)

    csv_path = out_dir / "issue3_pipeline_results.csv"
    write_results_csv(csv_path, all_results)

    summary_path = out_dir / "issue3_pipeline_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["my_class_id", MY_CLASS_ID])
        w.writerow(["my_train_count", len(my_train)])
        w.writerow(["my_test_count", len(my_test)])
        w.writerow(["friend_test_count", len(friend_selfies)])
        w.writerow(["phase2_my_rejected", sum(r.rejected for r in p2_my)])
        w.writerow(["phase2_friend_rejected", sum(r.rejected for r in p2_fr)])
        w.writerow(
            ["phase4_my_test_correct_class", sum(r.predicted_class == MY_CLASS_ID for r in p4_my_test)],
        )
        w.writerow(["phase4_my_test_accepted", sum(not r.rejected for r in p4_my_test)])
        w.writerow(["phase4_friend_rejected", sum(r.rejected for r in p4_friend)])

    print(f"\nWrote {csv_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
