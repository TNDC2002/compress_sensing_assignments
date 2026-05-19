#!/usr/bin/env python3
"""
COMP5340 Homework 6 — Issue 3

Preprocess selfies to Yale B format, test SRC recognition, and demonstrate
outlier rejection on an extended database (Yale + selfies).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from assignment_6_src_common import (
    OutlierThresholds,
    SrcDecision,
    build_dictionary,
    calibrate_outlier_thresholds,
    default_mat_path,
    is_outlier,
    load_yale_database,
    preprocess_selfie_to_yale,
    src_decision_one,
    src_predict_one,
    stratified_train_test_split,
    vectorize_faces,
)

Array = np.ndarray
UNKNOWN_LABEL = 0
SELFIE_LABEL = 39


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


def load_preprocessed_selfies(
    selfie_dir: Path,
    yale_mean: float,
    yale_std: float,
) -> List[Tuple[str, Array]]:
    chips: List[Tuple[str, Array]] = []
    for path in sorted(selfie_dir.glob("*")):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        chip = preprocess_selfie_to_yale(path, yale_mean=yale_mean, yale_std=yale_std)
        chips.append((path.name, chip))
    return chips


def evaluate_outlier_detection(
    decisions: Sequence[SrcDecision],
    labels_unknown: Sequence[bool],
    thresholds: OutlierThresholds,
) -> Tuple[float, float, float]:
    """Return acceptance rate on in-DB, rejection rate on unknown, overall accuracy."""
    in_db_accept = 0
    in_db_total = 0
    unknown_reject = 0
    unknown_total = 0
    correct = 0
    total = len(decisions)

    for decision, is_unknown in zip(decisions, labels_unknown):
        rejected = is_outlier(decision, thresholds)
        if is_unknown:
            unknown_total += 1
            if rejected:
                unknown_reject += 1
                correct += 1
        else:
            in_db_total += 1
            if not rejected:
                in_db_accept += 1
                correct += 1

    in_db_rate = 100.0 * in_db_accept / max(in_db_total, 1)
    unknown_rate = 100.0 * unknown_reject / max(unknown_total, 1)
    overall = 100.0 * correct / max(total, 1)
    return in_db_rate, unknown_rate, overall


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assignment 6 issue 3: selfies and outlier rejection")
    parser.add_argument("--mat-path", type=Path, default=default_mat_path())
    parser.add_argument("--selfie-dir", type=Path, default=Path(__file__).resolve().parent / "selfy")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-ratio", type=float, default=0.5)
    parser.add_argument("--lambda", dest="lam", type=float, default=0.01)
    parser.add_argument("--max-iter", type=int, default=250)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument(
        "--friend-test-index",
        type=int,
        default=0,
        help="Index into Yale test split used as in-database friend probe",
    )
    parser.add_argument(
        "--calibration-max",
        type=int,
        default=250,
        help="Max Yale test images for threshold calibration (0 = all)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    out_dir = args.output_dir
    pre_dir = out_dir / "preprocessed_selfies"
    pre_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    db = load_yale_database(args.mat_path)
    yale_mean = float(db.faces.mean())
    yale_std = float(db.faces.std())

    print(f"Loading Yale database and preprocessing selfies from {args.selfie_dir}")
    selfie_chips = load_preprocessed_selfies(args.selfie_dir, yale_mean=yale_mean, yale_std=yale_std)
    if not selfie_chips:
        raise FileNotFoundError(f"No selfie images found in {args.selfie_dir}")

    for name, chip in selfie_chips:
        save_chip_png(chip, pre_dir / f"{Path(name).stem}_yale.png")

    grid_path = out_dir / "preprocessed_selfies_grid.png"
    save_preprocessed_grid(selfie_chips, grid_path)

    X = vectorize_faces(db.faces)
    split = stratified_train_test_split(db.labels, args.train_ratio, rng)
    dictionary = build_dictionary(X, split.train_idx, db.labels[split.train_idx])

    calib_idx = split.test_idx
    if args.calibration_max > 0:
        calib_idx = calib_idx[: args.calibration_max]

    print(f"Calibrating outlier thresholds on {len(calib_idx)} in-database Yale test faces...")
    calib_decisions = [
        src_decision_one(X[:, idx], dictionary, args.lam, args.max_iter, args.tol) for idx in calib_idx
    ]
    thresholds = calibrate_outlier_thresholds(
        calib_decisions,
        residual_quantile=0.95,
        total_quantile=0.95,
    )
    in_db_accept, _, overall_calib = evaluate_outlier_detection(
        calib_decisions, [False] * len(calib_decisions), thresholds
    )
    print(
        f"Thresholds: r1<={thresholds.max_min_residual:.4f}, "
        f"r2/r1>={thresholds.min_residual_ratio:.3f}, "
        f"total<={thresholds.max_total_residual:.4f}, "
        f"conc>={thresholds.min_sparsity_concentration:.3f}"
    )
    print(f"Calibration in-DB acceptance: {in_db_accept:.1f}% (overall gate accuracy {overall_calib:.1f}%)")

    friend_idx = split.test_idx[args.friend_test_index % len(split.test_idx)]
    friend_label = int(db.labels[friend_idx])
    friend_chip = db.faces[friend_idx]

    rows: List[dict] = []
    probe_specs = (
        [("selfie", name, chip, True, -1) for name, chip in selfie_chips]
        + [("yale_friend_in_db", f"yale_test_{friend_idx}", friend_chip, False, friend_label)]
    )

    print("\nProbe results (SRC + outlier gate):")
    print(f"{'probe':<22} {'raw_pred':>8} {'r1':>8} {'r2/r1':>8} {'conc':>8} {'reject':>8}")
    for probe_type, name, chip, is_unknown, true_label in probe_specs:
        y = chip.reshape(-1)
        decision = src_decision_one(y, dictionary, args.lam, args.max_iter, args.tol)
        rejected = is_outlier(decision, thresholds)
        raw_pred = src_predict_one(y, dictionary, args.lam, args.max_iter, args.tol)
        recognized_you = (not rejected) and (probe_type == "selfie")
        rows.append(
            {
                "probe_type": probe_type,
                "name": name,
                "true_label": true_label,
                "raw_predicted_class": raw_pred,
                "min_residual": decision.min_residual,
                "residual_ratio": decision.residual_ratio,
                "sparsity_concentration": decision.sparsity_concentration,
                "total_residual": decision.total_residual,
                "rejected_as_unknown": int(rejected),
                "recognized_in_database": int(recognized_you),
            }
        )
        verdict = "UNKNOWN" if rejected else f"class {raw_pred}"
        print(
            f"{name:<22} {raw_pred:>8} {decision.min_residual:8.4f} "
            f"{decision.residual_ratio:8.3f} {decision.sparsity_concentration:8.3f} {verdict:>8}"
        )

    selfie_decisions = [
        src_decision_one(chip.reshape(-1), dictionary, args.lam, args.max_iter, args.tol)
        for _, chip in selfie_chips
    ]
    selfie_unknown_flags = [True] * len(selfie_decisions)
    friend_decision = src_decision_one(friend_chip.reshape(-1), dictionary, args.lam, args.max_iter, args.tol)

    eval_decisions = calib_decisions + selfie_decisions + [friend_decision]
    eval_unknown = [False] * len(calib_decisions) + selfie_unknown_flags + [False]
    in_db_rate, unknown_rate, overall = evaluate_outlier_detection(eval_decisions, eval_unknown, thresholds)

    n_selfies_rejected = sum(is_outlier(d, thresholds) for d in selfie_decisions)
    print(f"\nSelfies rejected as unknown: {n_selfies_rejected}/{len(selfie_decisions)}")
    print(f"Friend (Yale in-DB) accepted: {not is_outlier(friend_decision, thresholds)} (true class {friend_label})")
    print(f"Combined gate — in-DB accept {in_db_rate:.1f}%, unknown reject {unknown_rate:.1f}%, overall {overall:.1f}%")

    csv_path = out_dir / "issue3_selfie_probe_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_path = out_dir / "issue3_outlier_gate_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["max_min_residual", thresholds.max_min_residual])
        writer.writerow(["min_residual_ratio", thresholds.min_residual_ratio])
        writer.writerow(["max_total_residual", thresholds.max_total_residual])
        writer.writerow(["min_sparsity_concentration", thresholds.min_sparsity_concentration])
        writer.writerow(["selfies_rejected", n_selfies_rejected])
        writer.writerow(["selfies_total", len(selfie_decisions)])
        writer.writerow(["friend_accepted", int(not is_outlier(friend_decision, thresholds))])
        writer.writerow(["friend_true_class", friend_label])
        writer.writerow(["friend_raw_predicted", friend_decision.predicted_class])
        writer.writerow(["in_db_accept_rate_percent", in_db_rate])
        writer.writerow(["unknown_reject_rate_percent", unknown_rate])
        writer.writerow(["overall_gate_accuracy_percent", overall])

    print(f"\nWrote {grid_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
