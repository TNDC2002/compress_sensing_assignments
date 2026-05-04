#!/usr/bin/env python3
"""
COMP5340 Homework 4 (Spring 2026)

Compressed sensing on real images with unknown sparsity and optional noise.

Sensing schemes:
1) Random Gaussian sensing matrix.
2) Random pixel-domain subsampling (rows from identity).

Sparsifying basis:
- 2D IDCT on patches.

Outputs:
- CSV with PSNR and timing versus measurement count M.
- PNG plots of PSNR vs M.
- Markdown summary with observations.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.fft import dctn, idctn, dct, idct

Array = np.ndarray


@dataclass(frozen=True)
class ImageConfig:
    name: str
    path: Path


def psnr(x_true: Array, x_hat: Array, max_value: float = 255.0) -> float:
    mse = float(np.mean((x_hat - x_true) ** 2))
    if mse <= 1e-15:
        return float("inf")
    return 10.0 * math.log10((max_value**2) / mse)


def load_grayscale_image(path: Path) -> Array:
    with Image.open(path) as img:
        gray = img.convert("L")
        return np.asarray(gray, dtype=float)


def dct_matrix(n: int) -> Array:
    return dct(np.eye(n), type=2, norm="ortho", axis=0)


def idct_matrix(n: int) -> Array:
    return idct(np.eye(n), type=2, norm="ortho", axis=0)


def patchify(image: Array, patch_size: int) -> Tuple[Array, Tuple[int, int]]:
    h, w = image.shape
    h2 = (h // patch_size) * patch_size
    w2 = (w // patch_size) * patch_size
    cropped = image[:h2, :w2]
    patches = (
        cropped.reshape(h2 // patch_size, patch_size, w2 // patch_size, patch_size)
        .swapaxes(1, 2)
        .reshape(-1, patch_size, patch_size)
    )
    return patches, (h2, w2)


def unpatchify(patches: Array, shape: Tuple[int, int], patch_size: int) -> Array:
    h, w = shape
    out = (
        patches.reshape(h // patch_size, w // patch_size, patch_size, patch_size)
        .swapaxes(1, 2)
        .reshape(h, w)
    )
    return out


def soft_threshold(x: Array, thresh: float) -> Array:
    return np.sign(x) * np.maximum(np.abs(x) - thresh, 0.0)


def ista_lasso(
    B: Array,
    y: Array,
    lam: float,
    max_iter: int,
    tol: float,
) -> Array:
    n = B.shape[1]
    a = np.zeros(n, dtype=float)
    # Lipschitz constant of grad(0.5||Ba - y||^2) is ||B^T B||_2.
    L = float(np.linalg.norm(B, ord=2) ** 2)
    step = 1.0 / max(L, 1e-12)

    for _ in range(max_iter):
        grad = B.T @ (B @ a - y)
        a_next = soft_threshold(a - step * grad, lam * step)
        if np.linalg.norm(a_next - a) <= tol * max(1.0, np.linalg.norm(a)):
            a = a_next
            break
        a = a_next
    return a


def build_sensing_matrix(
    scheme: str,
    m: int,
    n: int,
    rng: np.random.Generator,
) -> Tuple[Array, Array | None]:
    if scheme == "gaussian":
        A = rng.standard_normal((m, n)) / math.sqrt(m)
        return A, None
    if scheme == "subsampling":
        idx = rng.choice(n, size=m, replace=False)
        A = np.eye(n)[idx, :]
        return A, idx
    raise ValueError(f"Unknown scheme: {scheme}")


def estimate_average_sparsity(patches: Array, eps: float) -> float:
    counts: List[float] = []
    for patch in patches:
        coeff = dctn(patch, type=2, norm="ortho")
        counts.append(float(np.sum(np.abs(coeff) > eps)))
    return float(np.mean(counts)) if counts else 0.0


def reconstruct_image_from_measurements(
    image: Array,
    patch_size: int,
    m: int,
    scheme: str,
    lam: float,
    noise_sigma: float,
    ista_max_iter: int,
    ista_tol: float,
    rng: np.random.Generator,
    max_patches: int | None,
) -> Tuple[Array, float]:
    patches, shape = patchify(image, patch_size)
    n = patch_size * patch_size
    psi = idct_matrix(n)  # x = psi @ alpha

    if max_patches is not None and max_patches > 0:
        patch_count = min(max_patches, patches.shape[0])
    else:
        patch_count = patches.shape[0]

    reconstructed = np.zeros_like(patches)

    t0 = time.perf_counter()
    for i in range(patch_count):
        x_patch = patches[i].reshape(-1)
        A, _ = build_sensing_matrix(scheme, m, n, rng)
        y = A @ x_patch
        if noise_sigma > 0.0:
            y = y + rng.normal(0.0, noise_sigma, size=y.shape)

        B = A @ psi
        alpha_hat = ista_lasso(B, y, lam=lam, max_iter=ista_max_iter, tol=ista_tol)
        x_hat = psi @ alpha_hat
        reconstructed[i] = np.clip(x_hat.reshape(patch_size, patch_size), 0.0, 255.0)

    # Keep untouched patches when running quick smoke checks.
    if patch_count < patches.shape[0]:
        reconstructed[patch_count:] = patches[patch_count:]

    elapsed = time.perf_counter() - t0
    image_hat = unpatchify(reconstructed, shape, patch_size)
    return image_hat, elapsed


def nearest_measurements(n: int, ratios: List[float]) -> List[int]:
    vals = sorted(set(max(1, min(n, int(round(r * n)))) for r in ratios))
    return vals


def write_csv(
    path: Path,
    m_values: List[int],
    results: Dict[str, Dict[str, List[float]]],
) -> None:
    schemes = list(results.keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["M"] + [f"{s}_psnr" for s in schemes] + [f"{s}_time_sec" for s in schemes])
        for i, m in enumerate(m_values):
            row = [m]
            for s in schemes:
                row.append(results[s]["psnr"][i])
            for s in schemes:
                row.append(results[s]["time"][i])
            writer.writerow(row)


def save_plot(path: Path, m_values: List[int], results: Dict[str, Dict[str, List[float]]], title: str) -> None:
    plt.figure(figsize=(8, 5))
    for scheme, by_metric in results.items():
        plt.plot(m_values, by_metric["psnr"], marker="o", linewidth=1.5, label=scheme)
    plt.xlabel("Number of measurements per patch (M)")
    plt.ylabel("PSNR (dB)")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def best_scheme_by_psnr(results: Dict[str, Dict[str, List[float]]]) -> str:
    return max(results.keys(), key=lambda k: float(np.mean(results[k]["psnr"])))


def fastest_scheme(results: Dict[str, Dict[str, List[float]]]) -> str:
    return min(results.keys(), key=lambda k: float(np.mean(results[k]["time"])))


def first_reasonable_m(m_values: List[int], psnrs: List[float], threshold: float) -> int | None:
    for m, p in zip(m_values, psnrs):
        if p >= threshold:
            return m
    return None


def write_summary(
    path: Path,
    image_names: List[str],
    m_values: List[int],
    all_results: Dict[str, Dict[str, Dict[str, List[float]]]],
    avg_sparsity: Dict[str, float],
    psnr_reasonable_threshold: float,
    patch_n: int,
) -> None:
    lines: List[str] = []
    lines.append("# Assignment 4 Report: Compressed Sensing on Images")
    lines.append("")
    lines.append("## Setup")
    lines.append("- Sparsifying basis: 2D IDCT on small patches.")
    lines.append("- Recovery: ISTA for LASSO (noise-aware, no known sparsity required).")
    lines.append("- Sensing schemes: `gaussian` and `subsampling`.")
    lines.append("")

    lines.append("## Observations by image")
    for name in image_names:
        results = all_results[name]
        best = best_scheme_by_psnr(results)
        fastest = fastest_scheme(results)
        m_reasonable = {
            s: first_reasonable_m(m_values, results[s]["psnr"], psnr_reasonable_threshold)
            for s in results
        }
        lines.append(f"### {name}")
        lines.append(f"- Best recovery quality (average PSNR): `{best}`.")
        lines.append(f"- Fastest sensing/reconstruction pipeline: `{fastest}`.")
        lines.append(
            f"- Approximate average DCT sparsity per patch (|coef| > eps): `{avg_sparsity[name]:.2f}` / {patch_n}*."
        )
        for scheme, m_val in m_reasonable.items():
            if m_val is None:
                lines.append(
                    f"- `{scheme}` does not reach {psnr_reasonable_threshold:.1f} dB in tested M range."
                )
            else:
                lines.append(f"- `{scheme}` reaches {psnr_reasonable_threshold:.1f} dB at about M={m_val}.")
        lines.append("")

    lines.append("## Overall trends")
    lines.append("- Random Gaussian usually gives stronger incoherence with IDCT and better quality.")
    lines.append("- Pixel subsampling is typically faster due to simpler sensing operation.")
    lines.append("- More textured/real images tend to have higher effective sparsity level than synthetic phantom.")
    lines.append("")
    lines.append("*Denominator for sparsity is patch dimension n = patch_size^2.")

    path.write_text("\n".join(lines), encoding="utf-8")


def parse_ratios(values: str) -> List[float]:
    ratios = [float(x.strip()) for x in values.split(",") if x.strip()]
    if not ratios:
        raise ValueError("At least one measurement ratio is required.")
    return ratios


def main() -> None:
    parser = argparse.ArgumentParser(description="Assignment 4 image compressed sensing experiments")
    parser.add_argument("--materials-dir", default="Assignment_4/materials")
    parser.add_argument("--output-dir", default="Assignment_4")
    parser.add_argument("--patch-size", type=int, default=8, choices=[8, 16])
    parser.add_argument("--ratios", type=str, default="0.1,0.2,0.3,0.4,0.5,0.6")
    parser.add_argument("--noise-sigma", type=float, default=0.0)
    parser.add_argument("--lam", type=float, default=0.08, help="LASSO lambda for ISTA")
    parser.add_argument("--ista-max-iter", type=int, default=180)
    parser.add_argument("--ista-tol", type=float, default=1e-4)
    parser.add_argument("--sparsity-eps", type=float, default=3.0)
    parser.add_argument("--reasonable-psnr", type=float, default=28.0)
    parser.add_argument("--max-patches", type=int, default=0, help="0 means use all patches")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    materials = Path(args.materials_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    images = [
        ImageConfig("Phantom", materials / "Phantom.gif"),
        ImageConfig("Brain", materials / "Brain.gif"),
        ImageConfig("Boat", materials / "Boat.tif"),
    ]

    patch_n = args.patch_size * args.patch_size
    m_values = nearest_measurements(patch_n, parse_ratios(args.ratios))
    schemes = ["gaussian", "subsampling"]

    all_results: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    all_sparsity: Dict[str, float] = {}

    rng = np.random.default_rng(args.seed)

    for image_cfg in images:
        print(f"Processing {image_cfg.name} ...")
        image = load_grayscale_image(image_cfg.path)
        patches, _ = patchify(image, args.patch_size)
        all_sparsity[image_cfg.name] = estimate_average_sparsity(patches, eps=args.sparsity_eps)

        image_results: Dict[str, Dict[str, List[float]]] = {
            s: {"psnr": [], "time": []} for s in schemes
        }

        for m in m_values:
            print(f"  M={m}")
            for scheme in schemes:
                rec, elapsed = reconstruct_image_from_measurements(
                    image=image,
                    patch_size=args.patch_size,
                    m=m,
                    scheme=scheme,
                    lam=args.lam,
                    noise_sigma=args.noise_sigma,
                    ista_max_iter=args.ista_max_iter,
                    ista_tol=args.ista_tol,
                    rng=rng,
                    max_patches=(args.max_patches if args.max_patches > 0 else None),
                )
                target = image[: rec.shape[0], : rec.shape[1]]
                val = psnr(target, rec, max_value=255.0)
                image_results[scheme]["psnr"].append(val)
                image_results[scheme]["time"].append(elapsed)
                print(f"    {scheme}: PSNR={val:.3f} dB, time={elapsed:.2f}s")

        all_results[image_cfg.name] = image_results
        write_csv(out_dir / f"{image_cfg.name.lower()}_psnr_vs_m.csv", m_values, image_results)
        save_plot(
            out_dir / f"{image_cfg.name.lower()}_psnr_vs_m.png",
            m_values,
            image_results,
            title=f"{image_cfg.name}: PSNR vs measurements per patch",
        )

    write_summary(
        out_dir / "assignment_4_report.md",
        [x.name for x in images],
        m_values,
        all_results,
        all_sparsity,
        psnr_reasonable_threshold=args.reasonable_psnr,
        patch_n=patch_n,
    )
    print(f"Done. Outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
