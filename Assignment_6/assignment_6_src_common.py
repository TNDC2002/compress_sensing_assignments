"""Shared utilities for Assignment 6 SRC / robust SRC."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import scipy.io as sio
from PIL import Image, ImageOps

Array = np.ndarray


@dataclass(frozen=True)
class YaleDatabase:
    faces: Array
    labels: Array
    image_shape: Tuple[int, int]


@dataclass(frozen=True)
class SplitData:
    train_idx: Array
    test_idx: Array


@dataclass(frozen=True)
class TrainingDictionary:
    A: Array
    column_labels: Array
    gram: Array
    class_to_cols: Dict[int, Array]


@dataclass(frozen=True)
class SrcDecision:
    predicted_class: int
    min_residual: float
    second_residual: float
    residual_ratio: float
    class_residuals: Dict[int, float]
    total_residual: float
    sparsity_concentration: float


@dataclass(frozen=True)
class OutlierThresholds:
    max_min_residual: float
    min_residual_ratio: float
    max_total_residual: float
    min_sparsity_concentration: float


def default_mat_path() -> Path:
    return Path(__file__).resolve().parent / "material" / "CroppedYale_96_84_2414_subset.mat"


def load_yale_database(mat_path: Path) -> YaleDatabase:
    data = sio.loadmat(str(mat_path), squeeze_me=True)
    faces = np.asarray(data["faces"], dtype=float)
    labels = np.asarray(data["facecls"], dtype=int).ravel()
    if faces.ndim != 3:
        raise ValueError(f"Expected faces to be 3D, got shape {faces.shape}")
    return YaleDatabase(faces=faces, labels=labels, image_shape=(faces.shape[1], faces.shape[2]))


def vectorize_faces(faces: Array) -> Array:
    n_images = faces.shape[0]
    return faces.reshape(n_images, -1).T


def stratified_train_test_split(
    labels: Array,
    train_ratio: float,
    rng: np.random.Generator,
) -> SplitData:
    labels = np.asarray(labels, dtype=int)
    train_idx: list[int] = []
    test_idx: list[int] = []

    for cls in np.unique(labels):
        cls_idx = np.flatnonzero(labels == cls)
        rng.shuffle(cls_idx)
        n_train = int(round(train_ratio * len(cls_idx)))
        n_train = min(max(n_train, 1), len(cls_idx) - 1)
        train_idx.extend(cls_idx[:n_train].tolist())
        test_idx.extend(cls_idx[n_train:].tolist())

    train_idx_arr = np.asarray(train_idx, dtype=int)
    test_idx_arr = np.asarray(test_idx, dtype=int)
    rng.shuffle(train_idx_arr)
    rng.shuffle(test_idx_arr)
    return SplitData(train_idx=train_idx_arr, test_idx=test_idx_arr)


def l2_normalize_columns(A: Array, eps: float = 1e-12) -> Array:
    norms = np.linalg.norm(A, axis=0, keepdims=True)
    norms = np.maximum(norms, eps)
    return A / norms


def l2_normalize_vector(y: Array, eps: float = 1e-12) -> Array:
    norm = np.linalg.norm(y)
    if norm < eps:
        return y
    return y / norm


def soft_threshold(x: Array, thresh: float) -> Array:
    return np.sign(x) * np.maximum(np.abs(x) - thresh, 0.0)


def class_indices(column_labels: Array) -> Dict[int, Array]:
    mapping: Dict[int, Array] = {}
    for cls in np.unique(column_labels):
        mapping[int(cls)] = np.flatnonzero(column_labels == cls)
    return mapping


def build_dictionary(X: Array, train_idx: Array, train_labels: Array) -> TrainingDictionary:
    A = l2_normalize_columns(X[:, train_idx])
    return TrainingDictionary(
        A=A,
        column_labels=train_labels,
        gram=A.T @ A,
        class_to_cols=class_indices(train_labels),
    )


def solve_lasso_ista(
    gram: Array,
    at_y: Array,
    lam: float,
    max_iter: int = 300,
    tol: float = 1e-6,
) -> Array:
    n = gram.shape[0]
    alpha = np.zeros(n, dtype=float)
    lipschitz = float(np.linalg.norm(gram, ord=2))
    step = 1.0 / max(lipschitz, 1e-12)

    for _ in range(max_iter):
        grad = gram @ alpha - at_y
        alpha_next = soft_threshold(alpha - step * grad, lam * step)
        if np.linalg.norm(alpha_next - alpha) <= tol * max(1.0, np.linalg.norm(alpha)):
            return alpha_next
        alpha = alpha_next
    return alpha


def solve_robust_lasso_ista(
    A: Array,
    y: Array,
    lam_alpha: float,
    lam_e: float,
    max_iter: int = 300,
    tol: float = 1e-6,
) -> Tuple[Array, Array]:
    """
    Joint ISTA for
        min ||alpha||_1 + ||e||_1 + (lam/2) ||y - A alpha - e||_2^2
    with lam_alpha, lam_e scaling the two L1 terms (use lam_e = lam_alpha typically).
    """
    m, n = A.shape
    alpha = np.zeros(n, dtype=float)
    e = np.zeros(m, dtype=float)
    gram = A.T @ A
    lipschitz = float(np.linalg.norm(gram, ord=2)) + 1.0
    step = 1.0 / max(lipschitz, 1e-12)

    for _ in range(max_iter):
        residual = A @ alpha + e - y
        grad_alpha = A.T @ residual
        alpha_next = soft_threshold(alpha - step * grad_alpha, lam_alpha * step)
        e_next = soft_threshold(e - step * residual, lam_e * step)
        if (
            np.linalg.norm(alpha_next - alpha) <= tol * max(1.0, np.linalg.norm(alpha))
            and np.linalg.norm(e_next - e) <= tol * max(1.0, np.linalg.norm(e))
        ):
            return alpha_next, e_next
        alpha, e = alpha_next, e_next
    return alpha, e


def class_residuals_from_alpha(
    y: Array,
    dictionary: TrainingDictionary,
    alpha: Array,
) -> Dict[int, float]:
    residuals: Dict[int, float] = {}
    for cls, cols in dictionary.class_to_cols.items():
        residuals[int(cls)] = float(np.linalg.norm(y - dictionary.A[:, cols] @ alpha[cols]))
    return residuals


def sparsity_concentration(alpha: Array, column_labels: Array) -> float:
    """Fraction of L1 mass on the predicted class support."""
    abs_alpha = np.abs(alpha)
    total = float(np.sum(abs_alpha))
    if total <= 1e-12:
        return 0.0
    pred_cols = int(np.argmax(abs_alpha))
    pred_class = int(column_labels[pred_cols])
    class_energy = float(np.sum(abs_alpha[column_labels == pred_class]))
    return class_energy / total


def src_predict_from_alpha(
    y: Array,
    dictionary: TrainingDictionary,
    alpha: Array,
) -> int:
    residuals = class_residuals_from_alpha(y, dictionary, alpha)
    return min(residuals, key=residuals.get)


def src_decision_one(
    y: Array,
    dictionary: TrainingDictionary,
    lam: float,
    max_iter: int,
    tol: float,
) -> SrcDecision:
    y_norm = l2_normalize_vector(y)
    alpha = solve_lasso_ista(dictionary.gram, dictionary.A.T @ y_norm, lam, max_iter=max_iter, tol=tol)
    residuals = class_residuals_from_alpha(y_norm, dictionary, alpha)
    sorted_residuals = sorted(residuals.values())
    min_residual = sorted_residuals[0]
    second_residual = sorted_residuals[1] if len(sorted_residuals) > 1 else sorted_residuals[0]
    ratio = second_residual / max(min_residual, 1e-12)
    predicted = min(residuals, key=residuals.get)
    total_residual = float(np.linalg.norm(y_norm - dictionary.A @ alpha))
    concentration = sparsity_concentration(alpha, dictionary.column_labels)
    return SrcDecision(
        predicted_class=int(predicted),
        min_residual=float(min_residual),
        second_residual=float(second_residual),
        residual_ratio=float(ratio),
        class_residuals=residuals,
        total_residual=total_residual,
        sparsity_concentration=float(concentration),
    )


def src_predict_one(
    y: Array,
    dictionary: TrainingDictionary,
    lam: float,
    max_iter: int,
    tol: float,
) -> int:
    return src_decision_one(y, dictionary, lam, max_iter, tol).predicted_class


def calibrate_outlier_thresholds(
    decisions: Sequence[SrcDecision],
    residual_quantile: float = 0.98,
    ratio_quantile: float = 0.05,
    total_quantile: float = 0.98,
    concentration_quantile: float = 0.05,
) -> OutlierThresholds:
    """Calibrate rejection thresholds from in-database validation decisions."""
    min_residuals = np.array([d.min_residual for d in decisions], dtype=float)
    ratios = np.array([d.residual_ratio for d in decisions], dtype=float)
    totals = np.array([d.total_residual for d in decisions], dtype=float)
    concentrations = np.array([d.sparsity_concentration for d in decisions], dtype=float)
    return OutlierThresholds(
        max_min_residual=float(np.quantile(min_residuals, residual_quantile)),
        min_residual_ratio=float(np.quantile(ratios, ratio_quantile)),
        max_total_residual=float(np.quantile(totals, total_quantile)),
        min_sparsity_concentration=float(np.quantile(concentrations, concentration_quantile)),
    )


def is_outlier(decision: SrcDecision, thresholds: OutlierThresholds) -> bool:
    """Reject unknown identity when fit quality or class concentration is poor."""
    if decision.min_residual > thresholds.max_min_residual:
        return True
    if decision.residual_ratio < thresholds.min_residual_ratio:
        return True
    if decision.total_residual > thresholds.max_total_residual:
        return True
    if decision.sparsity_concentration < thresholds.min_sparsity_concentration:
        return True
    return False


def detect_face_bbox_gray(gray: Array) -> Tuple[int, int, int, int] | None:
    try:
        import cv2
    except ModuleNotFoundError:
        return None

    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        return None

    image_u8 = np.clip(gray, 0, 255).astype(np.uint8)
    faces = detector.detectMultiScale(image_u8, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    return int(x), int(y), int(w), int(h)


def preprocess_selfie_to_yale(
    image_path: Path,
    target_shape: Tuple[int, int] = (96, 84),
    yale_mean: float | None = None,
    yale_std: float | None = None,
) -> Array:
    """
    Convert a selfie to Yale-style grayscale face chips (height=96, width=84).
    Uses face detection when available, otherwise a centered upper-body crop.
    """
    target_h, target_w = target_shape
    with Image.open(image_path) as img:
        gray = ImageOps.grayscale(img)
        gray = ImageOps.exif_transpose(gray)
        arr = np.asarray(gray, dtype=float)

    h, w = arr.shape
    bbox = detect_face_bbox_gray(arr)
    if bbox is not None:
        x, y, bw, bh = bbox
        pad_y = int(0.35 * bh)
        pad_x = int(0.25 * bw)
        top = max(y - pad_y, 0)
        bottom = min(y + bh + pad_y, h)
        left = max(x - pad_x, 0)
        right = min(x + bw + pad_x, w)
        crop = arr[top:bottom, left:right]
    else:
        crop_h = int(h * 0.62)
        crop_w = int(crop_h * target_w / target_h)
        crop_w = min(crop_w, w)
        crop_h = min(crop_h, h)
        top = int(h * 0.10)
        left = max((w - crop_w) // 2, 0)
        crop = arr[top : top + crop_h, left : left + crop_w]

    chip = Image.fromarray(np.clip(crop, 0, 255).astype(np.uint8))
    chip = chip.resize((target_w, target_h), Image.Resampling.LANCZOS)
    out = np.asarray(chip, dtype=float)

    if yale_mean is not None and yale_std is not None and yale_std > 1e-8:
        out = (out - out.mean()) / max(out.std(), 1e-8) * yale_std + yale_mean
    out = np.clip(out, 0.0, 255.0)
    return out


def robust_src_predict_one(
    y: Array,
    dictionary: TrainingDictionary,
    lam_alpha: float,
    lam_e: float,
    max_iter: int,
    tol: float,
) -> int:
    y_norm = l2_normalize_vector(y)
    alpha, _ = solve_robust_lasso_ista(
        dictionary.A,
        y_norm,
        lam_alpha=lam_alpha,
        lam_e=lam_e,
        max_iter=max_iter,
        tol=tol,
    )
    return src_predict_from_alpha(y_norm, dictionary, alpha)


def corrupt_random_pixels(
    y: Array,
    corruption_frac: float,
    rng: np.random.Generator,
    value_range: Tuple[float, float] = (0.0, 255.0),
) -> Array:
    """Replace a fraction of entries with uniform random values."""
    if corruption_frac <= 0.0:
        return y.copy()
    y_corrupt = y.copy()
    n = y.size
    n_corrupt = int(round(corruption_frac * n))
    n_corrupt = min(max(n_corrupt, 0), n)
    if n_corrupt == 0:
        return y_corrupt
    idx = rng.choice(n, size=n_corrupt, replace=False)
    y_corrupt[idx] = rng.uniform(value_range[0], value_range[1], size=n_corrupt)
    return y_corrupt


def corrupt_random_patches(
    y: Array,
    image_shape: Tuple[int, int],
    corruption_frac: float,
    patch_size: Tuple[int, int],
    rng: np.random.Generator,
    value_range: Tuple[float, float] = (0.0, 255.0),
) -> Array:
    """Overwrite random rectangular patches until the target pixel fraction is reached."""
    if corruption_frac <= 0.0:
        return y.copy()

    h, w = image_shape
    ph, pw = patch_size
    img = y.reshape(h, w).copy()
    mask = np.zeros((h, w), dtype=bool)
    target = int(round(corruption_frac * h * w))
    if target <= 0:
        return img.ravel()

    attempts = 0
    max_attempts = max(500, target // max(ph * pw, 1) * 20)
    while mask.sum() < target and attempts < max_attempts:
        attempts += 1
        r0 = int(rng.integers(0, max(h - ph + 1, 1)))
        c0 = int(rng.integers(0, max(w - pw + 1, 1)))
        patch_mask = np.zeros((h, w), dtype=bool)
        patch_mask[r0 : r0 + ph, c0 : c0 + pw] = True
        new_pixels = patch_mask & ~mask
        if not np.any(new_pixels):
            continue
        mask |= new_pixels
        img[new_pixels] = rng.uniform(value_range[0], value_range[1], size=int(new_pixels.sum()))

    if mask.sum() < target:
        remaining = np.flatnonzero(~mask.ravel())
        extra = min(target - int(mask.sum()), len(remaining))
        if extra > 0:
            pick = rng.choice(remaining, size=extra, replace=False)
            img.ravel()[pick] = rng.uniform(value_range[0], value_range[1], size=extra)

    return img.ravel()
