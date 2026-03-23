#!/usr/bin/env python3
"""
Inspect MATLAB .mat files and print stored variables.

Usage:
    python inspect_mat.py COMP5340HW1.mat
"""

from __future__ import annotations

import argparse
import os
from typing import Any

HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"


def summarize_value(value: Any, max_items: int = 8) -> str:
    """Return a short printable summary for arrays/scalars/other values."""
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            shape = value.shape
            dtype = value.dtype
            flat = value.ravel()
            preview = flat[:max_items]
            preview_str = np.array2string(preview, threshold=max_items, edgeitems=max_items)
            return f"ndarray shape={shape}, dtype={dtype}, preview={preview_str}"
        if np.isscalar(value):
            return f"scalar ({type(value).__name__}) = {value}"
    except Exception:
        pass
    return f"{type(value).__name__}: {value}"


def inspect_with_scipy(path: str) -> None:
    try:
        import scipy.io as sio
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "This file requires scipy. Install it with: pip install scipy"
        ) from exc

    data = sio.loadmat(path, squeeze_me=False, struct_as_record=False)
    print(f"\nLoaded with scipy.io.loadmat: {path}\n")
    for key, value in data.items():
        if key.startswith("__"):
            continue
        print(f"- {key}: {summarize_value(value)}")


def inspect_with_h5py(path: str) -> None:
    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "This file requires h5py. Install it with: pip install h5py"
        ) from exc

    print(f"\nLoaded with h5py (MATLAB v7.3 likely): {path}\n")
    with h5py.File(path, "r") as f:
        def visitor(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Dataset):
                print(f"- {name}: dataset shape={obj.shape}, dtype={obj.dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"- {name}: group")

        f.visititems(visitor)

def is_hdf5_file(path: str) -> bool:
    with open(path, "rb") as f:
        return f.read(8) == HDF5_SIGNATURE


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect variables stored in a .mat file")
    parser.add_argument("mat_file", help="Path to .mat file")
    args = parser.parse_args()

    mat_path = args.mat_file
    if not os.path.isfile(mat_path):
        raise FileNotFoundError(f"File not found: {mat_path}")

    if is_hdf5_file(mat_path):
        inspect_with_h5py(mat_path)
    else:
        inspect_with_scipy(mat_path)


if __name__ == "__main__":
    main()
