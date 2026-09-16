"""
Reconstruction and patch extraction module for ASL-SR-DPT.
Implements:
  1. Coordinated patch extraction with (x, y, patch_id) metadata.
  2. 2D orthonormal DCT and IDCT.
  3. Normalized 2D Hamming-window aggregation.
"""

import numpy as np
from hybrid_sparse_solver_v7_fixed import dct_2d, idct_2d


def dct_patch(patch):
    """
    Compute 2D orthonormal DCT of an 8x8 patch.
    """
    return dct_2d(patch)


def idct_patch(coefficients):
    """
    Compute 2D orthonormal IDCT of 8x8 DCT coefficients.
    """
    return idct_2d(coefficients)


def extract_patches(image, patch_size=8, stride=2):
    """
    Extract overlapping 2D patches from an image with coordinates.

    Parameters:
      image: np.ndarray of shape (H, W)
      patch_size: int, default 8
      stride: int, default 2

    Returns:
      patch_records: list of dicts with:
        "patch": np.ndarray of shape (patch_size, patch_size)
        "x": int, x-coordinate (column)
        "y": int, y-coordinate (row)
        "patch_id": int
    """
    img = np.asarray(image, dtype=float)
    if img.ndim != 2:
        raise ValueError(f"image must be a 2D array, got shape {img.shape}.")

    H, W = img.shape
    if H < patch_size or W < patch_size:
        raise ValueError(f"Image ({H}, {W}) is smaller than patch_size ({patch_size}).")

    y_coords = list(range(0, H - patch_size + 1, stride))
    if y_coords[-1] != H - patch_size:
        y_coords.append(H - patch_size)

    x_coords = list(range(0, W - patch_size + 1, stride))
    if x_coords[-1] != W - patch_size:
        x_coords.append(W - patch_size)

    records = []
    patch_id = 0
    for y in y_coords:
        for x in x_coords:
            p = img[y : y + patch_size, x : x + patch_size].copy()
            records.append({
                "patch": p,
                "x": int(x),
                "y": int(y),
                "patch_id": int(patch_id),
            })
            patch_id += 1

    return records


def reconstruct_image(patch_records, image_shape, patch_size=8, epsilon=1e-12):
    """
    Reconstruct full spatial image from reconstructed patches using
    normalized 2D Hamming-window aggregation.

    Parameters:
      patch_records: list of dicts containing:
        "patch": np.ndarray of shape (patch_size, patch_size) (or 64 flat coeff array converted to spatial)
        "x": int
        "y": int
      image_shape: tuple of (H, W)
      patch_size: int, default 8
      epsilon: float, numerical safety floor for division

    Returns:
      reconstructed: np.ndarray of shape image_shape, clipped to [0, 1].
    """
    H, W = image_shape
    weighted_accumulator = np.zeros((H, W), dtype=float)
    weight_accumulator = np.zeros((H, W), dtype=float)

    window_1d = np.hamming(patch_size)
    window_2d = np.outer(window_1d, window_1d)

    for rec in patch_records:
        p = rec["patch"]
        x = rec["x"]
        y = rec["y"]

        if p.shape != (patch_size, patch_size):
            p = p.reshape((patch_size, patch_size))

        weighted_accumulator[y : y + patch_size, x : x + patch_size] += p * window_2d
        weight_accumulator[y : y + patch_size, x : x + patch_size] += window_2d

    reconstructed = weighted_accumulator / np.maximum(weight_accumulator, epsilon)
    reconstructed = np.clip(reconstructed, 0.0, 1.0)

    if not np.all(np.isfinite(reconstructed)):
        raise FloatingPointError("Reconstruction resulted in NaN or Inf values.")

    return reconstructed
