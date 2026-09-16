"""
Dataset module for BSD68 grayscale benchmark.
Provides deterministic loading, shape verification, and image ID assignment.
"""

import os
import glob
import numpy as np
from PIL import Image


def get_bsd68_filenames(dataset_dir):
    """
    Return deterministically sorted list of BSD68 image paths.
    """
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    extensions = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(dataset_dir, ext)))
        files.extend(glob.glob(os.path.join(dataset_dir, ext.upper())))

    # Deduplicate and sort deterministically by basename
    unique_files = sorted(list(set(files)), key=lambda p: os.path.basename(p))
    return unique_files


def load_image_grayscale(image_path):
    """
    Load a single image as float64 array in range [0.0, 1.0].
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    with Image.open(image_path) as img:
        img_gray = img.convert("L")
        arr = np.asarray(img_gray, dtype=float) / 255.0

    if arr.ndim != 2:
        raise ValueError(f"Expected 2D grayscale image, got shape {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise FloatingPointError(f"Image {image_path} contains non-finite values.")
    return arr


def load_bsd68(dataset_dir="data/BSD68", limit=None, image_ids=None, strict_count=68):
    """
    Load BSD68 images deterministically.

    Parameters:
      dataset_dir: str, path to directory containing BSD68 images.
      limit: int or None, max number of images to return (takes first N in deterministic order).
      image_ids: list of str or None, specific image IDs to load.
      strict_count: int or None, expected total count in dataset (default 68).

    Returns:
      dataset: dict mapping image_id -> dict with:
        "image_id": str (e.g. "test001")
        "path": str
        "image": np.ndarray of shape (H, W) in [0, 1]
        "shape": tuple (H, W)
    """
    all_files = get_bsd68_filenames(dataset_dir)
    total_found = len(all_files)

    if strict_count is not None and total_found != strict_count:
        if limit is None or limit > total_found:
            raise ValueError(
                f"BSD68 validation error: Expected exactly {strict_count} images, but found {total_found} in {dataset_dir}."
            )

    selected_files = all_files
    if image_ids is not None:
        selected_set = set(image_ids)
        selected_files = [f for f in all_files if os.path.splitext(os.path.basename(f))[0] in selected_set]
    elif limit is not None and limit > 0:
        selected_files = all_files[:limit]

    dataset = {}
    for fpath in selected_files:
        img_id = os.path.splitext(os.path.basename(fpath))[0]
        img_arr = load_image_grayscale(fpath)
        dataset[img_id] = {
            "image_id": img_id,
            "path": fpath,
            "image": img_arr,
            "shape": img_arr.shape,
        }

    return dataset
