"""
Metrics module for ASL-SR-DPT benchmark system.
Computes full-image spatial MSE, PSNR, and SSIM with strict numerical checks.
"""

import numpy as np
from skimage.metrics import structural_similarity as compute_skimage_ssim


def validate_image_pair(clean, reconstructed):
    """
    Validate that clean and reconstructed images have identical shape and finite values.
    """
    clean_arr = np.asarray(clean, dtype=float)
    rec_arr = np.asarray(reconstructed, dtype=float)

    if clean_arr.shape != rec_arr.shape:
        raise ValueError(
            f"Shape mismatch: clean image has shape {clean_arr.shape}, "
            f"reconstructed has shape {rec_arr.shape}."
        )

    if not np.all(np.isfinite(clean_arr)):
        raise FloatingPointError("Clean image contains NaN or Inf.")
    if not np.all(np.isfinite(rec_arr)):
        raise FloatingPointError("Reconstructed image contains NaN or Inf.")

    return clean_arr, rec_arr


def compute_mse(clean, reconstructed):
    """
    Compute Mean Squared Error (MSE) between two full spatial images.
    """
    clean_arr, rec_arr = validate_image_pair(clean, reconstructed)
    diff = clean_arr - rec_arr
    mse = float(np.mean(diff * diff))
    if not np.isfinite(mse) or mse < 0:
        raise FloatingPointError(f"Invalid MSE computed: {mse}")
    return mse


def compute_psnr(clean, reconstructed, max_val=1.0):
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) in dB on the full spatial image.
    PSNR = 10 * log10(max_val^2 / MSE)
    """
    mse = compute_mse(clean, reconstructed)
    if mse == 0.0:
        return float("inf")

    psnr = 10.0 * np.log10((max_val * max_val) / mse)
    if not np.isfinite(psnr):
        raise FloatingPointError(f"Invalid PSNR computed: {psnr}")
    return float(psnr)


def compute_ssim(clean, reconstructed, max_val=1.0):
    """
    Compute Structural Similarity Index (SSIM) on full spatial image.
    """
    clean_arr, rec_arr = validate_image_pair(clean, reconstructed)
    score = compute_skimage_ssim(clean_arr, rec_arr, data_range=max_val)
    score = float(score)
    if not np.isfinite(score):
        raise FloatingPointError(f"SSIM computation resulted in NaN or Inf: {score}")
    return score


def compute_all_metrics(clean, reconstructed, max_val=1.0):
    """
    Compute MSE, PSNR, and SSIM in a single call.
    Returns dict: {"mse": float, "psnr": float, "ssim": float}
    """
    mse = compute_mse(clean, reconstructed)
    psnr = 10.0 * np.log10((max_val * max_val) / mse) if mse > 0 else float("inf")
    ssim = compute_ssim(clean, reconstructed, max_val=max_val)

    return {
        "mse": float(mse),
        "psnr": float(psnr),
        "ssim": float(ssim),
    }
