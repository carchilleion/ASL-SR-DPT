"""
Sensing module for ASL-SR-DPT benchmark system.
Implements:
  1. Standard Gaussian sensing with row normalization.
  2. DC-preserving sensing for AC coefficients.
  3. Measurement vector generation.
  4. DC component restoration.
  5. Deterministic AWGN generation.
"""

import numpy as np


def generate_random_sensing(M=38, N=64, seed=None):
    """
    Generate an M x N standard Gaussian random sensing matrix with unit-normalized rows.

    Parameters:
      M: int, number of measurements (rows).
      N: int, ambient dimension (columns).
      seed: int or None, random seed for reproducibility.

    Returns:
      A: np.ndarray of shape (M, N) with row norms equal to 1.
    """
    if M <= 0 or N <= 0:
        raise ValueError("M and N must be positive integers.")
    if M > N:
        raise ValueError("Compressed sensing requires M <= N.")

    rng = np.random.default_rng(seed)
    A = rng.standard_normal(size=(M, N))

    # Required row normalization: each row has unit L2 norm
    row_norms = np.linalg.norm(A, axis=1, keepdims=True)
    row_norms = np.where(row_norms < 1e-12, 1.0, row_norms)
    A = A / row_norms

    if not np.all(np.isfinite(A)):
        raise FloatingPointError("Sensing matrix contains NaN or Inf.")
    return A


def generate_dc_sensing(M_ac=37, N_ac=63, seed=None):
    """
    Generate an M_ac x N_ac sensing matrix for the 63 AC DCT coefficients.
    """
    return generate_random_sensing(M=M_ac, N=N_ac, seed=seed)


def generate_standard_measurement(patch_dct, A):
    """
    Compute standard linear measurement vector y = A @ theta.

    Parameters:
      patch_dct: np.ndarray of shape (8, 8) or (64,)
      A: np.ndarray of shape (M, N)

    Returns:
      y: np.ndarray of shape (M,)
    """
    theta = np.asarray(patch_dct, dtype=float).reshape(-1)
    if theta.size != A.shape[1]:
        raise ValueError(f"patch_dct size ({theta.size}) must match A columns ({A.shape[1]}).")
    y = A @ theta
    return y


def generate_dc_preserving_measurement(patch_dct, A_ac):
    """
    Extract DC coefficient directly without compression and measure AC coefficients.

    Parameters:
      patch_dct: np.ndarray of shape (8, 8) or (64,)
      A_ac: np.ndarray of shape (M_ac, 63)

    Returns:
      theta_dc: float, noisy uncompressed DC coefficient.
      y_ac: np.ndarray of shape (M_ac,), linear measurement of the 63 AC coefficients.
    """
    theta = np.asarray(patch_dct, dtype=float).reshape(-1)
    if theta.size != 64:
        raise ValueError(f"patch_dct must have exactly 64 coefficients, got {theta.size}.")
    if A_ac.shape[1] != 63:
        raise ValueError(f"A_ac must have 63 columns, got {A_ac.shape[1]}.")

    theta_dc = float(theta[0])
    theta_ac = theta[1:]
    y_ac = A_ac @ theta_ac
    return theta_dc, y_ac


def restore_dc_component(theta_dc, theta_ac_hat):
    """
    Reassemble full 64-coefficient DCT vector by prepending the preserved DC coefficient.

    Parameters:
      theta_dc: float, preserved DC coefficient.
      theta_ac_hat: np.ndarray of shape (63,), recovered AC coefficients.

    Returns:
      theta_hat: np.ndarray of shape (64,)
    """
    theta_ac_hat = np.asarray(theta_ac_hat, dtype=float).reshape(-1)
    if theta_ac_hat.size != 63:
        raise ValueError(f"theta_ac_hat must have length 63, got {theta_ac_hat.size}.")

    theta_hat = np.empty(64, dtype=float)
    theta_hat[0] = float(theta_dc)
    theta_hat[1:] = theta_ac_hat
    return theta_hat


def add_awgn(image, sigma_noise, seed=None):
    """
    Add zero-mean Additive White Gaussian Noise (AWGN) to an image in [0, 1].

    Parameters:
      image: np.ndarray, clean image with values in [0, 1].
      sigma_noise: float, noise level in 0-255 scale (e.g. 15, 25, 50).
      seed: int or None, random seed for noise reproducibility.

    Returns:
      noisy_image: np.ndarray clipped to [0, 1].
    """
    img = np.asarray(image, dtype=float)
    sigma_scaled = float(sigma_noise) / 255.0
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=sigma_scaled, size=img.shape)
    noisy = np.clip(img + noise, 0.0, 1.0)
    return noisy
