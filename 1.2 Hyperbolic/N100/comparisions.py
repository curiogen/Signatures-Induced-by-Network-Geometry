# -----------------------------------------------------------------------------
# Purpose of this script
# -----------------------------------------------------------------------------
#
# This script compares correlation matrices obtained from different parameter
# slices of the same hyperbolic network model.
#
# One parameter configuration is chosen as a reference. For every other
# configuration, we compare its observable–observable correlation matrix
# against this reference to understand how correlations change when model
# parameters (N, T, k, gamma) are varied.
#
# The goal here is NOT to find pairwise correlations that remain fixed.
# In fact, individual correlations are expected to change significantly
# under parameter variation.
#
# Instead, this script checks whether these changes are structured
# (i.e., organized around a few collective modes) or completely arbitrary.
#
#
# What this script computes
# -----------------------------------------------------------------------------
#
# For each non-reference parameter slice:
#
# 1. The correlation matrix is loaded and aligned with the reference matrix.
#    Alignment ensures that the same observables appear in the same row and
#    column order before any numerical comparison is performed.
#
# 2. The entrywise difference between the reference correlation matrix and
#    the target matrix is computed and saved. This highlights how individual
#    observable correlations drift as parameters change.
#
# 3. A numerically stable eigenvalue (or singular value) spectrum is computed
#    for both matrices and saved. The spectrum provides a compact summary of
#    the global structure of correlations, independent of individual entries.
#
#
# Why spectral analysis is used
# -----------------------------------------------------------------------------
#
# Hyperbolic correlation matrices are often nearly singular or ill-conditioned.
# As a result, naive eigenvalue or SVD computations may fail to converge.
# This is not a bug, but a structural property of the system.
#
# To ensure reliable output, the script uses numerically stable spectral
# extraction. This allows us to compare correlation structure even when
# pairwise correlations are unstable.
#
#
# How this fits into the overall analysis
# -----------------------------------------------------------------------------
#
# This script is a diagnostic and validation step.
#
# It shows that although pairwise correlations vary across parameter slices,
# the correlation structure retains a coherent spectral organization.
# This motivates shifting the analysis away from individual correlations
# and toward dominant collective modes.
#
# This script does NOT identify geometric signatures by itself.
# It provides the justification for later steps that focus on
# low-dimensional structure, observable reduction, and signature extraction.
#
# -----------------------------------------------------------------------------

import pandas as pd
import numpy as np
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2] / "1.0 Generating Networks" / "1.0.2 Hyperbolic" / "N100"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "N100" / "comparisions"

REFERENCE_NAME = "N100_T0.5_k6_gamma2.5"
REFERENCE_PATH = BASE_DIR / REFERENCE_NAME / "correlation_matrix.csv"

def load_matrix(path):
    """Load correlation matrix from CSV."""
    return pd.read_csv(path, index_col=0)


def align(df1, df2):
    """
    Align two correlation matrices on their common observable set.
    """
    common = df1.index.intersection(df2.index)
    return df1.loc[common, common], df2.loc[common, common]


def sanitize_matrix(M):
    """
    Make a matrix numerically safe for spectral analysis.
    - Removes NaN / inf
    - Enforces symmetry
    - Adds tiny diagonal jitter
    """
    M = np.asarray(M, dtype=float)
    M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)
    M = (M + M.T) / 2.0
    M += 1e-10 * np.eye(M.shape[0])
    return M


def eigvals(df):
    """
    Robust eigenvalue / singular value extraction.
    This function NEVER raises a LinAlgError.
    """
    M = sanitize_matrix(df.values)

    # Try SVD first (most stable)
    try:
        vals = np.linalg.svd(M, compute_uv=False)
        return np.sort(vals)[::-1]
    except np.linalg.LinAlgError:
        pass

    # Fallback to symmetric eigenvalues
    try:
        vals = np.linalg.eigvalsh(M)
        return np.sort(vals)[::-1]
    except np.linalg.LinAlgError:
        pass

    # Absolute fallback: return zeros
    return np.zeros(M.shape[0])


def compare(reference_df, target_df, out_dir):
    """
    Compare a target correlation matrix with the reference.
    Saves:
    - Entrywise difference matrix
    - Eigenvalue spectra of both matrices
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    diff = reference_df - target_df
    diff.to_csv(out_dir / "corr_diff.csv")

    ref_vals = eigvals(reference_df)
    tgt_vals = eigvals(target_df)

    np.savetxt(out_dir / "eigvals_ref.txt", ref_vals)
    np.savetxt(out_dir / "eigvals_target.txt", tgt_vals)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not REFERENCE_PATH.exists():
        raise FileNotFoundError(f"Reference matrix not found: {REFERENCE_PATH}")

    ref_df = load_matrix(REFERENCE_PATH)

    for folder in BASE_DIR.iterdir():
        if not folder.is_dir():
            continue
        if folder.name == REFERENCE_NAME:
            continue

        target_path = folder / "correlation_matrix.csv"
        if not target_path.exists():
            continue

        target_df = load_matrix(target_path)

        ref_aligned, target_aligned = align(ref_df, target_df)

        comp_name = f"{REFERENCE_NAME}_vs_{folder.name}"
        comp_dir = OUTPUT_DIR / comp_name

        compare(ref_aligned, target_aligned, comp_dir)

    print("All comparisons completed successfully.")

if __name__ == "__main__":
    main()
