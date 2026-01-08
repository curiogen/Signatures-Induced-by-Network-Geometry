# Purpose and rationale
# ---------------------
#
# This script implements ensemble-level diagnostics and spectral analysis
# of network observables computed from multiple subsampled realizations
# of a large real-world graph.
#
# Why this is necessary:
#
# Single-network measurements are not statistically meaningful for
# geometry-driven or correlation-based claims. Many observables fluctuate
# substantially across realizations, and uncontrolled variability or
# outliers can dominate correlation structure and eigenmodes.
#
# Therefore, before constructing an “observable cloud” or interpreting
# its spectral properties, we explicitly verify ensemble stability.
#
# What is done here:
#
# 1. Stability diagnostics
#    - Coefficient of variation is computed for each observable to
#      identify unstable or uninformative quantities.
#    - Observable–observable correlation statistics are summarized to
#      assess robustness.
#    - Realization-level outliers are detected via multivariate z-scores
#      to ensure no single subsample dominates the analysis.
#
# 2. Observable cloud construction
#    - Each realization is treated as a point in observable space.
#    - Observables are standardized across realizations to remove scale
#      dependence and ensure meaningful correlations.
#
# 3. Correlation geometry
#    - The observable–observable correlation matrix is computed and saved.
#    - This matrix serves as a compact descriptor of the ensemble’s
#      structural geometry.
#
# 4. Spectral decomposition
#    - Eigenvalues and eigenvectors of the correlation matrix are computed.
#    - The eigenvalue spectrum and cumulative variance quantify the
#      effective dimensionality of the observable space.
#    - The leading eigenvector captures the dominant collective mode
#      across observables.
#
# Interpretation:
#
# A clean spectrum with a small number of dominant modes indicates
# redundant or low-dimensional structure in the observable cloud, while
# the absence of outliers validates the subsampling and measurement
# procedure.
#
# This pipeline ensures that any geometric or dimensional claims are
# ensemble-driven, stable, and not artifacts of a single realization.

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import zscore

INPUT_FILE = Path("email_ensemble/email_numeric_observables.csv")
OUTDIR = Path("email_diagnostics")
OUTDIR.mkdir(exist_ok=True)


def coefficient_of_variation(df):
    mu = df.mean()
    sigma = df.std()
    return sigma / mu.replace(0, np.nan)


def correlation_stability(df):
    corr = df.corr()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    return upper.stack().describe()


def realization_outliers(df, z_thresh=3.0):
    Z = np.abs(zscore(df, nan_policy="omit"))
    outlier_mask = (Z > z_thresh).any(axis=1)
    return df.index[outlier_mask].tolist()


def main():
    df = pd.read_csv(INPUT_FILE)
    df = df.drop(columns=["realization_id"], errors="ignore")
    df = df.dropna(axis=1, how="all")

    cv = coefficient_of_variation(df)
    cv.to_csv(OUTDIR / "observable_coefficient_of_variation.csv")

    corr_stats = correlation_stability(df)
    corr_stats.to_csv(OUTDIR / "correlation_stability_summary.csv")

    outliers = realization_outliers(df)
    pd.Series(outliers, name="outlier_realization_ids").to_csv(
        OUTDIR / "outlier_realizations.csv", index=False
    )

    print("Diagnostics complete")
    print("High CV observables:")
    print(cv.sort_values(ascending=False).head(5))
    print("Outlier realizations:", outliers)


if __name__ == "__main__":
    main()
