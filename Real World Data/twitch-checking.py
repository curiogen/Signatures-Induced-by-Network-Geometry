# ============================================================
# WHAT THIS SCRIPT DOES (SHORT CONTEXT)
# ============================================================
#
# This script takes a correlation matrix of network observables
# (here: Twitch subsampled ensemble) and analyzes its collective
# geometric structure.
#
# Conceptually, each network realization is a point in a
# high-dimensional observable space. The correlation matrix
# describes the shape of the resulting point cloud.
#
# We compute:
# 1. The eigenvalue spectrum of the correlation matrix
#    → tells us how many effective dimensions the cloud has.
# 2. The cumulative explained variance
#    → tells us how quickly the cloud collapses onto a few axes.
# 3. The leading eigenvector
#    → tells us which observables define the main geometric direction.
#
# This is the first step before comparing Twitch directly
# with the hyperbolic synthetic ensemble using the same lens.
#
# ============================================================

import numpy as np
import pandas as pd
from pathlib import Path

CORR_PATH = Path("correlations") / "corr_twitch.csv"
OUT_DIR = Path("twitch_spectral_analysis")
OUT_DIR.mkdir(exist_ok=True)

corr_df = pd.read_csv(CORR_PATH, index_col=0)
labels = corr_df.index.tolist()
C = corr_df.values

C = 0.5 * (C + C.T)
C += 1e-10 * np.eye(C.shape[0])

eigvals, eigvecs = np.linalg.eigh(C)

idx = np.argsort(eigvals)[::-1]
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]

eig_df = pd.DataFrame({
    "index": np.arange(1, len(eigvals) + 1),
    "eigenvalue": eigvals
})
eig_df.to_csv(OUT_DIR / "eigenvalue_spectrum.csv", index=False)

total_var = np.sum(eigvals)
cumulative = np.cumsum(eigvals) / total_var

cum_df = pd.DataFrame({
    "index": np.arange(1, len(eigvals) + 1),
    "cumulative_variance": cumulative
})
cum_df.to_csv(OUT_DIR / "cumulative_variance.csv", index=False)

v1 = eigvecs[:, 0]
loadings = np.abs(v1)

loading_df = pd.DataFrame({
    "observable": labels,
    "abs_loading": loadings
}).sort_values("abs_loading", ascending=False)

loading_df.to_csv(OUT_DIR / "leading_eigenvector_loadings.csv", index=False)

print("\n================ SPECTRAL SUMMARY ================\n")

print("Top eigenvalues:")
for i in range(min(5, len(eigvals))):
    print(f"  λ{i+1} = {eigvals[i]:.3f}")

print("\nCumulative variance explained:")
for i in range(min(5, len(cumulative))):
    print(f"  first {i+1} modes → {cumulative[i]*100:.1f}%")

print("\nTop observables in leading collective mode:")
for _, row in loading_df.head(7).iterrows():
    print(f"  {row['observable']:25s} {row['abs_loading']:.4f}")


#
# ================ SPECTRAL SUMMARY ================
#
# Top eigenvalues:
#   λ1 = 7.106
#   λ2 = 4.045
#   λ3 = 1.742
#   λ4 = 1.116
#   λ5 = 0.926
#
# Cumulative variance explained:
#   first 1 modes → 41.8%
#   first 2 modes → 65.6%
#   first 3 modes → 75.8%
#   first 4 modes → 82.4%
#   first 5 modes → 87.9%
#
# Top observables in leading collective mode:
#   degree_var                0.3711
#   avg_local_clustering      0.3630
#   max_degree                0.3285
#   gromov_delta              0.3198
#   ball_growth               0.3134
#   modularity                0.2949
#   deg_clust_corr            0.2829
