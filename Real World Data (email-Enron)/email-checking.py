import numpy as np
import pandas as pd
from pathlib import Path

CORR_PATH = Path("email_correlations") / "corr_email.csv"
OUT_DIR = Path("email_spectral_analysis")
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

loading_df.to_csv(
    OUT_DIR / "leading_eigenvector_loadings.csv",
    index=False
)

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
#
# ================ SPECTRAL SUMMARY ================
#
# Top eigenvalues:
#   λ1 = 6.040
#   λ2 = 3.641
#   λ3 = 2.371
#   λ4 = 1.189
#   λ5 = 0.814
#
# Cumulative variance explained:
#   first 1 modes → 37.8%
#   first 2 modes → 60.5%
#   first 3 modes → 75.3%
#   first 4 modes → 82.8%
#   first 5 modes → 87.8%
#
# Top observables in leading collective mode:
#   degree_variance           0.3689
#   edge_jaccard_mean         0.3429
#   modularity                0.3406
#   edge_jaccard_variance     0.3404
#   kcore_depth               0.3305
#   global_clustering         0.3023
#   avg_degree                0.2752
