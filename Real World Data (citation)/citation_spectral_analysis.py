import numpy as np
import pandas as pd
from pathlib import Path

corr = pd.read_csv("citation_correlations/corr_citation.csv", index_col=0)
labels = corr.index.tolist()
C = 0.5 * (corr.values + corr.values.T) + 1e-10 * np.eye(len(labels))

vals, vecs = np.linalg.eigh(C)
idx = np.argsort(vals)[::-1]
vals = vals[idx]
vecs = vecs[:, idx]

outdir = Path("citation_spectral_analysis")
outdir.mkdir(exist_ok=True)

pd.DataFrame({
    "index": range(1, len(vals) + 1),
    "eigenvalue": vals
}).to_csv(outdir / "eigenvalues.csv", index=False)

pd.DataFrame({
    "index": range(1, len(vals) + 1),
    "cumulative_variance": np.cumsum(vals) / vals.sum()
}).to_csv(outdir / "cumulative_variance.csv", index=False)

pd.DataFrame({
    "observable": labels,
    "abs_loading": np.abs(vecs[:, 0])
}).sort_values("abs_loading", ascending=False)\
 .to_csv(outdir / "leading_mode_loadings.csv", index=False)

print("Citation spectral analysis complete")
