import numpy as np
import pandas as pd
from pathlib import Path

C = pd.read_csv("citation_correlations/corr_citation.csv", index_col=0).values
C = 0.5 * (C + C.T) + 1e-8 * np.eye(C.shape[0])

vals = np.linalg.eigvalsh(C)
vals = np.sort(vals)[::-1]

outdir = Path("citation_cloud")
outdir.mkdir(exist_ok=True)

pd.DataFrame({
    "index": range(1, len(vals) + 1),
    "eigenvalue": vals
}).to_csv(outdir / "eigen_spectrum.csv", index=False)

print("Saved citation eigen spectrum")
