import numpy as np
import pandas as pd
from pathlib import Path

df = pd.read_csv("citation_ensemble/citation_numeric_observables.csv")
df = df.select_dtypes(include=[np.number]).drop(columns=["realization_id"], errors="ignore")
df = df.dropna(axis=1)

stds = df.std()
df = df.loc[:, stds > 1e-12]

corr = df.corr()

outdir = Path("citation_correlations")
outdir.mkdir(exist_ok=True)
corr.to_csv(outdir / "corr_citation.csv")

print("Saved citation correlation matrix")
