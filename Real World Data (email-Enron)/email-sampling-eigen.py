import numpy as np
import pandas as pd
from pathlib import Path

CORR_DIR = Path("email_correlations")
OUT_FILE = Path("email_cloud/eigen_spectra.csv")
OUT_FILE.parent.mkdir(exist_ok=True)

EPS = 1e-8


def safe_eigenvalues(C):
    C = 0.5 * (C + C.T)
    C = C + EPS * np.eye(C.shape[0])
    vals = np.linalg.eigvalsh(C)
    return np.sort(vals)[::-1]


rows = []

for file in sorted(CORR_DIR.glob("corr_*.csv")):
    df = pd.read_csv(file, index_col=0)
    C = df.values

    try:
        eigvals = safe_eigenvalues(C)
    except Exception:
        continue

    for i, val in enumerate(eigvals):
        rows.append({
            "source": file.stem,
            "mode": i + 1,
            "eigenvalue": val
        })

out = pd.DataFrame(rows)
out.to_csv(OUT_FILE, index=False)

print(f"Saved eigen spectra to {OUT_FILE}")
