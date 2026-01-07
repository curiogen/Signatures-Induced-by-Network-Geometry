import numpy as np
import pandas as pd
from pathlib import Path

IN_FILE = "twitch_ensemble/twitch_numeric_observables.csv"
OUT_DIR = Path("correlations")
OUT_DIR.mkdir(exist_ok=True)

OUT_FILE = OUT_DIR / "corr_twitch.csv"

df = pd.read_csv(IN_FILE)

df = df.select_dtypes(include=[np.number])

df = df.dropna(axis=1)

stds = df.std()
df = df.loc[:, stds > 1e-12]

corr = df.corr()

corr.to_csv(OUT_FILE)

print(f"Saved correlation matrix to {OUT_FILE}")
