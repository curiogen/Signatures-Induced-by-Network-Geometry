# =============================================================================
# WHAT THIS SCRIPT DOES
#
# This script visualizes and compares the collective correlation structure of
# real-world Twitch networks against hyperbolic synthetic network ensembles.
#
# The goal is to move from raw correlation matrices to a geometric interpretation
# of structure using the "cloud" analogy:
#
# - Each correlation matrix defines a cloud of network realizations in observable
#   space.
# - Eigenvalues describe how variance is distributed across directions.
# - Eigenvectors describe the dominant collective modes of variation.
#
# This script generates four plots:
#
# 1. Eigenvalue spectrum (log-scale) for Twitch vs hyperbolic ensembles.
# 2. Cumulative variance explained by eigenmodes.
# 3. Stability of the leading eigenvector (cosine similarity).
# 4. Observable contributions to the leading collective mode.
#
# All plots are saved to the folder: ./plots
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CORR_DIR = "correlations"
TWITCH_FILE = "corr_twitch.csv"

PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

def load_corr(path):
    df = pd.read_csv(path, index_col=0)
    return df.values, df.index.tolist()

def safe_eigh(C):
    eps = 1e-6
    C = np.nan_to_num(C)
    C = (C + C.T) / 2
    C = C + eps * np.eye(C.shape[0])
    vals, vecs = np.linalg.eigh(C)
    idx = np.argsort(vals)[::-1]
    return vals[idx], vecs[:, idx]

def cosine_sim(a, b):
    return abs(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

spectra = {}
eigenvectors = {}
labels = None

for fname in os.listdir(CORR_DIR):
    if not fname.endswith(".csv"):
        continue
    C, labs = load_corr(os.path.join(CORR_DIR, fname))
    vals, vecs = safe_eigh(C)
    spectra[fname] = vals
    eigenvectors[fname] = vecs[:, 0]
    if labels is None:
        labels = labs

plt.figure(figsize=(7,5))
for name, vals in spectra.items():
    plt.plot(vals, marker="o", label=name.replace(".csv",""))
plt.yscale("log")
plt.xlabel("Eigenvalue index")
plt.ylabel("Eigenvalue (log scale)")
plt.title("Eigenvalue spectra: Twitch vs Hyperbolic")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "eigenvalue_spectra.png"))
plt.close()

plt.figure(figsize=(7,5))
for name, vals in spectra.items():
    cum = np.cumsum(vals) / np.sum(vals)
    plt.plot(cum, marker="o", label=name.replace(".csv",""))
plt.xlabel("Number of modes")
plt.ylabel("Cumulative variance explained")
plt.title("Cumulative variance explained")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "cumulative_variance.png"))
plt.close()

ref_vec = eigenvectors[TWITCH_FILE]
sims = {}

for name, v in eigenvectors.items():
    sims[name] = cosine_sim(ref_vec, v)

plt.figure(figsize=(7,4))
plt.bar(range(len(sims)), sims.values())
plt.xticks(range(len(sims)), [k.replace(".csv","") for k in sims.keys()],
           rotation=30, ha="right")
plt.ylabel("Cosine similarity with Twitch v₁")
plt.title("Leading eigenvector alignment")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "eigenvector_similarity.png"))
plt.close()

weights = {}
for name, v in eigenvectors.items():
    for obs, w in zip(labels, np.abs(v)):
        weights.setdefault(obs, []).append(w)

avg_weights = {k: np.mean(v) for k, v in weights.items()}
avg_weights = dict(sorted(avg_weights.items(), key=lambda x: x[1], reverse=True))

plt.figure(figsize=(9,4))
plt.bar(avg_weights.keys(), avg_weights.values())
plt.xticks(rotation=45, ha="right")
plt.ylabel("Average |v₁ component|")
plt.title("Dominant observables in collective mode")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "leading_mode_observables.png"))
plt.close()

print("All plots saved to ./plots")
