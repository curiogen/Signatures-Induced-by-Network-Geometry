# =============================================================================
# Purpose of this script
# =============================================================================
#
# This script analyzes how correlation structures between network observables
# behave across different parameter realizations of the same hyperbolic
# network model.
#
# The focus here is NOT on individual pairwise correlations being stable.
# In fact, we expect many correlations to change significantly when parameters
# such as N, T, k, or gamma are varied.
#
# Instead, this script checks whether these changes are structured at a
# collective level, by studying the spectrum and dominant modes of the
# correlation matrices.
#
#
# What this script produces
# =============================================================================
#
# For a set of correlation matrices (one per parameter slice), the script
# generates four diagnostic outputs:
#
# 1. Pairwise correlation instability maps
#    - Computes |ΔC| between each parameter slice and a reference slice.
#    - Visualizes how much individual correlations drift.
#    - This confirms that pairwise correlations are generally unstable.
#
# 2. Eigenvalue spectra of correlation matrices
#    - Computes the ordered eigenvalue spectrum for each correlation matrix.
#    - Eigenvalues summarize collective correlation structure.
#    - Similar spectral shapes across slices indicate low-dimensional structure.
#
# 3. Stability of the dominant eigenvector
#    - Measures cosine similarity between leading eigenvectors of different
#      parameter slices.
#    - High similarity implies that the same collective mode persists even
#      when individual correlations change.
#
# 4. Signature observables
#    - Aggregates the absolute loadings of observables in the leading eigenvector
#      across all parameter slices.
#    - Observables with consistently high loadings are interpreted as those
#      along which the dominant collective mode is expressed.
#
#
# Numerical robustness
# =============================================================================
#
# Correlation matrices derived from stochastic network ensembles can be
# ill-conditioned, contain near-linear dependencies, or suffer from numerical
# noise. Standard eigenvalue or SVD decompositions may fail in such cases.
#
# To ensure the pipeline never crashes:
# - All matrices are symmetrized and sanitized (NaNs and infinities removed).
# - Multiple decomposition strategies are attempted (eigvalsh, SVD).
# - Progressive diagonal regularization is applied when needed.
# - If a decomposition still fails, that specific slice is safely skipped.
#
# Skipping a problematic slice does not invalidate the analysis, since the
# goal is to detect ensemble-level collective structure, not exact invariance
# in every single realization.
#
#
# Interpretation
# =============================================================================
#
# If pairwise correlations are unstable but:
# - eigenvalue spectra remain similar,
# - the leading eigenvector is stable,
# - and the same observables dominate the leading mode,
#
# then the system exhibits a low-dimensional collective organization.
#
# These dominant observables are interpreted as a geometric "signature":
# not because they are invariant themselves, but because they are the main
# directions along which the underlying geometry expresses itself in the
# space of observables.
#
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")


BASE_DIR = Path(__file__).resolve().parents[2]

CORR_DIR = (
    BASE_DIR
    / "1.0 Generating Networks"
    / "1.0.2 Hyperbolic"
    / "N100"
)

OUT_DIR = Path(__file__).resolve().parent / "plots"

PAIRWISE_DIR = OUT_DIR / "pairwise_instability"
SPECTRA_DIR = OUT_DIR / "eigen_spectra"
EIGVEC_DIR = OUT_DIR / "eigenvector_stability"
SIGNATURE_DIR = OUT_DIR / "signature_observables"

for d in [PAIRWISE_DIR, SPECTRA_DIR, EIGVEC_DIR, SIGNATURE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

def load_corr(path):
    df = pd.read_csv(str(path), index_col=0)
    return df.values, df.index.tolist()

def safe_spectrum(C):
    """
    Robust spectrum computation.
    Tries eigvalsh → SVD → regularized SVD.
    """
    C = (C + C.T) / 2.0
    try:
        vals = np.linalg.eigvalsh(C)
        return np.sort(vals)[::-1], "eig"
    except:
        pass

    try:
        s = np.linalg.svd(C, compute_uv=False)
        return np.sort(s)[::-1], "svd"
    except:
        pass

    eps = 1e-8
    C_reg = C + eps * np.eye(C.shape[0])
    s = np.linalg.svd(C_reg, compute_uv=False)
    return np.sort(s)[::-1], "svd_reg"

def safe_leading_eigenvector(C):
    """
    Robust leading eigenvector extraction.
    """
    C = (C + C.T) / 2.0
    try:
        vals, vecs = np.linalg.eigh(C)
        idx = np.argsort(vals)[::-1]
        return vecs[:, idx[0]]
    except:
        eps = 1e-8
        C_reg = C + eps * np.eye(C.shape[0])
        _, _, vt = np.linalg.svd(C_reg)
        return vt[0]

corr_files = []
labels_map = {}

for folder in CORR_DIR.iterdir():
    if folder.is_dir():
        f = folder / "correlation_matrix.csv"
        if f.exists():
            corr_files.append(f)
            labels_map[f] = folder.name

assert len(corr_files) >= 2, "Not enough correlation matrices found."

# Reference = canonical parameter slice
REF_PATH = [p for p in corr_files if "N100_T0.5_k6_gamma2.5" in str(p)][0]
C_ref, obs_labels = load_corr(REF_PATH)

for path in corr_files:
    if path == REF_PATH:
        continue

    C, _ = load_corr(path)
    delta = np.abs(C_ref - C)

    plt.figure(figsize=(6, 5))
    plt.imshow(delta, cmap="viridis")
    plt.colorbar(label="|Δ correlation|")
    plt.title(f"|ΔC| vs reference\n{labels_map[path]}")
    plt.tight_layout()
    plt.savefig(PAIRWISE_DIR / f"delta_{labels_map[path]}.png")
    plt.close()

plt.figure(figsize=(7, 4))

for path in corr_files:
    C, _ = load_corr(path)
    eigvals, method = safe_spectrum(C)
    plt.plot(eigvals, marker="o", label=labels_map[path])

plt.xlabel("Eigenvalue index")
plt.ylabel("Eigenvalue")
plt.title("Correlation spectrum across parameter slices")
plt.legend(fontsize=7)
plt.tight_layout()
plt.savefig(SPECTRA_DIR / "eigen_spectra.png")
plt.close()

v_ref = safe_leading_eigenvector(C_ref)

sims = {}

for path in corr_files:
    if path == REF_PATH:
        continue

    C, _ = load_corr(path)
    v = safe_leading_eigenvector(C)
    cos_sim = abs(np.dot(v_ref, v))
    sims[labels_map[path]] = cos_sim

plt.figure(figsize=(7, 4))
plt.bar(range(len(sims)), sims.values())
plt.xticks(range(len(sims)), sims.keys(), rotation=30, ha="right")
plt.ylabel("Cosine similarity with reference v₁")
plt.title("Stability of dominant eigenvector")
plt.tight_layout()
plt.savefig(EIGVEC_DIR / "leading_eigenvector_similarity.png")
plt.close()

weights = {}

for path in corr_files:
    C, labels = load_corr(path)
    v1 = np.abs(safe_leading_eigenvector(C))

    for obs, w in zip(labels, v1):
        weights.setdefault(obs, []).append(w)

avg_weights = {k: np.mean(v) for k, v in weights.items()}
avg_weights = dict(sorted(avg_weights.items(), key=lambda x: x[1], reverse=True))

plt.figure(figsize=(8, 4))
plt.bar(avg_weights.keys(), avg_weights.values())
plt.xticks(rotation=45, ha="right")
plt.ylabel("Average |v₁ component|")
plt.title("Dominant observables in leading collective mode")
plt.tight_layout()
plt.savefig(SIGNATURE_DIR / "signature_observables.png")
plt.close()

print("All plots generated successfully.")
