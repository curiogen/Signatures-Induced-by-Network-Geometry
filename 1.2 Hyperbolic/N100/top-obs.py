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
    df = pd.read_csv(path, index_col=0)
    C = df.values.astype(float)
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
    return C, df.index.tolist()

def safe_spectrum(C):
    """
    Absolutely safe eigenvalue spectrum extraction.
    Never throws. Returns None if completely impossible.
    """
    C = (C + C.T) / 2.0
    C = np.nan_to_num(C)

    # Attempt 1: eigvalsh
    try:
        vals = np.linalg.eigvalsh(C)
        return np.sort(vals)[::-1]
    except:
        pass

    # Attempt 2: SVD
    try:
        s = np.linalg.svd(C, compute_uv=False)
        return np.sort(s)[::-1]
    except:
        pass

    # Attempt 3: progressively stronger regularization
    for eps in [1e-8, 1e-6, 1e-4]:
        try:
            C_reg = C + eps * np.eye(C.shape[0])
            s = np.linalg.svd(C_reg, compute_uv=False)
            return np.sort(s)[::-1]
        except:
            continue

    # Give up safely
    return None

def safe_leading_vector(C):
    """
    Safe dominant eigenvector extraction.
    """
    C = (C + C.T) / 2.0
    C = np.nan_to_num(C)

    try:
        vals, vecs = np.linalg.eigh(C)
        return vecs[:, np.argmax(vals)]
    except:
        try:
            C_reg = C + 1e-6 * np.eye(C.shape[0])
            _, _, vt = np.linalg.svd(C_reg)
            return vt[0]
        except:
            return None

corr_files = []
labels = {}

for folder in CORR_DIR.iterdir():
    if folder.is_dir():
        f = folder / "correlation_matrix.csv"
        if f.exists():
            corr_files.append(f)
            labels[f] = folder.name

assert len(corr_files) >= 2, "Need at least two correlation matrices."

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
    plt.title(f"|ΔC| vs reference\n{labels[path]}")
    plt.tight_layout()
    plt.savefig(PAIRWISE_DIR / f"delta_{labels[path]}.png")
    plt.close()

plt.figure(figsize=(7, 4))

for path in corr_files:
    C, _ = load_corr(path)
    eigvals = safe_spectrum(C)

    if eigvals is None:
        print(f"[WARNING] Skipping spectrum for {labels[path]}")
        continue

    plt.plot(eigvals, marker="o", label=labels[path])

plt.xlabel("Eigenvalue index")
plt.ylabel("Eigenvalue")
plt.title("Correlation spectrum across parameter slices")
plt.legend(fontsize=7)
plt.tight_layout()
plt.savefig(SPECTRA_DIR / "eigen_spectra.png")
plt.close()

v_ref = safe_leading_vector(C_ref)

sims = {}

for path in corr_files:
    if path == REF_PATH:
        continue

    C, _ = load_corr(path)
    v = safe_leading_vector(C)

    if v is None or v_ref is None:
        continue

    sims[labels[path]] = abs(np.dot(v_ref, v))

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
    C, labels_obs = load_corr(path)
    v1 = safe_leading_vector(C)

    if v1 is None:
        continue

    for obs, w in zip(labels_obs, np.abs(v1)):
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
