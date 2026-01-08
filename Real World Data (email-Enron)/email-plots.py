import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CORR_DIR = "email_correlations"
REF_FILE = "corr_email.csv"

PLOTS_DIR = "email_plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


def load_corr(path):
    df = pd.read_csv(path, index_col=0)
    return df.values, df.index.tolist()


def safe_eigh(C):
    eps = 1e-6
    C = np.nan_to_num(C)
    C = 0.5 * (C + C.T)
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


plt.figure(figsize=(7, 5))
for name, vals in spectra.items():
    plt.plot(vals, marker="o", label=name.replace(".csv", ""))
plt.yscale("log")
plt.xlabel("Eigenvalue index")
plt.ylabel("Eigenvalue (log scale)")
plt.title("Eigenvalue spectra")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "eigenvalue_spectra.png"))
plt.close()


plt.figure(figsize=(7, 5))
for name, vals in spectra.items():
    cum = np.cumsum(vals) / np.sum(vals)
    plt.plot(cum, marker="o", label=name.replace(".csv", ""))
plt.xlabel("Number of modes")
plt.ylabel("Cumulative variance explained")
plt.title("Cumulative variance explained")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "cumulative_variance.png"))
plt.close()


ref_vec = eigenvectors[REF_FILE]
sims = {name: cosine_sim(ref_vec, v) for name, v in eigenvectors.items()}

plt.figure(figsize=(7, 4))
plt.bar(range(len(sims)), sims.values())
plt.xticks(
    range(len(sims)),
    [k.replace(".csv", "") for k in sims.keys()],
    rotation=30,
    ha="right"
)
plt.ylabel("Cosine similarity with reference v1")
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

plt.figure(figsize=(9, 4))
plt.bar(avg_weights.keys(), avg_weights.values())
plt.xticks(rotation=45, ha="right")
plt.ylabel("Average |v1 component|")
plt.title("Dominant observables in leading mode")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "leading_mode_observables.png"))
plt.close()

print("All plots saved to", PLOTS_DIR)
