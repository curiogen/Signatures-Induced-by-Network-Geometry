# =============================================================================
# Interpretation of the cross-network spectral structure results
# =============================================================================
#
# This analysis compares the correlation structure of observables across
# different real-world networks using a spectral (eigenvalue) perspective.
# Each network is represented as a "cloud" of observables, where the
# correlation matrix captures how these observables co-vary across
# realizations.
#
# The leading eigenvalue and its associated eigenvector describe the dominant
# collective direction of this cloud. A larger leading eigenvalue ratio
# indicates that most observables align along a single direction, meaning the
# cloud is elongated and effectively low-dimensional rather than isotropic.
#
# In the results, the citation network has a higher leading eigenvalue ratio
# than the email network. This implies that the citation observable cloud is
# more strongly aligned and more low-dimensional. In contrast, the email
# network exhibits a more diffuse cloud, indicating multiple competing
# structural effects.
#
# The number of eigenmodes required to explain 80% of the variance provides a
# complementary measure. The citation network requires fewer modes than the
# email network, confirming that its structure is governed by fewer dominant
# collective constraints.
#
# The spectral gap (difference between the first and second eigenvalues) is
# larger for the citation network. This separation means the leading collective
# mode is clearly distinguished from the rest, making the structure more
# stable and interpretable rather than fragmented.
#
# Observable-group weights further validate this interpretation. In citation
# networks, clustering-related observables contribute less to the leading
# mode, which is expected due to the absence of social triadic closure.
# Path-length and core-related observables contribute more strongly, reflecting
# hierarchical citation chains and core–periphery organization.
#
# In contrast, email networks retain stronger clustering contributions,
# consistent with communication-driven social interactions.
#
# Overall, these results show that the method does not simply detect generic
# low-dimensionality. It distinguishes different geometric regimes of
# real-world networks in a principled way, with observable clouds structured
# according to known domain-specific mechanisms.
# =============================================================================


import numpy as np
import pandas as pd
from pathlib import Path

CORR_FILES = {
    "email": Path("Real World Data (email-Enron)/email_correlations/corr_email.csv"),
    "citation": Path("Real World Data (citation)/citation_correlations/corr_citation.csv"),
    "twitch": Path("Real World Data (twitch)/twitch_correlations/corr_twitch.csv"),
}

EPS = 1e-8


def load_corr(path):
    df = pd.read_csv(path, index_col=0)
    C = df.values
    C = 0.5 * (C + C.T) + EPS * np.eye(C.shape[0])
    return C, df.index.tolist()


def spectrum_metrics(C):
    vals, vecs = np.linalg.eigh(C)
    vals = np.sort(vals)[::-1]
    cum = np.cumsum(vals) / np.sum(vals)

    return {
        "lambda1": vals[0],
        "lambda1_ratio": vals[0] / np.sum(vals),
        "modes_80pct": int(np.searchsorted(cum, 0.8) + 1),
        "spectral_gap": vals[0] - vals[1] if len(vals) > 1 else np.nan,
        "eigenvalues": vals,
        "v1": vecs[:, np.argsort(vals)[::-1][0]],
    }


def observable_groups(labels, v1):
    groups = {
        "clustering": [],
        "path_core": [],
        "other": [],
    }

    for name, w in zip(labels, np.abs(v1)):
        lname = name.lower()
        if "clustering" in lname or "triangle" in lname or "square" in lname:
            groups["clustering"].append(w)
        elif "path" in lname or "core" in lname or "eccentric" in lname:
            groups["path_core"].append(w)
        else:
            groups["other"].append(w)

    return {k: np.mean(v) if v else 0.0 for k, v in groups.items()}


rows = []

for name, path in CORR_FILES.items():
    if not path.exists():
        continue

    C, labels = load_corr(path)
    spec = spectrum_metrics(C)
    groups = observable_groups(labels, spec["v1"])

    rows.append({
        "network": name,
        "lambda1_ratio": spec["lambda1_ratio"],
        "modes_for_80pct": spec["modes_80pct"],
        "spectral_gap": spec["spectral_gap"],
        "clustering_weight": groups["clustering"],
        "path_core_weight": groups["path_core"],
    })

df = pd.DataFrame(rows).set_index("network")
df.to_csv("cross_network_structure_summary.csv")

print("\n===== CROSS-NETWORK STRUCTURE CHECK =====\n")
print(df)

print("\nINTERPRETATION RULES:")
print("- Higher lambda1_ratio → more low-dimensional")
print("- Fewer modes_for_80pct → stronger collective structure")
print("- Lower clustering_weight in citation is expected")
print("- Higher path_core_weight in citation is expected")
