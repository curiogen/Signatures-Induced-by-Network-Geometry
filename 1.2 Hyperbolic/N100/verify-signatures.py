# =============================================================================
# Purpose of this script
# =============================================================================
#
# This script verifies whether a robust *collective* signature exists in a
# family of hyperbolic network ensembles generated under different parameters.
#
# The goal is NOT to check whether individual correlations between observables
# are invariant. In fact, we expect pairwise correlations to change substantially
# when parameters such as temperature, degree, or power-law exponent vary.
#
# Instead, the script tests whether:
#   (i) correlations reorganize in a structured way,
#   (ii) a dominant low-dimensional collective mode persists, and
#   (iii) the same set of observables consistently defines that mode.
#
#
# High-level idea
# =============================================================================
#
# Each correlation matrix is treated as a point in "observable space".
# Even if individual entries change, the global organization of correlations
# can remain stable.
#
# This stability is detected through spectral analysis:
#   - eigenvalues describe the strength of collective modes,
#   - eigenvectors describe *which observables* participate in those modes.
#
# A robust hyperbolic signature is therefore defined as:
#   unstable pairwise correlations +
#   stable dominant eigenvector +
#   consistent observable participation.
#
#
# What this script checks
# =============================================================================
#
# Given a reference correlation matrix and multiple parameter slices, the script
# performs four diagnostic tests:
#
# 1. Pairwise correlation instability
#    --------------------------------
#    Measures the maximum absolute change in correlations across slices.
#    Large values confirm that no single correlation is stable by itself.
#
# 2. Leading eigenvalue stability
#    -----------------------------
#    Compares the range of the first and second eigenvalues across slices.
#    This indicates whether the dominant collective mode remains separated
#    from the bulk, even if its magnitude varies.
#
# 3. Leading eigenvector stability
#    ------------------------------
#    Computes cosine similarity between the leading eigenvectors of each slice
#    and the reference slice.
#    High similarity means the same *direction* in observable space is preserved.
#
# 4. Emergent observables
#    --------------------
#    Aggregates the absolute weights of observables in the leading eigenvector
#    across all slices.
#    Observables that consistently receive high weight are interpreted as
#    emergent components of the hyperbolic collective signature.
#
#
# Interpretation of outcomes
# =============================================================================
#
# - Pairwise instability is EXPECTED and REQUIRED.
# - Eigenvalue stability can be moderate or borderline without invalidating
#   the signature.
# - Eigenvector stability is the critical criterion.
# - Consistent recovery of key observables validates the signature claim.
#
# The script prints a final verdict summarizing whether the data supports
# the existence of a robust hyperbolic collective signature.
#
# =============================================================================

import numpy as np
import pandas as pd
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent

HYPERBOLIC_ROOT = (
    THIS_DIR.parents[1]
    / "1.0 Generating Networks"
    / "1.0.2 Hyperbolic"
    / "N100"
)

REFERENCE_SLICE = "N100_T0.5_k6_gamma2.5"
REFERENCE_PATH = (
    HYPERBOLIC_ROOT
    / REFERENCE_SLICE
    / "correlation_matrix.csv"
)

SIGNATURE_SET = {
    "core_periphery",
    "kcore_depth",
    "avg_degree",
    "avg_local_clustering",
    "degree_var",
}

PAIRWISE_THRESHOLD = 0.2
EIGENVECTOR_THRESHOLD_WEAK = 0.7
EIGENVECTOR_THRESHOLD_STRONG = 0.85

def sanitize_matrix(C, eps=1e-8):
    """
    Make matrix numerically safe for spectral decomposition.
    """
    C = np.array(C, dtype=float)

    # Replace NaNs and infs
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)

    # Enforce symmetry
    C = 0.5 * (C + C.T)

    # Fix diagonal
    np.fill_diagonal(C, 1.0)

    # Shift spectrum if needed
    min_eig = np.linalg.eigvalsh(C).min()
    if min_eig < 0:
        C += (-min_eig + eps) * np.eye(C.shape[0])

    return C


def safe_spectrum(C):
    """
    Guaranteed spectrum computation.
    Returns eigenvalues and eigenvectors.
    """
    C = sanitize_matrix(C)

    try:
        vals, vecs = np.linalg.eigh(C)
    except np.linalg.LinAlgError:
        # Final fallback: SVD on sanitized matrix
        U, S, _ = np.linalg.svd(C)
        vals = S
        vecs = U

    idx = np.argsort(vals)[::-1]
    return vals[idx], vecs[:, idx]


def cosine_similarity(a, b):
    return abs(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def load_corr(path):
    df = pd.read_csv(path, index_col=0)
    return df.values, df.index.tolist()

C_ref_raw, labels = load_corr(REFERENCE_PATH)
eigvals_ref, eigvecs_ref = safe_spectrum(C_ref_raw)
v1_ref = eigvecs_ref[:, 0]

pairwise_max = []
lambda1_vals = []
lambda2_vals = []
eigenvector_sims = []
weights = {lab: [] for lab in labels}

for folder in sorted(HYPERBOLIC_ROOT.iterdir()):
    if not folder.is_dir():
        continue

    corr_path = folder / "correlation_matrix.csv"
    if not corr_path.exists():
        continue

    C_raw, labs = load_corr(corr_path)
    eigvals, eigvecs = safe_spectrum(C_raw)

    # Pairwise instability
    delta = np.abs(sanitize_matrix(C_ref_raw) - sanitize_matrix(C_raw))
    pairwise_max.append(np.max(delta))

    # Eigenvalues
    lambda1_vals.append(eigvals[0])
    lambda2_vals.append(eigvals[1])

    # Eigenvector similarity
    sim = cosine_similarity(v1_ref, eigvecs[:, 0])
    eigenvector_sims.append(sim)

    # Eigenvector weights
    v1 = np.abs(eigvecs[:, 0])
    for lab, w in zip(labs, v1):
        weights[lab].append(w)

pairwise_max = np.array(pairwise_max)
eigenvector_sims = np.array(eigenvector_sims)

avg_weights = {k: np.mean(v) for k, v in weights.items()}
ranked = sorted(avg_weights.items(), key=lambda x: x[1], reverse=True)

top5_found = {k for k, _ in ranked[:5]}
missing = SIGNATURE_SET - top5_found

print("\n==================== CLAIM VERIFICATION ====================\n")

print("1. Pairwise correlation instability:")
print(f"   max |ΔC| = {pairwise_max.max():.3f}")
print("   PASS" if pairwise_max.max() >= PAIRWISE_THRESHOLD else "   FAIL")
print()

print("2. Eigenvalue stability:")
print(f"   λ1 range = [{min(lambda1_vals):.3f}, {max(lambda1_vals):.3f}]")
print(f"   λ2 range = [{min(lambda2_vals):.3f}, {max(lambda2_vals):.3f}]")
print("   PASS" if np.std(lambda1_vals) < np.std(lambda2_vals) else "   BORDERLINE")
print()

print("3. Leading eigenvector stability:")
print(f"   cosine similarities = {np.round(eigenvector_sims,3)}")
min_sim = eigenvector_sims.min()

if min_sim >= EIGENVECTOR_THRESHOLD_STRONG:
    print("   PASS")
elif min_sim >= EIGENVECTOR_THRESHOLD_WEAK:
    print("   BORDERLINE")
else:
    print("   FAIL")
print()

print("4. Emergent observables:")
for k, v in ranked[:10]:
    print(f"   {k:30s} {v:.4f}")

if not missing:
    print("\n   PASS: all signature observables recovered")
else:
    print(f"\n   FAIL: missing → {missing}")

print("\n==================== FINAL VERDICT ====================\n")

if (
    pairwise_max.max() >= PAIRWISE_THRESHOLD
    and min_sim >= EIGENVECTOR_THRESHOLD_WEAK
    and not missing
):
    print("OVERALL RESULT: ROBUST HYPERBOLIC COLLECTIVE SIGNATURE DETECTED.")
else:
    print("OVERALL RESULT: CLAIMS MUST BE WEAKENED.")


# ==================== CLAIM VERIFICATION ====================
#
# 1. Pairwise correlation instability:
#    max |ΔC| = 1.048
#    PASS
#
# 2. Eigenvalue stability:
#    λ1 range = [9.269, 13.005]
#    λ2 range = [3.212, 6.289]
#    BORDERLINE
#
# 3. Leading eigenvector stability:
#    cosine similarities = [0.909 0.887 0.894 1.    0.93  0.983 0.878 0.945 0.956]
#    PASS
#
# 4. Emergent observables:
#    core_periphery                 0.2912
#    kcore_depth                    0.2907
#    avg_degree                     0.2898
#    degree_var                     0.2844
#    avg_local_clustering           0.2811
#    triangle_density               0.2796
#    max_degree                     0.2704
#    norm_avg_path                  0.2542
#    community_size_var             0.2420
#    norm_avg_ecc                   0.2125
#
#    PASS: all signature observables recovered
#
# ==================== FINAL VERDICT ====================
#
# OVERALL RESULT: ROBUST HYPERBOLIC COLLECTIVE SIGNATURE DETECTED.