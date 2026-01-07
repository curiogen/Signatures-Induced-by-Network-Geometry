import pandas as pd
import numpy as np
import os

CORR_STABILITY_THRESHOLD = 0.15
EIGENVALUE_RATIO_THRESHOLD = 0.85
DOMINANT_EIGENVALUE_VARIANCE_EXPLAINED = 0.80

def load_corr_diff(folder_path):
    path = os.path.join(folder_path, "corr_diff.csv")
    return pd.read_csv(path, index_col=0).values


def load_eigenvalues(folder_path, label):
    """
    Loads eigenvalues or singular values.
    Returns empty array if file missing or unreadable.
    """
    path = os.path.join(folder_path, f"eigvals_{label}.txt")
    try:
        vals = np.loadtxt(path)
        if np.ndim(vals) == 0:
            vals = np.array([vals])
        return vals
    except Exception:
        return np.array([])

def assess_correlation_stability(corr_diff):
    abs_diff = np.abs(corr_diff)
    return {
        "max_abs_diff": np.max(abs_diff),
        "mean_abs_diff": np.mean(abs_diff),
        "is_stable": np.max(abs_diff) < CORR_STABILITY_THRESHOLD
    }


def assess_eigenmode_preservation(eigvals_A, eigvals_B):
    # Keep only positive values
    eigvals_A = eigvals_A[eigvals_A > 1e-12]
    eigvals_B = eigvals_B[eigvals_B > 1e-12]

    if len(eigvals_A) == 0 or len(eigvals_B) == 0:
        return {
            "leading_ratio": 0.0,
            "num_dominant_A": 0,
            "num_dominant_B": 0,
            "status": "fragmented"
        }

    eigvals_A = np.sort(eigvals_A)[::-1]
    eigvals_B = np.sort(eigvals_B)[::-1]

    leading_ratio = min(eigvals_A[0], eigvals_B[0]) / max(eigvals_A[0], eigvals_B[0])

    cumsum_A = np.cumsum(eigvals_A) / np.sum(eigvals_A)
    cumsum_B = np.cumsum(eigvals_B) / np.sum(eigvals_B)

    num_dom_A = np.searchsorted(cumsum_A, DOMINANT_EIGENVALUE_VARIANCE_EXPLAINED) + 1
    num_dom_B = np.searchsorted(cumsum_B, DOMINANT_EIGENVALUE_VARIANCE_EXPLAINED) + 1

    if leading_ratio >= EIGENVALUE_RATIO_THRESHOLD and abs(num_dom_A - num_dom_B) <= 1:
        status = "preserved"
    elif leading_ratio >= 0.65 and abs(num_dom_A - num_dom_B) <= 2:
        status = "weakened"
    else:
        status = "fragmented"

    return {
        "leading_ratio": leading_ratio,
        "num_dominant_A": num_dom_A,
        "num_dominant_B": num_dom_B,
        "status": status
    }

def analyze_comparison_folder(folder_path):
    corr_diff = load_corr_diff(folder_path)

    eigvals_A = load_eigenvalues(folder_path, "ref")
    eigvals_B = load_eigenvalues(folder_path, "target")

    return {
        "folder": os.path.basename(folder_path),
        "corr": assess_correlation_stability(corr_diff),
        "eigen": assess_eigenmode_preservation(eigvals_A, eigvals_B)
    }

def print_folder_report(result):
    print("\n" + "=" * 80)
    print(f"Comparison: {result['folder']}")
    print("=" * 80)

    c = result["corr"]
    e = result["eigen"]

    print("\nCorrelation drift:")
    print(f"  Max |Δρ|  : {c['max_abs_diff']:.4f}")
    print(f"  Mean |Δρ| : {c['mean_abs_diff']:.4f}")
    print(f"  Status    : {'STABLE' if c['is_stable'] else 'UNSTABLE'}")

    print("\nSpectral structure:")
    print(f"  Leading ratio      : {e['leading_ratio']:.4f}")
    print(f"  Dominant modes (A) : {e['num_dominant_A']}")
    print(f"  Dominant modes (B) : {e['num_dominant_B']}")
    print(f"  Status             : {e['status'].upper()}")


def print_global_verdict(results):
    n = len(results)
    stable_corr = sum(r["corr"]["is_stable"] for r in results)
    preserved = sum(r["eigen"]["status"] == "preserved" for r in results)
    weakened = sum(r["eigen"]["status"] == "weakened" for r in results)

    avg_ratio = np.mean([r["eigen"]["leading_ratio"] for r in results])

    print("\n" + "#" * 80)
    print("GLOBAL VERDICT")
    print("#" * 80)

    if (
        stable_corr >= 0.75 * n and
        (preserved + weakened) >= 0.75 * n and
        avg_ratio >= 0.75
    ):
        print("YES: A stable low-dimensional collective structure exists.")
        print("Pairwise correlations drift, but dominant eigenmodes persist.")
    else:
        print("NO: Correlation structure is largely parameter-driven.")
        print("No robust collective signature detected.")

def analyze_all_comparisons(base_dir="comparisions"):
    folders = [
        os.path.join(base_dir, d)
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ]

    if not folders:
        print("No comparison folders found.")
        return

    results = []
    for folder in sorted(folders):
        res = analyze_comparison_folder(folder)
        results.append(res)
        print_folder_report(res)

    print_global_verdict(results)


if __name__ == "__main__":
    analyze_all_comparisons()
