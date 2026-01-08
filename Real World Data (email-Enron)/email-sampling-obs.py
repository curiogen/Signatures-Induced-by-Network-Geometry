# -----------------------------------------------------------------------------
# Purpose of this script
# -----------------------------------------------------------------------------
#
# This script computes a consistent set of topological and geometry-aware
# observables on an ensemble of connected subsamples extracted from a large
# real-world network (email-Enron).
#
# Each subsample is treated as one realization of the same underlying system,
# allowing ensemble-level analysis rather than single-network inference.
#
#
# Measurement philosophy
# -----------------------------------------------------------------------------
#
# The observables are chosen to capture structure at multiple scales:
#   - local structure (degree statistics, clustering),
#   - mesoscopic organization (k-core structure, communities),
#   - edge-level similarity (Jaccard overlap),
#   - coarse geometric signatures (ball growth and hyperbolicity proxies).
#
# Direct computation of some geometric quantities scales poorly with network
# size and can become unstable on large graphs.
#
# To address this, the script uses randomized proxy estimators for:
#   - square motif density,
#   - ball growth rate,
#   - Gromov δ-hyperbolicity.
#
# These proxies preserve relative trends across realizations while remaining
# computationally feasible.
#
#
# Controlled skipping of expensive observables
# -----------------------------------------------------------------------------
#
# For graphs larger than a fixed node threshold, the most expensive geometric
# observables are intentionally skipped and recorded as NaN.
#
# This avoids numerical artifacts, non-convergence, and biased estimates while
# maintaining a uniform observable set across the ensemble.
#
# Downstream analyses explicitly handle missing values where appropriate.
#
#
# Output and downstream use
# -----------------------------------------------------------------------------
#
# The output is a numeric table where each row corresponds to one subsampled
# realization and each column corresponds to an observable.
#
# This table is designed for subsequent steps:
#   - correlation matrix construction,
#   - spectral decomposition,
#   - identification of dominant collective modes,
#   - comparison against synthetic hyperbolic network ensembles.
#
# No claims about geometry are made at this stage; this script prepares the
# empirical ensemble required for testing them.
#
# -----------------------------------------------------------------------------


import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from scipy.stats import pearsonr

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

NODE_THRESHOLD = 5000
SKIP_VALUE = np.nan


def load_subsample(path):
    G = nx.read_edgelist(path, nodetype=int)
    G = G.to_undirected()
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    return G


def safe_pearson(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return pearsonr(x, y)[0]


def triangle_density(G):
    n = G.number_of_nodes()
    if n < 3:
        return np.nan
    t = sum(nx.triangles(G).values()) / 3
    return t / (n * (n - 1) * (n - 2) / 6)


def square_density_proxy(G, n_samples=2000):
    nodes = list(G.nodes())
    vals = []
    for _ in range(n_samples):
        u = np.random.choice(nodes)
        Nu = set(G.neighbors(u))
        if len(Nu) < 2:
            continue
        v = np.random.choice(list(Nu))
        Nv = set(G.neighbors(v))
        vals.append(len(Nu & Nv))
    return np.mean(vals) if vals else np.nan


def ball_growth_proxy(G, n_samples=50, max_radius=5):
    nodes = np.random.choice(list(G.nodes()), min(n_samples, G.number_of_nodes()), replace=False)
    ratios = []
    for v in nodes:
        dist = nx.single_source_shortest_path_length(G, v)
        for r in range(1, max_radius):
            b_r = sum(d <= r for d in dist.values())
            b_r1 = sum(d <= r + 1 for d in dist.values())
            if b_r > 0:
                ratios.append(b_r1 / b_r)
    return np.mean(ratios) if ratios else np.nan


def gromov_delta_proxy(G, n_samples=300):
    nodes = list(G.nodes())
    if len(nodes) < 4:
        return np.nan
    deltas = []
    for _ in range(n_samples):
        a, b, c, d = np.random.choice(nodes, 4, replace=False)
        try:
            s = sorted([
                nx.shortest_path_length(G, a, b) + nx.shortest_path_length(G, c, d),
                nx.shortest_path_length(G, a, c) + nx.shortest_path_length(G, b, d),
                nx.shortest_path_length(G, a, d) + nx.shortest_path_length(G, b, c)
            ])
            deltas.append((s[2] - s[1]) / 2)
        except nx.NetworkXNoPath:
            continue
    return np.mean(deltas) if deltas else np.nan


def calculate_observables(G):
    deg = np.array([d for _, d in G.degree()])
    core = nx.core_number(G)
    kmax = max(core.values())
    core_frac = sum(v == kmax for v in core.values()) / G.number_of_nodes()

    clustering = nx.clustering(G)
    clust_vals = list(clustering.values())

    communities = nx.community.louvain_communities(G, seed=RANDOM_SEED)
    comm_sizes = [len(c) for c in communities]

    edges = list(G.edges())
    idx = np.random.choice(len(edges), min(1000, len(edges)), replace=False)
    jacc = []
    for i in idx:
        u, v = edges[i]
        Nu, Nv = set(G.neighbors(u)), set(G.neighbors(v))
        den = len(Nu | Nv)
        jacc.append(len(Nu & Nv) / den if den > 0 else 0)

    obs = {
        "avg_degree": np.mean(deg),
        "degree_variance": np.var(deg),
        "max_degree": np.max(deg),
        "kcore_depth": kmax,
        "core_fraction": core_frac,
        "global_clustering": nx.transitivity(G),
        "avg_local_clustering": np.mean(clust_vals),
        "degree_clustering_corr": safe_pearson(deg, clust_vals),
        "triangle_density": triangle_density(G),
        "square_density": square_density_proxy(G),
        "modularity": nx.community.modularity(G, communities),
        "community_size_variance": np.var(comm_sizes),
        "edge_jaccard_mean": np.mean(jacc),
        "edge_jaccard_variance": np.var(jacc),
        "ball_growth": SKIP_VALUE,
        "gromov_delta": SKIP_VALUE
    }

    if G.number_of_nodes() <= NODE_THRESHOLD:
        obs["ball_growth"] = ball_growth_proxy(G)
        obs["gromov_delta"] = gromov_delta_proxy(G)

    return obs


def main():
    subsample_dir = Path("email_subsamples")
    files = sorted(subsample_dir.glob("subsample_*.edgelist"))

    rows = []
    for i, f in enumerate(files):
        G = load_subsample(f)
        obs = calculate_observables(G)
        obs["realization_id"] = i
        rows.append(obs)

    df = pd.DataFrame(rows)
    df_numeric = df.select_dtypes(include=[np.number])

    outdir = Path("email_ensemble")
    outdir.mkdir(exist_ok=True)
    df_numeric.to_csv(outdir / "email_numeric_observables.csv", index=False)

    print("Saved", len(df_numeric), "realizations")


if __name__ == "__main__":
    main()
