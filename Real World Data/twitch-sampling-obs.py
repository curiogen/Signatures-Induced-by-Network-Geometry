import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from scipy.stats import pearsonr

# =========================
# CONFIG
# =========================
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

NODE_THRESHOLD = 5000
SKIP_VALUE = "computational_less_power"


# =========================
# LOAD SUBSAMPLE
# =========================
def load_subsample(path):
    G = nx.read_edgelist(path, nodetype=int)
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    return G


# =========================
# SAFE HELPERS
# =========================
def safe_pearson(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return pearsonr(x, y)[0]


def safe_triangle_density(G):
    n = G.number_of_nodes()
    if n < 3:
        return np.nan
    triangles = sum(nx.triangles(G).values()) / 3
    denom = n * (n - 1) * (n - 2) / 6
    return triangles / denom if denom > 0 else np.nan


def ball_growth_proxy(G, n_samples=50, max_radius=5):
    nodes = np.random.choice(list(G.nodes()), min(n_samples, G.number_of_nodes()), replace=False)
    growth = []

    for v in nodes:
        dist = nx.single_source_shortest_path_length(G, v)
        for r in range(1, max_radius):
            b_r = sum(d <= r for d in dist.values())
            b_r1 = sum(d <= r + 1 for d in dist.values())
            if b_r > 0:
                growth.append(b_r1 / b_r)

    return np.mean(growth) if growth else np.nan


def gromov_delta_proxy(G, n_samples=500):
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


# =========================
# OBSERVABLES
# =========================
def calculate_observables(G):
    skip_expensive = G.number_of_nodes() > NODE_THRESHOLD

    degrees = dict(G.degree())
    deg_vals = list(degrees.values())

    core = nx.core_number(G)
    kmax = max(core.values())
    core_fraction = sum(1 for k in core.values() if k == kmax) / G.number_of_nodes()

    communities = nx.community.louvain_communities(G, seed=RANDOM_SEED)
    comm_sizes = [len(c) for c in communities]

    # Edge Jaccard (sampled)
    edges = list(G.edges())
    sample_idx = np.random.choice(len(edges), min(1000, len(edges)), replace=False)
    jacc = []
    for i in sample_idx:
        u, v = edges[i]
        Nu, Nv = set(G.neighbors(u)), set(G.neighbors(v))
        union = len(Nu | Nv)
        jacc.append(len(Nu & Nv) / union if union > 0 else 0)

    obs = {
        "avg_degree": np.mean(deg_vals),
        "degree_var": np.var(deg_vals),
        "max_degree": max(deg_vals),
        "kcore_depth": kmax,
        "core_periphery": core_fraction,
        "global_clustering": nx.transitivity(G),
        "avg_local_clustering": nx.average_clustering(G),
        "deg_clust_corr": safe_pearson(
            deg_vals,
            list(nx.clustering(G).values())
        ),
        "triangle_density": safe_triangle_density(G),
        "square_density": np.mean(list(nx.square_clustering(G).values())),
        "modularity": nx.community.modularity(G, communities),
        "community_size_var": np.var(comm_sizes),
        "edge_jaccard_mean": np.mean(jacc),
        "edge_jaccard_var": np.var(jacc)
    }

    # Expensive geometry-like proxies
    if skip_expensive:
        obs["ball_growth"] = SKIP_VALUE
        obs["gromov_delta"] = SKIP_VALUE
    else:
        obs["ball_growth"] = ball_growth_proxy(G)
        obs["gromov_delta"] = gromov_delta_proxy(G)

    return obs


# =========================
# MAIN
# =========================
def main():
    subsample_dir = Path("twitch_subsamples")
    files = sorted(subsample_dir.glob("subsample_*.edgelist"))

    rows = []
    for i, f in enumerate(files):
        G = load_subsample(f)
        obs = calculate_observables(G)
        obs["realization_id"] = i
        rows.append(obs)

    df = pd.DataFrame(rows)

    # Keep only numeric columns for downstream statistics
    df_numeric = df.select_dtypes(include=[np.number])

    outdir = Path("twitch_ensemble")
    outdir.mkdir(exist_ok=True)

    df_numeric.to_csv(outdir / "twitch_numeric_observables.csv", index=False)
    print("Saved numeric observable table")


if __name__ == "__main__":
    main()
