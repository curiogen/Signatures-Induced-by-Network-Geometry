import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from scipy.stats import pearsonr

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

NODE_THRESHOLD = 5000


def safe_pearson(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return pearsonr(x, y)[0]


def triangle_density_proxy(G, n_samples=2000):
    nodes = list(G.nodes())
    tri = 0
    tot = 0
    for _ in range(n_samples):
        u = np.random.choice(nodes)
        Nu = set(G.neighbors(u))
        if len(Nu) < 2:
            continue
        v, w = np.random.choice(list(Nu), 2, replace=False)
        tot += 1
        if G.has_edge(v, w):
            tri += 1
    return tri / tot if tot > 0 else np.nan


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


def global_clustering_proxy(G, n_samples=3000):
    nodes = list(G.nodes())
    closed = 0
    total = 0
    for _ in range(n_samples):
        u = np.random.choice(nodes)
        Nu = list(G.neighbors(u))
        if len(Nu) < 2:
            continue
        v, w = np.random.choice(Nu, 2, replace=False)
        total += 1
        if G.has_edge(v, w):
            closed += 1
    return closed / total if total > 0 else np.nan


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


def compute(G):
    deg = np.array([d for _, d in G.degree()])
    core = nx.core_number(G)
    kmax = max(core.values())

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

    return {
        "avg_degree": deg.mean(),
        "degree_variance": deg.var(),
        "max_degree": deg.max(),
        "kcore_depth": kmax,
        "core_fraction": sum(v == kmax for v in core.values()) / G.number_of_nodes(),
        "global_clustering": global_clustering_proxy(G),
        "avg_local_clustering": np.mean(clust_vals),
        "degree_clustering_corr": safe_pearson(deg, clust_vals),
        "triangle_density": triangle_density_proxy(G),
        "square_density": square_density_proxy(G),
        "modularity": nx.community.modularity(G, communities),
        "community_size_variance": np.var(comm_sizes),
        "edge_jaccard_mean": np.mean(jacc),
        "edge_jaccard_variance": np.var(jacc),
        "ball_growth": ball_growth_proxy(G),
        "gromov_delta": gromov_delta_proxy(G)
    }


base = Path(__file__).parent
rows = []

for i, f in enumerate(sorted((base / "citation_subsamples").glob("*.edgelist"))):
    G = nx.read_edgelist(f, nodetype=int)
    G = G.to_undirected()
    G.remove_edges_from(nx.selfloop_edges(G))

    obs = compute(G)
    obs["realization_id"] = i
    rows.append(obs)

outdir = base / "citation_ensemble"
outdir.mkdir(exist_ok=True)

pd.DataFrame(rows).to_csv(outdir / "citation_numeric_observables.csv", index=False)
print("Saved citation observables")
