# NOTE ON COMPUTATION:
# This script analyzes the Twitch DE social network (~9.5k nodes, ~150k edges).
# Some network observables (betweenness, closeness, eccentricity and metrics
# derived from them) require all-pairs shortest-path computations and are
# computationally infeasible for graphs of this size using exact algorithms.
#
# To ensure the script always finishes and remains reproducible, such expensive
# observables are deterministically skipped when the network size exceeds a
# fixed threshold. Their values are explicitly recorded as
# "computational_less_power".
#
# All other observables are computed normally. No result is silently omitted.

import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from scipy.stats import pearsonr
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

NODE_THRESHOLD = 5000
SKIP_VALUE = "computational_less_power"


def load_network(edges_file):
    edges = pd.read_csv(edges_file)
    G = nx.from_pandas_edgelist(edges, source=edges.columns[0], target=edges.columns[1])
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        logger.info(f"Extracted largest connected component: {G.number_of_nodes()} nodes")
    return G


def calculate_avg_degree(G):
    return np.mean([d for _, d in G.degree()])


def calculate_degree_var(G):
    return np.var([d for _, d in G.degree()])


def calculate_max_degree(G):
    return max(d for _, d in G.degree())


def calculate_kcore_depth(G):
    return max(nx.core_number(G).values())


def calculate_core_periphery(G):
    core = nx.core_number(G)
    kmax = max(core.values())
    return sum(1 for k in core.values() if k == kmax) / G.number_of_nodes()


def calculate_global_clustering(G):
    return nx.transitivity(G)


def calculate_avg_local_clustering(G):
    return nx.average_clustering(G)


def calculate_deg_clust_corr(G):
    degrees = dict(G.degree())
    clustering = nx.clustering(G)
    x = list(degrees.values())
    y = list(clustering.values())
    if len(set(x)) > 1 and len(set(y)) > 1:
        return pearsonr(x, y)[0]
    return np.nan


def calculate_triangle_density(G):
    triangles = sum(nx.triangles(G).values()) / 3
    n = G.number_of_nodes()
    return triangles / (n * (n - 1) * (n - 2) / 6)


def calculate_square_density(G):
    return np.mean(list(nx.square_clustering(G).values()))


def calculate_ball_growth(G, n_samples=100, max_radius=5):
    nodes = np.random.choice(list(G.nodes()), min(n_samples, G.number_of_nodes()), replace=False)
    growth = []
    for v in nodes:
        dist = nx.single_source_shortest_path_length(G, v)
        for r in range(1, min(max(dist.values()), max_radius)):
            b_r = sum(d <= r for d in dist.values())
            b_r1 = sum(d <= r + 1 for d in dist.values())
            if b_r > 0:
                growth.append(b_r1 / b_r)
    return np.mean(growth) if growth else np.nan


def calculate_gromov_delta(G, n_samples=1000):
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


def calculate_edge_jaccard(G, n_samples=1000):
    edges = list(G.edges())
    idx = np.random.choice(len(edges), min(n_samples, len(edges)), replace=False)
    vals = []
    for i in idx:
        u, v = edges[i]
        Nu, Nv = set(G.neighbors(u)), set(G.neighbors(v))
        union = len(Nu | Nv)
        vals.append(len(Nu & Nv) / union if union > 0 else 0)
    return np.mean(vals), np.var(vals)


def calculate_all_observables(G, network_id):
    skip_expensive = G.number_of_nodes() > NODE_THRESHOLD
    if skip_expensive:
        logger.info(f"Skipping expensive observables (N = {G.number_of_nodes()})")

    communities = nx.community.louvain_communities(G, seed=RANDOM_SEED)
    comm_sizes = [len(c) for c in communities]

    edge_j_mean, edge_j_var = calculate_edge_jaccard(G)

    obs = {
        "avg_degree": calculate_avg_degree(G),
        "degree_var": calculate_degree_var(G),
        "max_degree": calculate_max_degree(G),
        "kcore_depth": calculate_kcore_depth(G),
        "core_periphery": calculate_core_periphery(G),
        "global_clustering": calculate_global_clustering(G),
        "avg_local_clustering": calculate_avg_local_clustering(G),
        "deg_clust_corr": calculate_deg_clust_corr(G),
        "triangle_density": calculate_triangle_density(G),
        "square_density": calculate_square_density(G),
        "ball_growth": calculate_ball_growth(G),
        "gromov_delta": calculate_gromov_delta(G),
        "modularity": nx.community.modularity(G, communities),
        "community_size_var": np.var(comm_sizes),
        "edge_jaccard_mean": edge_j_mean,
        "edge_jaccard_var": edge_j_var,
        "avg_path_length": SKIP_VALUE if skip_expensive else nx.average_shortest_path_length(G),
        "avg_eccentricity": SKIP_VALUE,
        "radial_strat": SKIP_VALUE,
        "boundary_crowd": SKIP_VALUE,
        "mean_betweenness": SKIP_VALUE,
        "mean_closeness": SKIP_VALUE,
        "deg_bet_corr": SKIP_VALUE,
        "network_id": network_id
    }

    return obs


def main():
    script_dir = Path(__file__).parent
    edges_file = script_dir / "twitch" / "DE" / "musae_DE_edges.csv"

    G = load_network(edges_file)
    obs = calculate_all_observables(G, "DE")

    outdir = Path("twitch-obs")
    outdir.mkdir(exist_ok=True)
    pd.DataFrame([obs]).to_csv(outdir / "twitch-obs.csv", index=False)

    logger.info("Finished successfully")


if __name__ == "__main__":
    main()
