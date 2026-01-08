# ================================================================
# Controlled subsampling of the email-Enron network
# ================================================================
#
# The email-Enron network is sparse, heavy-tailed, and exhibits a
# pronounced core–periphery structure. Uniform random node sampling
# almost always produces disconnected or weakly connected subgraphs,
# which makes ensemble-based analysis unstable and unreliable.
#
# To avoid this, we adopt a two-stage subsampling strategy designed
# to guarantee connectivity while preserving the network’s internal
# geometry:
#
# (1) k-core restriction
#     We first extract a k-core of the original network. The k-core
#     isolates the structurally active backbone of the graph, removing
#     peripheral nodes that do not participate in sustained interaction.
#     This step is necessary to ensure that large connected samples
#     exist at all.
#
# (2) Connected growth sampling
#     From the k-core, each subsample is constructed by a randomized
#     breadth-first expansion starting from a randomly chosen seed
#     node. Nodes are added only through existing edges until the
#     target size is reached.
#
#     This guarantees:
#       - exact connectivity of every subsample,
#       - fixed sample size across realizations,
#       - bounded runtime with no rejection loops.
#
# Each subsample represents a finite-size realization of the same
# underlying communication geometry. Variability across realizations
# reflects sampling noise rather than structural fragmentation.
#
# This ensemble is used downstream to compute observable vectors,
# correlation matrices, and spectral decompositions, allowing direct
# comparison with Twitch and synthetic hyperbolic ensembles under a
# consistent statistical framework.
# ================================================================


import numpy as np
import networkx as nx
from pathlib import Path
from collections import deque

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

N_SUBSAMPLE = 3000
N_REALIZATIONS = 20
MIN_CORE_K = 5


def load_network(path):
    G = nx.read_edgelist(path, nodetype=int)
    G = G.to_undirected()
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    return G


def extract_sampling_core(G, k):
    core = nx.k_core(G, k)
    if core.number_of_nodes() < N_SUBSAMPLE:
        raise RuntimeError("k-core too small")
    return core


def connected_core_sample(G, n):
    start = np.random.choice(list(G.nodes()))
    visited = {start}
    queue = deque([start])

    while queue and len(visited) < n:
        v = queue.popleft()
        neighbors = list(G.neighbors(v))
        np.random.shuffle(neighbors)
        for u in neighbors:
            if u not in visited:
                visited.add(u)
                queue.append(u)
                if len(visited) == n:
                    break

    if len(visited) < n:
        raise RuntimeError("Failed to reach target size")

    return G.subgraph(visited).copy()


def generate_subsamples(G, n_nodes, n_realizations):
    return [connected_core_sample(G, n_nodes) for _ in range(n_realizations)]


def main():
    base = Path(__file__).parent
    edges_file = base / "email-Enron.txt"

    G = load_network(edges_file)
    G_core = extract_sampling_core(G, MIN_CORE_K)

    subsamples = generate_subsamples(G_core, N_SUBSAMPLE, N_REALIZATIONS)

    outdir = base / "email_subsamples"
    outdir.mkdir(exist_ok=True)

    for i, H in enumerate(subsamples):
        nx.write_edgelist(H, outdir / f"subsample_{i}.edgelist", data=False)

    print(len(subsamples), "connected k-core realizations saved")


if __name__ == "__main__":
    main()
