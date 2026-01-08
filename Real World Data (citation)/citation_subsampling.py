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

    G.remove_edges_from(nx.selfloop_edges(G))

    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()

    return G


def extract_core(G, k):
    core = nx.k_core(G, k)
    if core.number_of_nodes() < N_SUBSAMPLE:
        raise RuntimeError("k-core too small")
    return core


def connected_sample(G, n):
    start = np.random.choice(list(G.nodes()))
    visited = {start}
    queue = deque([start])

    while queue and len(visited) < n:
        v = queue.popleft()
        nbrs = list(G.neighbors(v))
        np.random.shuffle(nbrs)
        for u in nbrs:
            if u not in visited:
                visited.add(u)
                queue.append(u)
                if len(visited) == n:
                    break

    return G.subgraph(visited).copy()


base = Path(__file__).parent
G = load_network(base / "cit-HepPh.txt")
G = extract_core(G, MIN_CORE_K)

outdir = base / "citation_subsamples"
outdir.mkdir(exist_ok=True)

for i in range(N_REALIZATIONS):
    H = connected_sample(G, N_SUBSAMPLE)
    nx.write_edgelist(H, outdir / f"subsample_{i}.edgelist", data=False)

print("Saved", N_REALIZATIONS, "citation subsamples")
