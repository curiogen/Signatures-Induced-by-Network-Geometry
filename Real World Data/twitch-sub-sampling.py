import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

N_SUBSAMPLE = 3000      # target nodes per realization
N_REALIZATIONS = 20    # number of realizations
MAX_TRIES = 200        # safety cap to avoid infinite loops


def load_network(edges_file):
    edges = pd.read_csv(edges_file)
    G = nx.from_pandas_edgelist(
        edges,
        source=edges.columns[0],
        target=edges.columns[1]
    )
    # Work only with the giant component
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    return G


def generate_subsamples(G, n_nodes, n_realizations):
    nodes = np.array(list(G.nodes()))
    subsamples = []
    tries = 0

    while len(subsamples) < n_realizations and tries < MAX_TRIES:
        tries += 1

        sampled_nodes = np.random.choice(nodes, n_nodes, replace=False)
        H = G.subgraph(sampled_nodes).copy()

        # Always take the largest connected component
        if not nx.is_connected(H):
            H = H.subgraph(max(nx.connected_components(H), key=len)).copy()

        # Ensure size is reasonable
        if H.number_of_nodes() >= int(0.9 * n_nodes):
            subsamples.append(H)

    if len(subsamples) < n_realizations:
        raise RuntimeError(
            f"Only generated {len(subsamples)} subsamples. "
            f"Increase MAX_TRIES or reduce N_SUBSAMPLE."
        )

    return subsamples


def main():
    base = Path(__file__).parent
    edges_file = base / "twitch" / "DE" / "musae_DE_edges.csv"

    G = load_network(edges_file)
    subsamples = generate_subsamples(G, N_SUBSAMPLE, N_REALIZATIONS)

    outdir = Path("twitch_subsamples")
    outdir.mkdir(exist_ok=True)

    for i, H in enumerate(subsamples):
        nx.write_edgelist(
            H,
            outdir / f"subsample_{i}.edgelist",
            data=False
        )

    print(f"Saved {len(subsamples)} subsampled realizations")


if __name__ == "__main__":
    main()
