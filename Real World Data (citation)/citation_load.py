import networkx as nx
from pathlib import Path

path = Path(__file__).parent / "cit-HepPh.txt"

G = nx.read_edgelist(path, nodetype=int)
G = G.to_undirected()

if not nx.is_connected(G):
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()

print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())
print("Connected:", nx.is_connected(G))

# Nodes: 34401
# Edges: 420828
# Connected: True
