import networkx as nx

path = "email-Enron.txt"

G = nx.read_edgelist(path, nodetype=int)
G = G.to_undirected()

largest_cc = max(nx.connected_components(G), key=len)
G = G.subgraph(largest_cc).copy()

print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())
print("Connected:", nx.is_connected(G))


# OUTPUT
# Nodes: 33696
# Edges: 180811
# Connected: True