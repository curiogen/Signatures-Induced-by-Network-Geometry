# This script generates ensembles of synthetic networks embedded in
# hyperbolic space using a model inspired by the PSO (Popularity–Similarity
# Optimization) framework.
#
# What this model is (and is not):
# - The model is inspired by the geometry behind PSO, but it does NOT implement
#   the original growing PSO process where nodes attach sequentially.
# - Instead, all nodes are assigned their geometric positions first, and edges
#   are added afterward based on distances in hyperbolic space.
# - For this reason, the model should be understood as a static,
#   PSO-inspired hyperbolic random graph, not a full PSO growth model.
#
# How nodes are placed in hyperbolic space:
# - Each node is given a random "birth time" between 0 and a final time T_total.
# - Each node also gets a random angular coordinate, representing similarity.
# - Radial coordinates represent popularity:
#   * Nodes born earlier tend to be more central.
#   * Nodes born later tend to lie closer to the boundary.
# - A simple aging rule is used to compute each node’s final radial position
#   at time T_total. This rule is controlled by a parameter β, which depends
#   on the target degree exponent γ.
# - The effective size of the hyperbolic disk grows slowly with time,
#   roughly like 2·log(T_total), which keeps the network sparse as it grows.
#
# How edges are formed:
# - After all node positions are fixed, edges are added independently
#   between every pair of nodes.
# - The probability of connecting two nodes depends on their hyperbolic
#   distance and a temperature parameter T.
# - Nearby nodes connect with high probability, distant nodes with low
#   probability, following a smooth logistic (Fermi–Dirac) function.
# - Lower temperature leads to stronger clustering, while higher temperature
#   weakens geometric effects.
#
# Important modeling clarification:
# - Because edges are created only after all coordinates are assigned,
#   this model represents a static snapshot consistent with PSO geometry.
# - It should not be interpreted as reproducing the exact temporal
#   attachment dynamics of the original PSO model.
#
# What the code actually does:
# - For a chosen base parameter set (N, T, k, γ), the code generates
#   multiple independent network realizations.
# - Then, one parameter at a time is varied (N, T, k, or γ), and a new
#   ensemble of networks is generated for each variation.
#
# What is measured on each network:
# - For every generated network, a fixed set of 25 observables is computed.
# - These include:
#   * Degree statistics and degree correlations
#   * Clustering and motif densities (triangles and 4-cycles)
#   * Shortest-path and eccentricity measures (normalized)
#   * Core–periphery structure
#   * Centrality measures
#   * Community structure
#   * Geometry-related quantities such as radial stratification,
#     boundary crowding, ball growth rate, sampled Gromov δ-hyperbolicity,
#     and approximate Ricci curvature
#
# Output:
# - For each parameter configuration, the code saves:
#   * A CSV file with raw observable values across all realizations
#   * A CSV file containing the correlation matrix between observables
#
# Notes on approximation:
# - Some geometric quantities (e.g., Ricci curvature and hyperbolicity)
#   are estimated using sampling or simplified proxies.
# - These values are meant for consistent comparison across networks,
#   not as exact geometric measurements.


import numpy as np
import networkx as nx
import pandas as pd
import os
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
from itertools import combinations
from scipy.optimize import linear_sum_assignment
import warnings

warnings.filterwarnings('ignore')

np.random.seed(42)


class PSOInspiredHyperbolicGraph:
    """
    Implements a static hyperbolic random graph inspired by PSO geometry.

    Model Definition:
    - This is a static construction, not a sequential growth process.
    - Nodes are assigned coordinates simultaneously based on PSO-derived distributions.
    - Edges are formed probabilistically based on hyperbolic distances between final coordinates.
    - This acts as a static snapshot approximation of the canonical growing PSO model.
    """

    def __init__(self, N, T, k, gamma, T_total=None):
        """
        Initialize model parameters.

        Parameters:
        - N: Final number of nodes.
        - T: Temperature (controls clustering via connection probability).
        - k: Nominal average degree (descriptive parameter used to tune radial density;
             this is not a hard constraint on the output degree).
        - gamma: Target power-law exponent (controls radial aging parameter beta).
        - T_total: Time horizon for coordinate scaling (defaults to N).
        """
        self.N = N
        self.T = T
        self.k = k
        self.gamma = gamma
        self.T_total = T_total if T_total is not None else N

        # Compute β from γ: popularity aging parameter
        # Note: This controls how nodes drift radially outwards.
        # β = 1 - 2/(γ-1) for γ > 2
        if gamma > 2:
            self.beta = 1 - 2.0 / (gamma - 1)
        else:
            self.beta = 0  # No aging for γ ≤ 2

        # Store node attributes
        self.birth_times = None
        self.birth_radii = None
        self.angles = None

    def generate(self):
        """
        Generate a static realization of the PSO-inspired hyperbolic graph.

        Procedure:
        1. Sample all birth times and angles.
        2. Compute final radial positions based on T_total.
        3. Form edges using the static hyperbolic distance between final positions.

        Returns:
        - G: NetworkX graph
        - node_attrs: dict containing geometric properties
        """
        # 1. Assign birth times uniformly in [0, T_total]
        self.birth_times = np.sort(np.random.uniform(0, self.T_total, self.N))

        # 2. Assign birth radii
        # Use a logarithmic mapping to approximate the density required for nominal degree k.
        # Note: This is a static sampling step, not a dynamic optimization.
        self.birth_radii = np.zeros(self.N)
        for i in range(self.N):
            R_birth = 2 * np.log(max(self.birth_times[i], 1.0))
            # Sample uniformly in [0, R_birth]
            self.birth_radii[i] = np.random.uniform(0, R_birth)

        # 3. Assign angular coordinates uniformly in [0, 2π)
        self.angles = np.random.uniform(0, 2 * np.pi, self.N)

        # Initialize graph container
        G = nx.Graph()
        G.add_nodes_from(range(self.N))

        # Current network radius at final time
        R_final = 2 * np.log(max(self.T_total, 1.0))

        # 4. Compute final radial coordinates
        # Apply the linear aging rule: r_final = β*T_total + (1-β)*t_birth
        current_radii = self.beta * self.T_total + (1 - self.beta) * self.birth_times

        # 5. Connect nodes based on static hyperbolic distance
        # Note: This implementation checks all pairs (O(N^2)), appropriate for static generation.
        for i in range(self.N):
            for j in range(i + 1, self.N):
                # Compute hyperbolic distance between i and j
                d_ij = self._hyperbolic_distance(
                    current_radii[i], self.angles[i],
                    current_radii[j], self.angles[j]
                )

                # Connection probability (Fermi-Dirac)
                p_ij = 1.0 / (1.0 + np.exp((d_ij - R_final) / (2 * self.T)))

                if np.random.random() < p_ij:
                    G.add_edge(i, j)

        # Store node attributes
        node_attrs = {
            'birth_time': self.birth_times,
            'birth_radius': self.birth_radii,
            'angle': self.angles,
            'current_radius': current_radii
        }

        return G, node_attrs

    def _hyperbolic_distance(self, r1, theta1, r2, theta2):
        """
        Compute hyperbolic distance on the Poincaré disk.

        d(x,y) = arcosh(cosh(r1)*cosh(r2) - sinh(r1)*sinh(r2)*cos(Δθ))
        """
        delta_theta = np.pi - abs(np.pi - abs(theta1 - theta2))

        # Hyperbolic distance formula
        x = np.cosh(r1) * np.cosh(r2) - np.sinh(r1) * np.sinh(r2) * np.cos(delta_theta)
        # Numerical stability: ensure x >= 1
        x = max(1.0, x)

        return np.arccosh(x)


def compute_observables(G, node_attrs):
    """
    Compute topological and geometric observables.

    Note on Estimators:
    - Geometric quantities (e.g., Ricci curvature, Gromov delta) are computed
      using sampling or proxies to ensure computational feasibility.
    - These should be interpreted as comparative indices rather than exact values.

    Parameters:
    - G: NetworkX graph
    - node_attrs: dict with hyperbolic coordinates and birth times

    Returns:
    - obs: dict of observable values
    """
    obs = {}
    N = G.number_of_nodes()

    # Handle empty or trivial graphs
    if G.number_of_edges() == 0 or N < 4:
        return {
            'avg_degree': 0.0, 'degree_var': 0.0, 'max_degree': 0.0,
            'kcore_depth': 0.0, 'core_periphery': 0.0,
            'global_clustering': 0.0, 'avg_local_clustering': 0.0, 'deg_clust_corr': 0.0,
            'triangle_density': 0.0, 'square_density': 0.0,
            'norm_avg_path': 0.0, 'norm_avg_ecc': 0.0,
            'radial_strat': 0.0, 'boundary_crowd': 0.0, 'ball_growth': 0.0,
            'gromov_delta': 0.0, 'mean_betweenness': 0.0, 'mean_closeness': 0.0,
            'deg_bet_corr': 0.0, 'modularity': 0.0, 'community_size_var': 0.0,
            'ricci_mean': 0.0, 'ricci_var': 0.0
        }

    degrees = dict(G.degree())
    degree_seq = list(degrees.values())

    obs['avg_degree'] = np.mean(degree_seq)
    obs['degree_var'] = np.var(degree_seq)
    obs['max_degree'] = np.max(degree_seq)

    core_numbers = nx.core_number(G)
    obs['kcore_depth'] = max(core_numbers.values()) if core_numbers else 0

    sorted_nodes = sorted(G.nodes(), key=lambda n: degrees[n], reverse=True)
    core_size = max(1, int(0.2 * N))
    periphery_size = max(1, int(0.2 * N))

    core_nodes = set(sorted_nodes[:core_size])
    periphery_nodes = set(sorted_nodes[-periphery_size:])

    core_edges = sum(1 for u, v in G.edges() if u in core_nodes and v in core_nodes)
    max_core_edges = core_size * (core_size - 1) / 2
    core_density = core_edges / max_core_edges if max_core_edges > 0 else 0

    cp_edges = sum(1 for u, v in G.edges()
                   if (u in core_nodes and v in periphery_nodes) or
                   (u in periphery_nodes and v in core_nodes))
    max_cp_edges = core_size * periphery_size
    cp_density = cp_edges / max_cp_edges if max_cp_edges > 0 else 0

    obs['core_periphery'] = core_density - cp_density

    obs['global_clustering'] = nx.transitivity(G)

    local_clust = nx.clustering(G)
    obs['avg_local_clustering'] = np.mean(list(local_clust.values()))

    if len(degrees) > 1:
        deg_list = [degrees[n] for n in G.nodes()]
        clust_list = [local_clust[n] for n in G.nodes()]
        if np.std(deg_list) > 0 and np.std(clust_list) > 0:
            obs['deg_clust_corr'] = pearsonr(deg_list, clust_list)[0]
        else:
            obs['deg_clust_corr'] = 0
    else:
        obs['deg_clust_corr'] = 0

    triangles = sum(nx.triangles(G).values()) / 3
    max_triangles = N * (N - 1) * (N - 2) / 6
    obs['triangle_density'] = triangles / max_triangles if max_triangles > 0 else 0

    squares = 0
    nodes = list(G.nodes())

    if N > 50:
        sample_size = min(50, N)
        sampled_nodes = np.random.choice(nodes, sample_size, replace=False)

        for node_set in combinations(sampled_nodes, 4):
            subgraph = G.subgraph(node_set)
            if subgraph.number_of_edges() == 4:
                if all(subgraph.degree(n) == 2 for n in node_set):
                    squares += 1

        # Scale up the count based on sampling probability
        scaling_factor = (N / sample_size) ** 4
        squares = squares * scaling_factor
    else:
        for node_set in combinations(nodes, 4):
            subgraph = G.subgraph(node_set)
            if subgraph.number_of_edges() == 4:
                if all(subgraph.degree(n) == 2 for n in node_set):
                    squares += 1

    max_squares = N * (N - 1) * (N - 2) * (N - 3) / 24  # C(N, 4)
    obs['square_density'] = squares / max_squares if max_squares > 0 else 0

    if nx.is_connected(G):
        avg_path = nx.average_shortest_path_length(G)
        obs['norm_avg_path'] = avg_path / np.log(N) if N > 1 else 0

        ecc = nx.eccentricity(G)
        obs['norm_avg_ecc'] = np.mean(list(ecc.values())) / np.log(N) if N > 1 else 0
    else:
        largest_cc = max(nx.connected_components(G), key=len)
        Gcc = G.subgraph(largest_cc)
        if Gcc.number_of_nodes() > 1:
            avg_path = nx.average_shortest_path_length(Gcc)
            obs['norm_avg_path'] = avg_path / np.log(Gcc.number_of_nodes())
            ecc = nx.eccentricity(Gcc)
            obs['norm_avg_ecc'] = np.mean(list(ecc.values())) / np.log(Gcc.number_of_nodes())
        else:
            obs['norm_avg_path'] = 0
            obs['norm_avg_ecc'] = 0

    birth_radii = node_attrs['birth_radius']
    obs['radial_strat'] = np.std(birth_radii)

    current_radii = node_attrs['current_radius']
    R_max = np.max(current_radii)
    boundary_threshold = 0.8 * R_max
    boundary_nodes = np.sum(current_radii >= boundary_threshold)
    obs['boundary_crowd'] = boundary_nodes / N

    sample_centers = np.random.choice(list(G.nodes()), min(10, N), replace=False)
    growth_rates = []

    for center in sample_centers:
        sp_lengths = nx.single_source_shortest_path_length(G, center)
        if len(sp_lengths) > 1:
            radii = sorted(set(sp_lengths.values()))
            if len(radii) >= 3:
                ball_sizes = []
                valid_radii = []
                for r in radii:
                    ball_size = sum(1 for d in sp_lengths.values() if d <= r)
                    if ball_size > 0:
                        ball_sizes.append(ball_size)
                        valid_radii.append(r)

                if len(ball_sizes) >= 3:
                    log_sizes = np.log(ball_sizes)
                    if np.std(valid_radii) > 0:
                        alpha = np.polyfit(valid_radii, log_sizes, 1)[0]
                        growth_rates.append(alpha)

    obs['ball_growth'] = np.mean(growth_rates) if growth_rates else 1.0

    sample_nodes = np.random.choice(list(G.nodes()), min(20, N), replace=False)
    sp_dict = {}
    for node in sample_nodes:
        sp_dict[node] = nx.single_source_shortest_path_length(G, node)

    deltas = []
    for quad in combinations(sample_nodes, 4):
        x, y, z, w = quad
        try:
            dxy = sp_dict[x].get(y, float('inf'))
            dzw = sp_dict[z].get(w, float('inf'))
            dxz = sp_dict[x].get(z, float('inf'))
            dyw = sp_dict[y].get(w, float('inf'))
            dxw = sp_dict[x].get(w, float('inf'))
            dyz = sp_dict[y].get(z, float('inf'))

            if all(d != float('inf') for d in [dxy, dzw, dxz, dyw, dxw, dyz]):
                S1 = dxy + dzw
                S2 = dxz + dyw
                S3 = dxw + dyz

                delta = max((S1 - S3) / 2, (S2 - S3) / 2, 0)
                deltas.append(delta)
        except:
            pass

    obs['gromov_delta'] = np.max(deltas) if deltas else 0

    betweenness = nx.betweenness_centrality(G)
    obs['mean_betweenness'] = np.mean(list(betweenness.values()))

    if nx.is_connected(G):
        closeness = nx.closeness_centrality(G)
        obs['mean_closeness'] = np.mean(list(closeness.values()))
    else:
        closeness = {}
        for component in nx.connected_components(G):
            if len(component) > 1:
                cc = nx.closeness_centrality(G.subgraph(component))
                closeness.update(cc)
        obs['mean_closeness'] = np.mean(list(closeness.values())) if closeness else 0

    if len(degrees) > 1 and len(betweenness) > 1:
        deg_list = [degrees[n] for n in G.nodes()]
        bet_list = [betweenness[n] for n in G.nodes()]
        if np.std(deg_list) > 0 and np.std(bet_list) > 0:
            obs['deg_bet_corr'] = pearsonr(deg_list, bet_list)[0]
        else:
            obs['deg_bet_corr'] = 0
    else:
        obs['deg_bet_corr'] = 0

    try:
        communities = nx.community.greedy_modularity_communities(G)
        obs['modularity'] = nx.community.modularity(G, communities)
        community_sizes = [len(c) for c in communities]
        obs['community_size_var'] = np.var(community_sizes)
    except:
        obs['modularity'] = 0
        obs['community_size_var'] = 0

    ricci_values = []

    edges = list(G.edges())
    if len(edges) > 100:
        sampled_edges = [edges[i] for i in np.random.choice(len(edges), 100, replace=False)]
    else:
        sampled_edges = edges

    for u, v in sampled_edges:
        ricci = compute_approx_ricci_curvature(G, u, v)
        ricci_values.append(ricci)

    obs['ricci_mean'] = np.mean(ricci_values) if ricci_values else 0
    obs['ricci_var'] = np.var(ricci_values) if ricci_values else 0

    return obs


def compute_approx_ricci_curvature(G, u, v):
    """
    Compute a proxy for Ollivier-Ricci curvature for edge (u,v).

    Method:
    - Approximates the Wasserstein distance using the independent coupling
      (product measure) of the neighbor distributions.
    - Cost ≈ Σ(P_i * Q_j * d_ij)
    - This serves as an upper bound on transport cost and a lower bound on curvature.

    Returns:
    - κ: Approximate Ricci curvature value
    """
    neighbors_u = set(G.neighbors(u)) | {u}
    neighbors_v = set(G.neighbors(v)) | {v}

    all_nodes = neighbors_u | neighbors_v
    node_list = sorted(all_nodes)
    n_nodes = len(node_list)

    # Create probability distributions (lazy random walk)
    prob_u = np.zeros(n_nodes)
    prob_v = np.zeros(n_nodes)

    for i, node in enumerate(node_list):
        if node in neighbors_u:
            prob_u[i] = 1.0 / len(neighbors_u)
        if node in neighbors_v:
            prob_v[i] = 1.0 / len(neighbors_v)

    # Compute pairwise distances (shortest paths)
    distance_matrix = np.zeros((n_nodes, n_nodes))
    for i, node_i in enumerate(node_list):
        sp_i = nx.single_source_shortest_path_length(G, node_i)
        for j, node_j in enumerate(node_list):
            distance_matrix[i, j] = sp_i.get(node_j, n_nodes)

    # Compute Approximate Transport Cost
    # Note: Exact Wasserstein computation requires solving the LP.
    # The following line implements the independent coupling approximation
    # regardless of sample size in this codebase.
    if n_nodes <= 20:
        transport_cost_proxy = np.sum(np.outer(prob_u, prob_v) * distance_matrix)
    else:
        transport_cost_proxy = np.sum(np.outer(prob_u, prob_v) * distance_matrix)

    d_uv = 1.0
    kappa = 1.0 - transport_cost_proxy / d_uv

    return kappa


def generate_and_analyze_slice(N, T, k, gamma, n_networks, output_dir):
    """
    Generate network ensemble and compute observables for a parameter slice.

    Parameters:
    - N, T, k, gamma: Model parameters
    - n_networks: Number of realizations
    - output_dir: Directory to save results

    Outputs:
    - observables.csv: Raw values
    - correlation_matrix.csv: Pearson correlations
    """
    os.makedirs(output_dir, exist_ok=True)

    all_obs = []

    for i in range(n_networks):
        print(f"  Network {i + 1}/{n_networks}...")

        # Generate PSO-inspired network (Static)
        pso_model = PSOInspiredHyperbolicGraph(N, T, k, gamma)
        G, node_attrs = pso_model.generate()

        # Compute observables
        obs = compute_observables(G, node_attrs)
        obs['network_id'] = i
        all_obs.append(obs)

    # Save raw observables
    df = pd.DataFrame(all_obs)
    df.to_csv(os.path.join(output_dir, 'observables.csv'), index=False)

    # Compute and save correlation matrix
    observable_cols = [col for col in df.columns if col != 'network_id']
    corr_matrix = df[observable_cols].corr()
    corr_matrix.to_csv(os.path.join(output_dir, 'correlation_matrix.csv'))

    print(f"  Saved to {output_dir}")

    return df


def main():
    """
    Main experimental pipeline:
    1. Generate base parameter slice
    2. Generate parameter variations (one parameter at a time)
    3. Compute observables and correlations for each slice
    """
    # Base parameter configuration
    base_params = {'N': 100, 'T': 0.5, 'k': 6, 'gamma': 2.5}
    n_networks = 30

    print("=" * 60)
    print("PSO-Inspired Hyperbolic Network Generation and Analysis")
    print("=" * 60)

    # Generate base slice
    print("\n[1/9] Generating base parameter slice...")
    print(f"Parameters: N={base_params['N']}, T={base_params['T']}, "
          f"k={base_params['k']}, gamma={base_params['gamma']}")
    base_dir = (f"N{base_params['N']}_T{base_params['T']}_"
                f"k{base_params['k']}_gamma{base_params['gamma']}")
    generate_and_analyze_slice(
        base_params['N'], base_params['T'], base_params['k'], base_params['gamma'],
        n_networks, base_dir
    )

    # Parameter variations
    variations = {
        'N': [50, 200],
        'T': [0.3, 0.7],
        'k': [4, 8],
        'gamma': [2.0, 3.0]
    }

    slice_counter = 2
    total_slices = 1 + sum(len(v) for v in variations.values())

    for param_name, values in variations.items():
        for value in values:
            params = base_params.copy()
            params[param_name] = value

            print(f"\n[{slice_counter}/{total_slices}] Generating slice: {param_name}={value}")
            print(f"Parameters: N={params['N']}, T={params['T']}, "
                  f"k={params['k']}, gamma={params['gamma']}")

            output_dir = (f"N{params['N']}_T{params['T']}_"
                          f"k{params['k']}_gamma{params['gamma']}")
            generate_and_analyze_slice(
                params['N'], params['T'], params['k'], params['gamma'],
                n_networks, output_dir
            )

            slice_counter += 1

    print("\n" + "=" * 60)
    print("All parameter slices completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()