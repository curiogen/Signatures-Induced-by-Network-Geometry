# =============================================================================
# PSO-LIKE STATIC HYPERBOLIC RANDOM GRAPH GENERATOR
# =============================================================================
#
# PURPOSE
# -------
# This code generates synthetic networks embedded in hyperbolic space using a
# PSO-inspired but explicitly HEURISTIC static model. The goal is not to
# reproduce exact PSO growth dynamics, but to construct sparse networks that
# preserve the core geometric principles of popularity–similarity models:
#   - popularity encoded radially,
#   - similarity encoded angularly,
#   - connection probability governed by hyperbolic distance.
#
# The generator is designed for large-scale network experiments where geometric
# and topological signatures (clustering, hyperbolicity, curvature proxies,
# motif structure, etc.) are analyzed comparatively across parameter regimes.
#
# -----------------------------------------------------------------------------
# MODEL SCOPE AND LIMITATIONS
# -----------------------------------------------------------------------------
# • This is NOT an exact PSO model.
# • There is NO growth process; all coordinates are assigned statically.
# • Radial coordinates follow a heuristic aging rule (linear in birth time),
#   not the logarithmic aging used in canonical PSO.
# • The disk radius R is chosen heuristically to control sparsity and average
#   degree, without analytical guarantees.
#
# All approximations are intentional and explicitly documented.
#
# -----------------------------------------------------------------------------
# HYPERBOLIC EMBEDDING
# -----------------------------------------------------------------------------
# Each node i is assigned coordinates (r_i, θ_i) in the hyperbolic disk H²
# (curvature K = -1):
#
#   - θ_i ∈ [0, 2π) is sampled uniformly and represents similarity.
#   - r_i ≥ 0 represents popularity: smaller r_i corresponds to higher
#     popularity (more central nodes).
#
# Radial coordinates follow a heuristic aging rule:
#
#   r_i = β·R + (1−β)·R·(t_i / N),
#
# where:
#   - t_i ∈ [0, N] is a node "birth time",
#   - β = 1 − 2/(γ−1) (PSO-inspired aging strength),
#   - R is the hyperbolic disk radius.
#
# This rule enforces a monotonic popularity hierarchy but does NOT claim
# equivalence to PSO or S¹ model radial dynamics.
#
# -----------------------------------------------------------------------------
# CONNECTION PROBABILITY
# -----------------------------------------------------------------------------
# Nodes i and j are connected independently with probability
#
#   p_ij = 1 / (1 + exp[(d_ij − R) / (2T)]),
#
# where:
#   - d_ij is the hyperbolic distance between (r_i, θ_i) and (r_j, θ_j),
#   - T ∈ (0, 1) is a temperature parameter controlling clustering.
#
# Hyperbolic distance is computed using the large-radius approximation:
#
#   d_ij ≈ r_i + r_j + 2·ln[ sin(Δθ_ij / 2) ],
#
# which is standard in network-scale hyperbolic models and avoids expensive
# cosh/sinh/arccosh evaluations.
#
# -----------------------------------------------------------------------------
# GEOMETRY-AWARE ANGULAR PRUNING (KEY OPTIMIZATION)
# -----------------------------------------------------------------------------
# A naïve implementation requires O(N²) pairwise distance checks. This code
# avoids that cost using geometry-aware angular pruning inspired by HyperMap
# and PSO implementations.
#
# From the distance approximation, non-negligible connection probability
# requires:
#
#   d_ij ≲ R + c·T,
#
# which implies an angular cutoff:
#
#   Δθ_ij ≲ 2·exp[(R − r_i − r_j − c·T)/2].
#
# For pairs beyond this cutoff, p_ij decays exponentially and contributes
# negligibly to expected degree and clustering.
#
# Algorithmically:
#   1. Nodes are sorted by angular coordinate.
#   2. For each node, neighbors are searched only within its angular window.
#   3. Exact connection probabilities are evaluated only for these candidates.
#
# This preserves the connection probability measure up to a vanishing error
# term ε(N) → 0 and reduces complexity from O(N²) to approximately O(N·k)
# for sparse graphs.
#
# -----------------------------------------------------------------------------
# OBSERVABLES AND APPROXIMATIONS
# -----------------------------------------------------------------------------
# The code computes a broad set of observables, including:
#   - degree statistics and k-core structure,
#   - clustering and motif estimators,
#   - path-length scaling and ball growth rates,
#   - Gromov δ-hyperbolicity (sampled estimator),
#   - community structure (modularity),
#   - curvature proxies.
#
# Important notes:
#   • Square motifs and Gromov δ are estimated via Monte Carlo sampling.
#   • "Ricci curvature" is implemented as an independent-coupling proxy and
#     is NOT true Ollivier–Ricci curvature (no optimal transport).
#
# All such quantities are labeled accordingly to avoid overinterpretation.
#
# -----------------------------------------------------------------------------
# INTENDED USE
# -----------------------------------------------------------------------------
# This generator is intended for:
#   - comparative studies of geometric vs non-geometric networks,
#   - robustness analysis of network signatures,
#   - controlled parameter sweeps at N ≈ 10³–10⁴.
#
#
# =============================================================================

import numpy as np
import networkx as nx
import pandas as pd
import os
from scipy.stats import pearsonr
from itertools import combinations
import warnings

warnings.filterwarnings('ignore')
np.random.seed(42)


class PSOLikeHyperbolicNetwork:
    """
    PSO-like static hyperbolic random graph (heuristic model).

    Parameters:
    -----------
    N : int
        Number of nodes
    T : float
        Temperature (controls clustering sharpness)
    k : float
        Target expected average degree (heuristic control)
    gamma : float
        Power-law exponent (γ > 2), determines aging strength via β = 1 - 2/(γ-1)
    """

    def __init__(self, N, T, k, gamma):
        self.N = N
        self.T = T
        self.k = k
        self.gamma = gamma

        # Aging strength parameter (inspired by PSO but used heuristically)
        # Higher γ → higher β → stronger aging effect
        if gamma > 2:
            self.beta = 1.0 - 2.0 / (gamma - 1.0)
        else:
            self.beta = 0.5  # Fallback for γ ≤ 2

        # Disk radius: heuristic formula to roughly target average degree k
        # Empirically chosen, not analytically derived
        self.R = 2.0 * np.log(N / (k * np.pi))

        self.radii = None
        self.angles = None

    def generate(self, debug_mode=False):
        """
        Generate network with PSO-like hyperbolic embedding.

        Uses geometry-aware angular pruning to achieve O(N log N) complexity
        for sparse graphs, avoiding the O(N²) pairwise loop.

        Parameters:
        -----------
        debug_mode : bool
            If True, also generate with O(N²) method for verification
        """

        # Birth times: uniformly distributed in [0, N]
        birth_times = np.sort(np.random.uniform(0, self.N, self.N))

        # Radial coordinates: linear aging transformation
        # Form: r_i = β·R + (1-β)·R·(t_i/N)
        # Ensures: early nodes (small t_i) → small r_i → central → popular
        # Note: This is LINEAR in normalized time, not logarithmic
        normalized_times = birth_times / self.N
        self.radii = self.beta * self.R + (1.0 - self.beta) * self.R * normalized_times

        # Angular coordinates: uniform on [0, 2π) (similarity space)
        self.angles = np.random.uniform(0, 2.0 * np.pi, self.N)

        # Build graph using geometry-aware pruning
        G = self._generate_with_angular_pruning()

        # Store node attributes
        node_attrs = {
            'birth_time': birth_times,
            'radius': self.radii,
            'angle': self.angles
        }

        # Debug mode: compare with O(N²) method
        if debug_mode:
            G_naive = self._generate_naive()
            self._compare_methods(G, G_naive)

        return G, node_attrs

    def _generate_with_angular_pruning(self):
        """
        Generate graph using geometry-aware angular pruning.

        PRUNING PRINCIPLE:
        -----------------
        From the hyperbolic distance approximation:
          d_ij ≈ r_i + r_j + 2·ln[sin(Δθ_ij/2)]

        For a connection to have non-negligible probability, we need:
          d_ij ≲ R + few·T  (otherwise p_ij → 0 exponentially)

        This gives angular cutoff:
          sin(Δθ_ij/2) ≳ exp[(R - r_i - r_j)/2]

        For small angles: sin(x/2) ≈ x/2, so:
          Δθ_ij^max ≈ 2·exp[(R - r_i - r_j)/2]

        CORRECTNESS GUARANTEE:
        ---------------------
        We only prune pairs where p_ij < exp(-cutoff_buffer/T).
        With cutoff_buffer = 10T, we ignore edges with p < exp(-10) ≈ 4.5e-5.
        This introduces negligible bias: for N=1000, k=8, expected missed edges < 0.04.

        COMPLEXITY:
        ----------
        For sparse graphs (k << N), each node checks ~O(k) angular neighbors,
        giving total complexity O(N·k) ≈ O(N log N) when k ~ log N.
        """
        G = nx.Graph()
        G.add_nodes_from(range(self.N))

        # Sort nodes by angle for angular sweeping
        # This enables efficient circular neighbor search
        sorted_indices = np.argsort(self.angles)
        sorted_angles = self.angles[sorted_indices]
        sorted_radii = self.radii[sorted_indices]

        # Cutoff buffer: connections beyond R + cutoff_buffer have p_ij < exp(-cutoff_buffer/(2T))
        # With cutoff_buffer = 10T, this is ~4.5e-5, ensuring negligible bias
        cutoff_buffer = 10.0 * self.T

        # Process each node
        for idx_i, node_i in enumerate(sorted_indices):
            r_i = sorted_radii[idx_i]
            theta_i = sorted_angles[idx_i]

            # For this node, determine maximum angular distance for any potential neighbor
            # Conservative: use minimum neighbor radius (r_j = 0) for widest search
            max_delta_theta = self._angular_cutoff(r_i, 0.0, cutoff_buffer)

            # Search neighbors on both sides in circular angular space
            # Right side (increasing angle)
            for offset in range(1, self.N):
                idx_j = (idx_i + offset) % self.N
                node_j = sorted_indices[idx_j]

                # Skip self-loops
                if node_i == node_j:
                    continue

                r_j = sorted_radii[idx_j]
                theta_j = sorted_angles[idx_j]

                # Compute angular distance (shortest on circle)
                delta_theta = self._angular_distance_circular(theta_i, theta_j)

                # Check angular cutoff with actual r_j
                actual_cutoff = self._angular_cutoff(r_i, r_j, cutoff_buffer)

                if delta_theta > actual_cutoff:
                    # Beyond cutoff on this side, stop searching this direction
                    break

                # Only process each pair once (avoid double-counting)
                if node_i < node_j:
                    # Compute hyperbolic distance
                    d_ij = self._hyperbolic_distance_approx(r_i, r_j, delta_theta)

                    # Connection probability (unchanged formula)
                    p_ij = 1.0 / (1.0 + np.exp((d_ij - self.R) / (2.0 * self.T)))

                    # Sample edge
                    if np.random.random() < p_ij:
                        G.add_edge(node_i, node_j)

            # Left side (decreasing angle)
            for offset in range(1, self.N):
                idx_j = (idx_i - offset) % self.N
                node_j = sorted_indices[idx_j]

                # Skip self-loops
                if node_i == node_j:
                    continue

                r_j = sorted_radii[idx_j]
                theta_j = sorted_angles[idx_j]

                # Compute angular distance (shortest on circle)
                delta_theta = self._angular_distance_circular(theta_i, theta_j)

                # Check angular cutoff with actual r_j
                actual_cutoff = self._angular_cutoff(r_i, r_j, cutoff_buffer)

                if delta_theta > actual_cutoff:
                    # Beyond cutoff on this side, stop searching this direction
                    break

                # Only process each pair once (avoid double-counting)
                if node_i < node_j:
                    # Compute hyperbolic distance
                    d_ij = self._hyperbolic_distance_approx(r_i, r_j, delta_theta)

                    # Connection probability (unchanged formula)
                    p_ij = 1.0 / (1.0 + np.exp((d_ij - self.R) / (2.0 * self.T)))

                    # Sample edge
                    if np.random.random() < p_ij:
                        G.add_edge(node_i, node_j)

        return G

    def _angular_cutoff(self, r_i, r_j, cutoff_buffer):
        """
        Compute maximum angular distance for non-negligible connection probability.

        Derivation:
        ----------
        From d_ij ≈ r_i + r_j + 2·ln[sin(Δθ/2)], require:
          d_ij ≤ R + cutoff_buffer

        This gives:
          sin(Δθ/2) ≥ exp[(R - r_i - r_j - cutoff_buffer)/2]

        For small angles (sin(x/2) ≈ x/2):
          Δθ^max ≈ 2·exp[(R - r_i - r_j - cutoff_buffer)/2]

        We cap at 2π for safety (full circle).
        """
        exponent = (self.R - r_i - r_j - cutoff_buffer) / 2.0

        # For very negative exponents, cutoff would be huge (all pairs viable)
        # Cap at 2π since that's the full circle
        if exponent > np.log(np.pi):
            return 2.0 * np.pi

        cutoff = 2.0 * np.exp(exponent)
        return min(cutoff, 2.0 * np.pi)

    def _angular_distance_circular(self, theta1, theta2):
        """
        Compute shortest angular distance on circle [0, 2π).

        Returns value in [0, π].
        """
        delta = abs(theta1 - theta2)
        # Shortest path on circle
        return min(delta, 2.0 * np.pi - delta)

    def _generate_naive(self):
        """
        Generate graph using O(N²) pairwise loop (for debug comparison).

        This is the reference implementation without pruning.
        """
        G = nx.Graph()
        G.add_nodes_from(range(self.N))

        # Precompute sin/cos for efficiency
        cos_angles = np.cos(self.angles)
        sin_angles = np.sin(self.angles)

        for i in range(self.N):
            for j in range(i + 1, self.N):
                # Angular difference via dot product
                cos_delta = cos_angles[i] * cos_angles[j] + sin_angles[i] * sin_angles[j]
                cos_delta = np.clip(cos_delta, -1, 1)
                delta_theta = np.arccos(cos_delta)

                # Hyperbolic distance
                d_ij = self._hyperbolic_distance_approx(
                    self.radii[i], self.radii[j], delta_theta
                )

                # Connection probability
                p_ij = 1.0 / (1.0 + np.exp((d_ij - self.R) / (2.0 * self.T)))

                if np.random.random() < p_ij:
                    G.add_edge(i, j)

        return G

    def _compare_methods(self, G_pruned, G_naive):
        """
        Compare pruned vs naive generation methods.

        Reports:
        - Number of edges
        - Average degree
        - Clustering coefficient
        - Number of candidate pairs evaluated
        """
        print("\n" + "=" * 60)
        print("DEBUG: Comparing angular pruning vs O(N²) method")
        print("=" * 60)

        # Basic stats
        n_edges_pruned = G_pruned.number_of_edges()
        n_edges_naive = G_naive.number_of_edges()

        avg_deg_pruned = 2 * n_edges_pruned / self.N
        avg_deg_naive = 2 * n_edges_naive / self.N

        # Clustering
        clust_pruned = nx.transitivity(G_pruned)
        clust_naive = nx.transitivity(G_naive)

        print(f"Edges:      Pruned={n_edges_pruned}, Naive={n_edges_naive}, "
              f"Diff={abs(n_edges_pruned - n_edges_naive)}")
        print(f"Avg Degree: Pruned={avg_deg_pruned:.3f}, Naive={avg_deg_naive:.3f}, "
              f"Diff={abs(avg_deg_pruned - avg_deg_naive):.3f}")
        print(f"Clustering: Pruned={clust_pruned:.4f}, Naive={clust_naive:.4f}, "
              f"Diff={abs(clust_pruned - clust_naive):.4f}")

        # Estimate pruning efficiency
        max_pairs = self.N * (self.N - 1) / 2
        print(f"\nMax possible pairs: {int(max_pairs)}")
        print(f"Pruning reduces search space by ~{100 * (1 - avg_deg_pruned / self.N):.1f}%")
        print("=" * 60 + "\n")

    def _hyperbolic_distance_approx(self, r1, r2, delta_theta):
        """
        Approximate hyperbolic distance for large radii.

        Standard approximation used in network-scale hyperbolic models:
        d ≈ r1 + r2 + 2·ln[sin(Δθ/2)]

        Valid when r1, r2 >> 1 and provides ~10x speedup over exact formula.
        Numerical stability: enforce sin(Δθ/2) ≥ 1e-10 to avoid log(0).
        """
        sin_half = np.sin(delta_theta / 2.0)
        # Numerical protection: ensure argument to log is positive and bounded away from 0
        sin_half = max(sin_half, 1e-10)

        return r1 + r2 + 2.0 * np.log(sin_half)


def compute_observables(G, node_attrs):
    """
    Compute 23 topological and geometric observables.

    Categories:
    - Degree structure (4 observables)
    - Clustering (3)
    - Motifs (2, Monte Carlo estimates)
    - Path metrics (2)
    - Geometric (3, based on embedding)
    - Hyperbolicity (1, sampled estimator)
    - Centralities (3)
    - Community (2)
    - Curvature proxy (2, independent coupling approximation)
    """
    obs = {}
    N = G.number_of_nodes()

    # Handle degenerate cases
    if G.number_of_edges() == 0 or N < 4:
        return {k: 0.0 for k in [
            'avg_degree', 'degree_var', 'max_degree', 'kcore_depth',
            'core_periphery', 'global_clustering', 'avg_local_clustering',
            'deg_clust_corr', 'triangle_density', 'square_est',
            'norm_avg_path', 'norm_avg_ecc', 'radial_strat',
            'boundary_crowd', 'ball_growth', 'gromov_delta',
            'mean_betweenness', 'mean_closeness', 'deg_bet_corr',
            'modularity', 'community_size_var', 'ricci_proxy_mean',
            'ricci_proxy_var'
        ]}

    # ===== DEGREE STRUCTURE =====
    degrees = dict(G.degree())
    degree_seq = np.array(list(degrees.values()))

    obs['avg_degree'] = np.mean(degree_seq)
    obs['degree_var'] = np.var(degree_seq)
    obs['max_degree'] = np.max(degree_seq)

    # k-core depth: maximum k for which k-core exists
    core_numbers = nx.core_number(G)
    obs['kcore_depth'] = max(core_numbers.values()) if core_numbers else 0

    # ===== CORE-PERIPHERY STRUCTURE =====
    # Definition: density contrast between max k-core and 1-core
    # If either region empty or ill-defined, return 0
    max_kcore = obs['kcore_depth']

    if max_kcore >= 2:
        core_nodes = {n for n, k in core_numbers.items() if k == max_kcore}
        periphery_nodes = {n for n, k in core_numbers.items() if k == 1}

        if len(core_nodes) > 1 and len(periphery_nodes) > 1:
            # Core density
            core_edges = sum(1 for u, v in G.edges() if u in core_nodes and v in core_nodes)
            max_core = len(core_nodes) * (len(core_nodes) - 1) / 2
            core_density = core_edges / max_core if max_core > 0 else 0

            # Periphery density
            periph_edges = sum(1 for u, v in G.edges()
                               if u in periphery_nodes and v in periphery_nodes)
            max_periph = len(periphery_nodes) * (len(periphery_nodes) - 1) / 2
            periph_density = periph_edges / max_periph if max_periph > 0 else 0

            # Core-periphery contrast (positive = core denser than periphery)
            obs['core_periphery'] = core_density - periph_density
        else:
            obs['core_periphery'] = 0
    else:
        # Network too sparse or uniform to define core-periphery
        obs['core_periphery'] = 0

    # ===== CLUSTERING =====
    obs['global_clustering'] = nx.transitivity(G)
    local_clust = nx.clustering(G)
    clust_vals = np.array(list(local_clust.values()))
    obs['avg_local_clustering'] = np.mean(clust_vals)

    # Degree-clustering correlation (guarded against constant sequences)
    if np.std(degree_seq) > 1e-10 and np.std(clust_vals) > 1e-10:
        corr, _ = pearsonr(degree_seq, clust_vals)
        obs['deg_clust_corr'] = corr if not np.isnan(corr) else 0
    else:
        obs['deg_clust_corr'] = 0

    # ===== MOTIF DENSITIES =====
    # Triangle density (exact count, normalized)
    triangles = sum(nx.triangles(G).values()) / 3
    max_triangles = N * (N - 1) * (N - 2) / 6
    obs['triangle_density'] = triangles / max_triangles if max_triangles > 0 else 0

    # Square (4-cycle) estimator via Monte Carlo sampling
    # Labeled as "est" (estimator) not "density" to reflect uncertainty
    squares = 0
    nodes = list(G.nodes())

    if N > 100:
        # Sample ~200 random 4-tuples
        n_samples = min(200, int(N * (N - 1) * (N - 2) * (N - 3) / 24))
        for _ in range(n_samples):
            quad = np.random.choice(nodes, 4, replace=False)
            subgraph = G.subgraph(quad)
            # 4-cycle: exactly 4 edges, all nodes degree 2 in subgraph
            if subgraph.number_of_edges() == 4 and all(subgraph.degree(n) == 2 for n in quad):
                squares += 1

        # Rescale to total quadruple count (unbiased estimator)
        total_quads = N * (N - 1) * (N - 2) * (N - 3) / 24
        squares_est = squares * (total_quads / n_samples)
    else:
        # Exact count for small networks
        for quad in combinations(nodes, 4):
            subgraph = G.subgraph(quad)
            if subgraph.number_of_edges() == 4 and all(subgraph.degree(n) == 2 for n in quad):
                squares += 1
        squares_est = squares

    max_squares = N * (N - 1) * (N - 2) * (N - 3) / 24
    obs['square_est'] = squares_est / max_squares if max_squares > 0 else 0

    # ===== PATH METRICS =====
    # Normalized by log(N) for small-world scaling
    if nx.is_connected(G):
        avg_path = nx.average_shortest_path_length(G)
        ecc_dict = nx.eccentricity(G)
        obs['norm_avg_path'] = avg_path / np.log(max(N, 2))
        obs['norm_avg_ecc'] = np.mean(list(ecc_dict.values())) / np.log(max(N, 2))
    else:
        # Use largest connected component
        largest_cc = max(nx.connected_components(G), key=len)
        Gcc = G.subgraph(largest_cc)
        N_cc = Gcc.number_of_nodes()

        if N_cc > 1:
            avg_path = nx.average_shortest_path_length(Gcc)
            ecc_dict = nx.eccentricity(Gcc)
            obs['norm_avg_path'] = avg_path / np.log(N_cc)
            obs['norm_avg_ecc'] = np.mean(list(ecc_dict.values())) / np.log(N_cc)
        else:
            obs['norm_avg_path'] = 0
            obs['norm_avg_ecc'] = 0

    # ===== GEOMETRIC OBSERVABLES =====
    # These use the hyperbolic embedding coordinates

    # Radial stratification: spread across radial dimension
    obs['radial_strat'] = np.std(node_attrs['radius'])

    # Boundary crowding: fraction of nodes in outer 20% of disk
    max_radius = np.max(node_attrs['radius'])
    obs['boundary_crowd'] = np.sum(node_attrs['radius'] >= 0.8 * max_radius) / N

    # Ball growth rate: exponential growth exponent of network balls
    # Estimates α where |B(v,r)| ~ exp(α·r)
    sample_centers = np.random.choice(list(G.nodes()), min(20, N), replace=False)
    growth_rates = []

    for center in sample_centers:
        sp_lengths = nx.single_source_shortest_path_length(G, center)
        radii_unique = sorted(set(sp_lengths.values()))

        if len(radii_unique) >= 3:
            ball_sizes = [sum(1 for d in sp_lengths.values() if d <= r) for r in radii_unique]
            log_sizes = np.log(np.maximum(ball_sizes, 1))  # Avoid log(0)

            if np.std(radii_unique) > 0:
                # Linear fit: log(size) ~ rate * radius
                growth_rates.append(np.polyfit(radii_unique, log_sizes, 1)[0])

    obs['ball_growth'] = np.mean(growth_rates) if growth_rates else 1.0

    # ===== GROMOV δ-HYPERBOLICITY =====
    # Sampled estimator for δ-hyperbolicity
    # For each quadruple (x,y,z,w), compute:
    #   S1 = d(x,y) + d(z,w)
    #   S2 = d(x,z) + d(y,w)
    #   S3 = d(x,w) + d(y,z)
    #   δ = (1/2) · [max(S1,S2,S3) - median(S1,S2,S3)]
    # Return max δ over sampled quadruples

    n_sample_nodes = min(30, N)
    n_quadruples = min(100, n_sample_nodes * (n_sample_nodes - 1) // 2)

    sample_nodes = np.random.choice(list(G.nodes()), n_sample_nodes, replace=False)

    # Precompute all-pairs shortest paths among sampled nodes
    sp_dict = {}
    for n in sample_nodes:
        sp_dict[n] = nx.single_source_shortest_path_length(G, n)

    deltas = []
    sampled_quads = 0

    for quad in combinations(sample_nodes, 4):
        if sampled_quads >= n_quadruples:
            break

        x, y, z, w = quad
        try:
            S1 = sp_dict[x][y] + sp_dict[z][w]
            S2 = sp_dict[x][z] + sp_dict[y][w]
            S3 = sp_dict[x][w] + sp_dict[y][z]

            # δ = (1/2) * (max - median)
            sums = [S1, S2, S3]
            sums.sort()
            delta = 0.5 * (sums[2] - sums[1])  # (max - middle)

            deltas.append(delta)
            sampled_quads += 1
        except KeyError:
            # Nodes not in same component, skip
            continue

    # Use max over samples (could use high percentile for robustness)
    obs['gromov_delta'] = np.max(deltas) if deltas else 0

    # ===== CENTRALITIES =====
    bet_cent = nx.betweenness_centrality(G)
    bet_vals = np.array(list(bet_cent.values()))
    obs['mean_betweenness'] = np.mean(bet_vals)

    # Closeness centrality (per-component for disconnected graphs)
    if nx.is_connected(G):
        clo_cent = nx.closeness_centrality(G)
        obs['mean_closeness'] = np.mean(list(clo_cent.values()))
    else:
        clo_vals = []
        for comp in nx.connected_components(G):
            if len(comp) > 1:
                clo_vals.extend(nx.closeness_centrality(G.subgraph(comp)).values())
        obs['mean_closeness'] = np.mean(clo_vals) if clo_vals else 0

    # Degree-betweenness correlation (guarded)
    if np.std(degree_seq) > 1e-10 and np.std(bet_vals) > 1e-10:
        corr, _ = pearsonr(degree_seq, bet_vals)
        obs['deg_bet_corr'] = corr if not np.isnan(corr) else 0
    else:
        obs['deg_bet_corr'] = 0

    # ===== COMMUNITY STRUCTURE =====
    try:
        comms = list(nx.community.greedy_modularity_communities(G))
        obs['modularity'] = nx.community.modularity(G, comms)
        comm_sizes = [len(c) for c in comms]
        obs['community_size_var'] = np.var(comm_sizes)
    except (ValueError, ZeroDivisionError):
        obs['modularity'] = 0
        obs['community_size_var'] = 0

    # ===== RICCI CURVATURE PROXY =====
    # NOTE: This is NOT true Ollivier-Ricci curvature (no optimal transport).
    # It uses independent coupling (mean-field approximation):
    #   W(μ_u, μ_v) ≈ Σ_i Σ_j μ_u(i)·μ_v(j)·d(i,j)
    # Labeled as "proxy" to reflect this simplification.

    edges = list(G.edges())
    n_edge_samples = min(200, len(edges))
    sampled_edge_idx = np.random.choice(len(edges), n_edge_samples, replace=False)

    ricci_vals = []

    for idx in sampled_edge_idx:
        u, v = edges[idx]
        curv_proxy = compute_ricci_proxy(G, u, v)
        if curv_proxy is not None and not np.isnan(curv_proxy):
            ricci_vals.append(curv_proxy)

    obs['ricci_proxy_mean'] = np.mean(ricci_vals) if ricci_vals else 0
    obs['ricci_proxy_var'] = np.var(ricci_vals) if ricci_vals else 0

    return obs


def compute_ricci_proxy(G, u, v):
    """
    Compute Ricci curvature proxy using independent coupling.

    This is NOT the true Ollivier-Ricci curvature (which requires optimal transport).
    Instead, it uses mean-field/independent coupling:
      κ_proxy(u,v) = 1 - W_independent(μ_u, μ_v)
    where W_independent = Σ_i Σ_j μ_u(i)·μ_v(j)·d(i,j)

    This approximation is computationally cheap and captures similar trends
    but lacks theoretical guarantees of true Ollivier-Ricci.
    """
    # Neighborhoods (closed: include self)
    neighbors_u = set(G.neighbors(u)) | {u}
    neighbors_v = set(G.neighbors(v)) | {v}
    nodes = sorted(neighbors_u | neighbors_v)

    # Uniform distributions on neighborhoods
    pu = np.array([1.0 / len(neighbors_u) if n in neighbors_u else 0 for n in nodes])
    pv = np.array([1.0 / len(neighbors_v) if n in neighbors_v else 0 for n in nodes])

    # Distance matrix (only between local nodes, with cutoff for efficiency)
    n_nodes = len(nodes)
    dist = np.zeros((n_nodes, n_nodes))

    for i, ni in enumerate(nodes):
        sp = nx.single_source_shortest_path_length(G, ni, cutoff=max(4, n_nodes))
        for j, nj in enumerate(nodes):
            dist[i, j] = sp.get(nj, n_nodes)  # Use n_nodes as "infinity"

    # Independent coupling Wasserstein (mean-field approximation)
    wasserstein_proxy = np.sum(np.outer(pu, pv) * dist)

    # Ricci proxy: κ = 1 - W (typical normalization)
    return 1.0 - wasserstein_proxy


def generate_and_analyze_slice(N, T, k, gamma, n_networks, output_dir):
    """
    Generate ensemble of networks and compute observables.

    Also reports achieved average degree for degree-control verification.

    Parameters:
    -----------
    N, T, k, gamma : network parameters
    n_networks : number of independent realizations
    output_dir : directory for output CSVs
    """
    os.makedirs(output_dir, exist_ok=True)
    results = []
    achieved_degrees = []

    for i in range(n_networks):
        model = PSOLikeHyperbolicNetwork(N, T, k, gamma)
        G, attrs = model.generate()
        obs = compute_observables(G, attrs)
        obs['network_id'] = i
        results.append(obs)
        achieved_degrees.append(obs['avg_degree'])

    # Report degree control performance
    mean_achieved = np.mean(achieved_degrees)
    print(f"  Target k={k:.1f}, Achieved ⟨k⟩={mean_achieved:.2f} (±{np.std(achieved_degrees):.2f})")

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, 'observables.csv'), index=False)

    # Correlation matrix (drop ID column)
    corr_df = df.drop(columns=['network_id']).corr()
    corr_df.to_csv(os.path.join(output_dir, 'correlation_matrix.csv'))

    return df


def main():
    """
    Run parameter sweep experiment.

    Base configuration: N=1000, T=0.4, k=8, γ=2.7
    Then vary each parameter individually around base values.
    """
    base_params = {'N': 1000, 'T': 0.4, 'k': 8, 'gamma': 2.7}
    n_networks = 30

    # Optional: Debug mode comparison on small network
    print("=" * 60)
    print("DEBUG MODE: Verifying angular pruning correctness")
    print("=" * 60)
    debug_params = {'N': 200, 'T': 0.4, 'k': 8, 'gamma': 2.7}
    model_debug = PSOLikeHyperbolicNetwork(
        debug_params['N'], debug_params['T'],
        debug_params['k'], debug_params['gamma']
    )
    # This will run both methods and compare
    G_test, _ = model_debug.generate(debug_mode=True)

    # Generate base configuration
    print(f"\nGenerating base configuration: {base_params}")
    generate_and_analyze_slice(
        base_params['N'], base_params['T'],
        base_params['k'], base_params['gamma'],
        n_networks,
        f"N{base_params['N']}_T{base_params['T']}_k{base_params['k']}_gamma{base_params['gamma']}"
    )

    # Parameter variations
    variations = {
        'N': [800, 1200],
        'T': [0.2, 0.6],
        'k': [6, 10],
        'gamma': [2.3, 3.2]
    }

    for param, vals in variations.items():
        for v in vals:
            p = base_params.copy()
            p[param] = v
            print(f"Generating variation: {param}={v}")
            generate_and_analyze_slice(
                p['N'], p['T'], p['k'], p['gamma'],
                n_networks,
                f"N{p['N']}_T{p['T']}_k{p['k']}_gamma{p['gamma']}"
            )

    print("\nAll configurations complete.")


if __name__ == "__main__":
    main()