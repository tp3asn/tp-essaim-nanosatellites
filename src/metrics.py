import networkx as nx 
from itertools import combinations
from networkx.algorithms.simple_paths import all_simple_paths
from functools import lru_cache
from swarm_sim import *
from collections import defaultdict
import random 
import networkx as nx
from networkx.algorithms.simple_paths import all_simple_paths
import numpy as np


#****************************** Metrics for hraph characterization ****************************************
def average_degree(G) :
     """ 
     Compute the average node degree of the graph
     Args :
         G(nx.Graph) : the graph to analyze
     Returns :
         float :the average degree of the graph
     """
     return sum(dict(G.degree()).values()) / G.number_of_nodes()
     
def average_clustering_coeff(G) :
    """
    Computes the average clustering coefficient of the graph.

    Args:
        G (nx.Graph): The input graph.

    Returns:
        float: Average clustering coefficient, or 0.0 if the graph has fewer than 2 nodes.
    """
    if G.number_of_nodes() < 2:
        return 0.0
    values = nx.clustering(G).values()
    return np.mean(list(values)) if values else 0.0

def avg_strength(G) :
    """
    Computes the average strength of nodes in the graph.
    For unweighted graphs, this is equivalent to average degree.

    Args:
        G (nx.Graph): The input graph.

    Returns:
        float: Average strength.
    """   
    node_strengths = G.degree()  # equivalent to weight = 1

    strengths = [s for _, s in node_strengths]
    return np.mean(strengths) if strengths else 0.0
     
def graph_density(G) :
    """
    Computes the density of the graph.

    Args:
        G (nx.Graph): The input graph.

    Returns:
        float: Graph density (between 0 and 1).
    """
    return nx.density(G)
    
def graph_diameter(G: nx.Graph) -> int:
    try:
        return nx.diameter(G)
    except nx.exception.NetworkXError:
        return float('inf')  # disconnected graph

def sample_size(G: nx.Graph) -> int:
    """
    returns the number of nodes in the graoh
    """
    return G.number_of_nodes()

def k_vicinity_distribution(G: nx.Graph, k: int = 2) -> dict:
    """
    Retourne la taille de la k-vicinity de chaque nœud.
    """
    result = {}
    for node in G.nodes:
        nodes_k = nx.single_source_shortest_path_length(G, node, cutoff=k)
        result[node] = len(nodes_k) - 1  # exclude the node itself
    return result

def inter_contact_time(graph_series: list) -> dict:
    """
    computes the % dof time during which each pair (u,v) is connected.

    Args:
        graph_series (list): List of graphs G_t (NetworkX) at different times.

    Returns:
        dict: {(u, v): % of time connected}
    """
    from collections import defaultdict
    counter = defaultdict(int)
    total = len(graph_series)

    for G in graph_series:
        for u, v in combinations(G.nodes, 2):
            if G.has_edge(u, v):
                counter[tuple(sorted((u, v)))] += 1

    return {pair: (connected / total) * 100 for pair, connected in counter.items()}

    
def jain_index(values):
    values = np.array(values)
    if np.sum(values) == 0:
        return 0
    return (np.sum(values) ** 2) / (len(values) * np.sum(values ** 2))

def flow_number(G: nx.Graph, group_dict: dict) -> dict:
    """
    Computes:
    - The total reference flow count (non-divided graph)
    - The theoretical flow after division (formula 4.1)

    Args:
        G (nx.Graph): Full graph (undirected, weighted or not)
        group_dict (dict): {node_id: group_id} mapping nodes to their groups

    Returns:
        dict:
            - flux_ref: int = number of edges in G
            - flux_div: int = number of flows by formula (4.1)
    """
    #number of edges in the global graph
    flux_ref = G.number_of_edges()

    # Group nodes by group ID
    groupes = defaultdict(list)
    for node, group in group_dict.items():
        groupes[group].append(node)
        
    x = len(groupes)  # nb of groups
    flux_inter = x * (x - 1)  # x(x-1) communications between the x groups
   
    flux_intra = 0
    for nodes in groupes.values():
        ni = len(nodes)
        if ni >= 2:  # ignore very small groups
            flux_intra += ni * (ni - 1) // 2

    flux_div = flux_inter + flux_intra

    return {
        "flux_ref": flux_ref,
        "flux_div": flux_div
    }

#***************************** metrics to evaluate graph Robustness ***********************************************

def flow_robustness(G: nx.Graph, divided=False, group_dict=None) -> float:
    """
    Computes the Flow Robustness of graph G.
    
    - If `divided=False`: global robustness (all node pairs)
    - If `divided=True`: intra-group robustness only (based on group_dict)

    Returns:
        float: proportion of connected pairs in the chosen mode
    """
    nodes = list(G.nodes)
    n = len(nodes)

    if n < 2:
        return 1.0  # Trivial case : 0 or 1 node → max robustness

    if not divided:
        # GLOBAL CASE : all node pairs
        total_pairs = n * (n - 1) // 2
        connected_pairs = sum(1 for u, v in combinations(nodes, 2) if nx.has_path(G, u, v))
        return connected_pairs / total_pairs

    else:
        # DIVIDED CASE : intra-group only
        if group_dict is None:
            raise ValueError("group_dict must be provided when divided=True")
        
        groups = {}
        for node, group in group_dict.items():
            if node in G: 
                groups.setdefault(group, []).append(node)
            

        total_pairs = 0
        connected_pairs = 0

        for group_nodes in groups.values():
            group_nodes_in_G = [n for n in group_nodes if n in G]
            subG = G.subgraph(group_nodes_in_G)
            m = len(group_nodes_in_G)
            if m < 2:
                continue
            pairs = m * (m - 1) // 2
            total_pairs += pairs
            connected_pairs += sum(1 for u, v in combinations(group_nodes_in_G, 2) if nx.has_path(subG, u, v))

        return connected_pairs / total_pairs if total_pairs > 0 else 0.0
import networkx as nx

def routing_cost(G: nx.Graph, divided=False, group_dict=None) -> float:
    """
    Computes the total routing cost:
    - Global case (divided=False): sum of distances between all connected pairs.
    - Divided case (divided=True): distances within each group only.
    
    Args:
        G (nx.Graph): the complete graph
        divided (bool): True for divided graph, and false for the global graph
        group_dict (dict): {node_id: group_id}, required if divided=True
    
    Returns:
        float: routing cost
    """
    total_cost = 0

    try:
        length_dict = dict(nx.all_pairs_shortest_path_length(G))
    except nx.NetworkXError:
        return float('inf')

    if not divided:
        # GLONAL CASE : all pairs are connected
        for u in G.nodes:
            for v in G.nodes:
                if u != v:
                    try:
                        total_cost += length_dict[u][v]
                    except KeyError:
                        continue
    else:
        if group_dict is None:
            raise ValueError("group_dict must be provided if divided = True")

        # Group nodes by group
        groups = {}
        for node, group in group_dict.items():
            if node in G:
                groups.setdefault(group, []).append(node)

        for group_nodes in groups.values():
            group_nodes_in_G = [n for n in group_nodes if n in G]
            for u in group_nodes_in_G:
                for v in group_nodes_in_G:
                    if u != v:
                        try:
                            total_cost += length_dict[u][v]
                        except KeyError:
                            continue

    return total_cost


def network_efficiency(G: nx.Graph, divided=False, group_dict=None) -> float:
    """
    Computes the network efficiency t(G), between 0 and 1.
    - Global case: average efficiency across all connected pairs.
    - Divided case: efficiency within each group only.
    """
    try:
        length_dict = dict(nx.all_pairs_shortest_path_length(G))
    except nx.NetworkXError:
        return 0.0

    if not divided:
        # GLOBAL CASE
        n = len(G)
        if n < 2:
            return 1.0
        total_efficiency = 0
        for u in G.nodes:
            for v in G.nodes:
                if u != v:
                    try:
                        d = length_dict[u][v]
                        total_efficiency += 1 / d
                    except KeyError:
                        continue
        return total_efficiency / (n * (n - 1))

    else:
        # DIVIDED CASE
        if group_dict is None:
            raise ValueError("group_dict must be provided if divided=True")

        # Groups only the nodes present in the graph
        groups = {}
        for node, group in group_dict.items():
            if node in G:
                groups.setdefault(group, []).append(node)

        total_efficiency = 0
        total_pairs = 0

        for group_nodes in groups.values():
            group_nodes_in_G = [n for n in group_nodes if n in G]
            m = len(group_nodes_in_G)
            if m < 2:
                continue
            total_pairs += m * (m - 1)
            for u in group_nodes_in_G:
                for v in group_nodes_in_G:
                    if u != v:
                        try:
                            d = length_dict[u][v]
                            total_efficiency += 1 / d
                        except KeyError:
                            continue

        return total_efficiency / total_pairs if total_pairs > 0 else 0.0




#*************************************** Metrics to Evaluate Graph Resilience ************************************


def path_redundancy(G: nx.Graph, max_extra_length: int = 1, divided=False, group_dict=None) -> float:
    """
    Computes path redundancy Ψt(G): average number of shortest paths between pairs.
    """
    nodes = list(G.nodes)
    n = len(nodes)
    if n < 2:
        return 0.0

    total = 0
    count = 0

    if not divided:
        pairs = [(u, v) for u in nodes for v in nodes if u < v]
    else:
        if group_dict is None:
            raise ValueError("group_dict must be provided if divided=True")
        pairs = [
            (u, v) for u in nodes for v in nodes
            if u < v and group_dict.get(u) == group_dict.get(v)
        ]

    for u, v in pairs:
        try:
            paths = list(nx.all_shortest_paths(G, source=u, target=v))
            total += len(paths)
            count += 1
        except nx.NetworkXNoPath:
            continue

    return total / count if count > 0 else 0.0




def pair_disparity_fast(shortest_paths):
    """
    Computes the average pairwise disparity between sets of intermediate nodes in shortest paths.
    """
    if len(shortest_paths) <= 1:
        return 0.0
    max_elem = len(shortest_paths[0]) - 2  # sans u et v
    if max_elem == 0:
        return 0.0

    intermediates = [set(p[1:-1]) for p in shortest_paths]
    disparity = 0
    count = 0
    for i in range(len(intermediates)):
        for j in range(i + 1, len(intermediates)):
            c = len(intermediates[i].intersection(intermediates[j]))
            disparity += 1 - (c / max_elem)
            count += 1
    return disparity / count if count > 0 else 0.0

def path_disparity(G, group_dict=None, divided=False, max_pairs=200, max_paths_per_pair=20):
    """
    Computes the disparity of shortest paths in the graph or within subgroups.
    
    Args :
        G : graph NetworkX
        group_dict : {node_id: group_id},required if divided=True
        divided : bool,to evaluate intra-groupe
        max_pairs : nb max of pairs (u,v) to evaluate
        max_paths_per_pair : limit of paths per pair (for simplification or the simulation will take long)
        
    Returns :
        float : average disparity
    """
    if divided:
        groups = {}
        for node, group in group_dict.items():
            groups.setdefault(group, []).append(node)
        node_pairs = []
        for group_nodes in groups.values():
            node_pairs += list(combinations(group_nodes, 2))
    else:
        node_pairs = list(combinations(G.nodes, 2))

    node_pairs = random.sample(node_pairs, min(max_pairs, len(node_pairs)))

    disparities = []
    for u, v in node_pairs:
        try:
            paths = list(nx.all_shortest_paths(G, u, v, weight="weight"))
            if len(paths) < 2 or len(paths) > max_paths_per_pair:
                continue
            d = pair_disparity_fast(paths)
            disparities.append(d)
        except:
            continue

    return sum(disparities) / len(disparities) if disparities else 0.0


def node_criticality(G: nx.Graph, divided=False, group_dict=None):
    """
    Computes the criticality BCt(i) of each node:
    - Global case: betweenness centrality over the full graph.
    - Divided case: intra-group centrality only.

    Returns:
        dict(node_id: criticity)
    """
    n = len(G)
    if n < 2:
        return {node: 0.0 for node in G.nodes}

    if not divided:
        # GLOBAL CASE
        norm_factor = 1 / (n * (n - 1))
        centrality = nx.betweenness_centrality(G, normalized=True)
        BC = {node: value * norm_factor for node, value in centrality.items()}
        return BC

    else:
        if group_dict is None:
            raise ValueError("group_dict must be provided if divided=True")

        #DIVIDED CASE: we compute centrality in each subgroup
        BC = {node: 0.0 for node in G.nodes}
        groups = {}
        for node, group in group_dict.items():
            if node in G:  # <--- filter the nodes always present in the graph (non fails)
                groups.setdefault(group, []).append(node)

        for group_nodes in groups.values():
            subG = G.subgraph(group_nodes)
            m = len(subG)
            if m < 2:
                continue
            norm_factor = 1 / (m * (m - 1))
            centrality = nx.betweenness_centrality(subG, normalized=True)
            for node, value in centrality.items():
                BC[node] = value * norm_factor

        return BC

# we can change epsillon 
def critical_nodes(G, epsilon=1e-4, divided=False, group_dict=None):
    """
    Returns the percentage of critical nodes (BCt(i) > epsillon)
    """
    BC = node_criticality(G, divided=divided, group_dict=group_dict)
    critical = [node for node, value in BC.items() if value >= epsilon]
    return len(critical) * 100 / len(G.nodes)
    

