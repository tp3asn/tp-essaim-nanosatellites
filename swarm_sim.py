import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from numpy.random import binomial
import math 
from mpl_toolkits import mplot3d
from random import seed, choice, sample


#==============================================================================================

class Node:
    """
    Node class, representing a satellite in the swarm. 
    """
    
    def __init__(self, id, x=0.0, y=0.0, z=0.0):
        """
        Node object constructor
        
        Args:
            id (int): the ID number of the satellite (mandatory)
            x (float, optional): the x-coordinate of the satellite. Defaults to 0.0.
            y (float, optional): the y-coordinate of the satellite. Defaults to 0.0.
            z (float, optional): the z-coordinate of the satellite. Defaults to 0.0.
        """
        self.id = int(id)
        self.x = float(x)
        self.y = float(y) 
        self.z = float(z) 
        self.neighbors = [] # List(Node), list of neighbor nodes to the node
        self.group = -1 # Group ID to which belongs the node
        
    def __str__(self):
        """
        Node object descriptor
        
        Returns:
            str: a string description of the node
        """
        nb_neigh = len(self.neighbors)
        return f"Node ID {self.id} ({self.x},{self.y},{self.z}) has {nb_neigh} neighbor(s)\tGroup: {self.group}"
    
    #*************** Common operations ****************
    def add_neighbor(self, node):
        """
        Function to add a node to the neighbor list of the node unless it is already in its list.
        
        Args:
            node (Node): the node to add.
        """
        if node not in self.neighbors:
            self.neighbors.append(node)
        
    def compute_dist(self, node):
        """
        Function to compute the Euclidean distance between two nodes.
        
        Args:
            node (Node): the node to compute the distance with.

        Returns:
            float: the Euclidean distance between the two nodes.
        """
        return math.dist((self.x, self.y, self.z) , (node.x, node.y, node.z))
    
    def is_neighbor(self, node, connection_range=0):
        """
        Function to verify whether two nodes are neighbors or not, based on the connection range. 
        Either adds or removes the second node from the neighbor list of the first.
        
        Args:
            node (Node): the second node to analyse.
            connection_range (int, optional): the maximum distance between two nodes to establish a connection. Defaults to 0.

        Returns:
            int: 1 if neighbors, 0 if not.
        """
        if node.id != self.id:
            if self.compute_dist(node) <= connection_range:
                self.add_neighbor(node)
                return 1 
            self.remove_neighbor(node)
        return 0
    
    def remove_neighbor(self, node):
        """
        Function to remove a node from the neighbor list of the node unless it is not in its list.
        
        Args:
            node (Node): the node to remove
        """
        if node in self.neighbors:
            self.neighbors.remove(node)   
     
    def set_group(self, c):
        """
        Function to appoint a group ID to the node.

        Args:
            c (int): group ID.
        """
        self.group = c  
   
    
    #************************************** Sampling algorithms **************************************************
    def proba_walk(self, p:float, s=1, overlap=False):
        """
        Function to perform a probabilistic hop from the node to its neighbor(s), usually used with the Forest Fire algorithm (see Swarm object).
        Each neighbor node has a probability p to be chosen for the next hop.

        Args:
            p (float): the success probability between 0 and 1.
            s (int, optional): the random seed. Defaults to 1.
            overlap (bool, optional): if True, node groups are allowed to overlap. Defaults to False.

        Returns:
            list(Node): the list of neighbor nodes selected as next hops.
        """
        seed(s)
        search_list = self.neighbors
        if not overlap: # Restrain the search list to unassigned nodes
            search_list = [n for n in self.neighbors if n.group==-1]
        trial = binomial(1, p, len(search_list))
        nodes = [n for i,n in enumerate(search_list) if trial[i]==1] # Select the nodes that obtained a success above
        return nodes
    
    def random_group(self, clist, s=1):
        """
        Function to appoint a group ID chosen randomly from the input list.

        Args:
            clist (list(int)): the list of group IDs.
            s (int, optional): the random seed. Defaults to 1.
        """
        seed(s)
        self.set_group(choice(clist))
        
    def random_walk(self, s=1, overlap=False):
        """
        Function to perform a random walk from the current node. One of its neighbor nodes is chosen randomly as the next hop.

        Args:
            s (int, optional): the random seed for the experiment. Defaults to 1.
            overlap (bool, optional): if True, node groups are allowed to overlap. Defaults to False.

        Returns:
            Node: the neighbor node selected as the next hop.
        """
        seed(s)
        search_list = self.neighbors
        if not overlap: # Restrain the search list to unassigned nodes
            search_list = [n for n in self.neighbors if n.group==-1]
        return choice(search_list)
    
    
#==============================================================================================

class Swarm:
    """
    Swarm object, representing a swarm of nanosatellites.
    """
    
    def __init__(self, connection_range=0, nodes=[]):
        """
        Swarm object constructor
        
        Args:
            connection_range (int, optional): the maximum distance between two nodes to establish a connection. Defaults to 0.
            nodes (list, optional): list of Node objects within the swarm. Defaults to [].
        """
        self.connection_range = connection_range
        self.nodes = nodes 
        
    def __str__(self):
        """
        Swarm object descriptor
        
        Returns:
            str: the string description of the swarm
        """
        nb_nodes = len(self.nodes)
        return f"Swarm of {nb_nodes} node(s), connection range: {self.connection_range}"
    
    #*************** Common operations ***************
    def add_node(self, node:Node):
        """
        Function to add a node to the swarm unless it is already in.

        Args:
            node (Node): the node to add.
        """
        if node not in self.nodes:
            self.nodes.append(node)
            
    def distance_matrix(self):
        """
        Function to compute the Euclidean distance matrix of the swarm.

        Returns:
            list(list(float)): the 2-dimensional distance matrix formatted as matrix[node1][node2] = distance.
        """
        matrix = []
        for n1 in self.nodes:
            matrix.append([n1.compute_dist(n2) for n2 in self.nodes if n1.id != n2.id])
        return matrix
    
    def get_node_by_id(self, id:int):
        """
        Function to retrieve a Node object in the swarm from its node ID.

        Args:
            id (int): the ID of the node.

        Returns:
            Node: the Node object with the corresponding ID.
        """
        for node in self.nodes:
            if node.id == id:
                return node
            
    def neighbor_matrix(self, connection_range=None):
        """
        Function to compute the neighbor matrix of the swarm.
        If two nodes are neighbors (according to the given connection range), the row[col] equals to 1. Else 0.

        Args:
            connection_range (int, optional): the connection range of the swarm. Defaults to None.

        Returns:
            list(list(int)): the 2-dimensional neighbor matrix formatted as matrix[node1][node2] = neighbor.
        """
        matrix = []
        if not connection_range:
            connection_range=self.connection_range # Use the attribute of the Swarm object if none specified
        for node in self.nodes:
            matrix.append([node.is_neighbor(nb,connection_range) for nb in self.nodes])
        return matrix
        
    def remove_node(self, node:Node):
        """
        Function to remove a node from the swarm unless it is already out.

        Args:
            node (Node): the node to remove.
        """
        if node in self.nodes:
            self.nodes.remove(node)
        
    def reset_connection(self):
        """
        Function to empty the neighbor list of each node of the swarm.
        """
        for node in self.nodes:
            node.neighbors = []
            
    def reset_groups(self):
        """
        Function to reset the group ID to -1 for each node of the swarm.
        """
        for node in self.nodes:
            node.set_group(-1)
    
    def swarm_to_nxgraph(self):
        """
        Function to convert a Swarm object into a NetworkX Graph. See help(networkx.Graph) for more information.

        Returns:
            nx.Graph: the converted graph.
        """
        G = nx.Graph()
        G.add_nodes_from([n.id for n in self.nodes])
        for ni in self.nodes:
            for nj in self.nodes:
                if ni.is_neighbor(nj, self.connection_range)==1:
                    G.add_edge(ni.id,nj.id) 
        return G
    
    
    #*********************************** Division algorithms ***********************************************************
    def center_of_mass(self, nodes):
        # average of coordinates x, y, z
        x = sum(n.x for n in nodes) / len(nodes)
        y = sum(n.y for n in nodes) / len(nodes)
        z = sum(n.z for n in nodes) / len(nodes)
        # Returns the closest node to this center of mass
        return min(nodes, key=lambda n: (n.x - x)**2 + (n.y - y)**2 + (n.z - z)**2)



    def kMeans(self, k=10, max_iter=100):
        """
        Performs k-means clustering on a Swarm based on the (x, y, z) positions of the nodes.
    
        Args:
            k (int): Number of clusters to form.
            max_iter (int): Maximum number of iterations for the algorithm.
    
        Returns:
            dict: A dictionary of Swarm instances, one per cluster.
        """
        # 1. Random initialization of centroids (nodes)
        centroids = sample(self.nodes, k)
        for i, c in enumerate(centroids):
            c.set_group(i)
    
        for _ in range(max_iter):
            # 2. Assign each node to the closest centroid
            for node in self.nodes:
                closest = min(centroids, key=lambda c: node.compute_dist(c))
                node.set_group(closest.group)
    
            # 3. Update centroids (optional depending on your model)
            new_centroids = []
            for i in range(k):
                group_nodes = [n for n in self.nodes if n.group == i]
                if group_nodes:
                    # Compute the geometric center 
                    new_centroid = self.center_of_mass(group_nodes)
                    new_centroid.set_group(i)
                    new_centroids.append(new_centroid)
            centroids = new_centroids
    
        # 4.  Create Swarms as in RND (grouped by assigned cluster)
        swarms = {}
        for i in range(k):
            swarms[i] = Swarm(self.connection_range,
                              nodes=[n for n in self.nodes if n.group == i])
        return swarms

    def FFD(self, n=10, p=0.7, s=1, overlap=False):
        """
        Function to perform graph sampling by the Forest Fire algorithm. 
        In the initial phase, n nodes are selected as "fire sources". Then, the fire spreads to the neighbors with a probability of p.
        We finally obtain n samples defined as the nodes burned by each source.

        Args:
            n (int, optional): the initial number of sources. Defaults to 10.
            p (float, optional): the fire spreading probability. Defaults to 0.7.
            s (int, optional): the random seed. Defaults to 1.
            overlap (bool, optional): if True, node groups are allowed to overlap. Defaults to False.

        Returns:
            dict(int:Swarm): the dictionary of group IDs and their corresponding Swarm sample.
        """
        sources = sample(self.nodes, n) # Initial random sources
        swarms = {} # Dict(group ID:Swarm)
        for i,src in enumerate(sources): # Initialize swarms
            src.set_group(i)
            swarms[i] = Swarm(self.connection_range, nodes=[src])
        free_nodes = [n for n in self.nodes if n.group==-1]
        burning_nodes = sources
        next_nodes = []
        while free_nodes: # Spread paths from each burning node in parallel
            for bn in burning_nodes:
                if not free_nodes:
                    break
                free_neighbors = set(free_nodes).intersection(bn.neighbors)
                if free_neighbors: # At least one unassigned neighbor
                    nodes = bn.proba_walk(p, i, overlap) # Next node(s)
                else:
                    if free_nodes == []:
                        break
                    nodes = [self.random_jump(free_nodes)] # If no neighbor, perform random jump in the graph
                for n in nodes:
                    n.set_group(bn.group)
                    swarms[bn.group].add_node(n) 
                    free_nodes.remove(n)
                    next_nodes.append(n)
            burning_nodes = next_nodes
        return swarms
    
    def MIRW(self, n=10, s=1, overlap=False):
        """
        Function to perform graph sampling by the Multi-Dimensional Random Walk algorithm.
        In the initial phase, n nodes are selected as sources. Then they all perform random walks in parallel (see help(Node.random_walk) for
        more information). 
        We finally obtain n samples defined as the random walks from each source.

        Args:
            n (int, optional): the initial number of sources. Defaults to 10.
            s (int, optional): the random seed. Defaults to 1.
            overlap (bool, optional): if True, node groups are allowed to overlap. Defaults to False.

        Returns:
            dict(int:Swarm): the dictionary of group IDs and their corresponding Swarm sample.
        """
        sources = sample(self.nodes, n) # Initial random sources
        swarms = {} # Dict(group ID:Swarm)
        for i,src in enumerate(sources): # Initialize swarms
            src.set_group(i)
            swarms[i] = Swarm(self.connection_range, nodes=[src])
        free_nodes = [n for n in self.nodes if n.group==-1]
        while free_nodes: # Spread paths to desired length
            for k in swarms.keys():
                n_i = swarms[k].nodes[-1] # Current node
                free_neighbors = set(free_nodes).intersection(n_i.neighbors)
                if free_neighbors: # At least one unassigned neighbor
                    n_j = n_i.random_walk(i, overlap) # Next node
                else:
                    if free_nodes == []:
                        break
                    n_j = self.random_jump(free_nodes) # If no neighbor, perform random jump in the graph
                n_j.set_group(n_i.group)
                swarms[k].add_node(n_j) 
                free_nodes.remove(n_j)
        return swarms
    

    def RND(self, n , s=1):
        """
        Function to perform graph sampling by the Random Node Sampling algorithm.
        Each node choses a random group ID from the list given as parameter.

        Args:
            clist (list(int)): list of group IDs. Defaults to range(10).
            s (int, optional): random seed. Defaults to 1.

        Returns:
            dict(int:Swarm): the dictionary of group IDs and their corresponding Swarm sample.
        """
        swarms = {}
        for i, node in enumerate(self.nodes):
            node.random_group(range(n), s*i)
        for i in range(n): # Separate into swarms
            swarms[i] = Swarm(self.connection_range,
                                nodes=[n for n in self.nodes if n.group==i])
        return swarms
    
    def random_jump(self,search_list, s=1):
        """
        Function to choose a new node in the graph by performing a random jump.

        Args:
            s (int, optional): the random seed. Defaults to 1.
    

        Returns:
            Node: the randomly chosen node.
        """
        seed(s)
        return choice(search_list)

    
def rmse(data, ref=None):
    """
    This function calculates the Root Mean Square Error (RMSE) between the observed distribution and a reference value.

    Parameters:
    data (list or numpy array): A list or numpy array containing the observed data points.
    ref (float, optional): A reference value to compare the observed distribution with. Defaults to the mean of the observed data.

    Returns:
    float: The RMSE value, which represents the standard deviation of the differences between the observed data and the reference value.

    Example:
    >>> data = [1, 2, 3, 4, 5]
    >>> ref = 3
    >>> rmse(data, ref)
    0.8164965809277461
    """
    if ref is None:
        ref = np.mean(data)
    errors = [(e - ref) ** 2 for e in data]
    ratio = sum(errors) / len(data)
    return np.sqrt(ratio)
            
