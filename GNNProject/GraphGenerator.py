import numpy as np
import torch
import networkx as nx

class GraphGenerator:
    """Generator for synthetic graph datasets with various topologies"""
    
    def __init__(self, seed: int = 42):
        """
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    def generate_erdos_renyi(self, num_nodes: int, edge_prob: float) -> np.ndarray:
        """
        Generate Erdos-Renyi random graph
        
        Args:
            num_nodes: Number of nodes
            edge_prob: Probability of edge between any two nodes
            
        Returns:
            Adjacency matrix (num_nodes x num_nodes)
        """
        A = np.random.rand(num_nodes, num_nodes) < edge_prob
        A = A.astype(float)
        # Make undirected and remove self-loops
        A = np.maximum(A, A.T)
        np.fill_diagonal(A, 0)
        return A
    
    def generate_barabasi_albert(self, num_nodes: int, m: int = 3) -> np.ndarray:
        """
        Generate Barabasi-Albert scale-free graph
        
        Args:
            num_nodes: Number of nodes
            m: Number of edges to attach from new node
            
        Returns:
            Adjacency matrix
        """
        G = nx.barabasi_albert_graph(num_nodes, m, seed=self.seed)
        A = nx.to_numpy_array(G)
        return A
    
    def generate_watts_strogatz(self, num_nodes: int, k: int = 4, p: float = 0.3) -> np.ndarray:
        """
        Generate Watts-Strogatz small-world graph
        
        Args:
            num_nodes: Number of nodes
            k: Each node connected to k nearest neighbors
            p: Rewiring probability
            
        Returns:
            Adjacency matrix
        """
        G = nx.watts_strogatz_graph(num_nodes, k, p, seed=self.seed)
        A = nx.to_numpy_array(G)
        return A
    
    def adjacency_to_edge_index(self, A: np.ndarray) -> torch.Tensor:
        """
        Convert adjacency matrix to edge list format for PyG
        
        Args:
            A: Adjacency matrix (N x N)
            
        Returns:
            edge_index: Edge list tensor (2 x num_edges)
        """
        edges = np.nonzero(A)
        edge_index = torch.tensor(np.vstack([edges[0], edges[1]]), dtype=torch.long)
        return edge_index