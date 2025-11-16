from typing import Tuple
import numpy as np
import networkx as nx

class NodeFeatureGenerator:
    """Generator for learnable synthetic node features"""
    
    def __init__(self, seed: int = None):
        """
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
    
    @staticmethod
    def structural_role_based(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate features based on local structure, predict global role
        
        Task: Predict betweenness centrality (global) from local features
        This is a good learnable task - requires multi-hop aggregation
        
        Args:
            A: Adjacency matrix (num_nodes x num_nodes)
            
        Returns:
            X: Local structural features (num_nodes x 3)
            y: Betweenness centrality (num_nodes x 1)
        """
        num_nodes = A.shape[0]
        
        # Feature 1: Degree (local connectivity)
        degree = A.sum(axis=1)
        degree_norm = degree / (degree.max() + 1e-8)
        
        # Feature 2: Local clustering coefficient
        clustering = np.zeros(num_nodes)
        for i in range(num_nodes):
            neighbors = np.where(A[i] > 0)[0]
            if len(neighbors) >= 2:
                # Count triangles
                subgraph = A[np.ix_(neighbors, neighbors)]
                triangles = subgraph.sum() / 2
                possible = len(neighbors) * (len(neighbors) - 1) / 2
                clustering[i] = triangles / possible if possible > 0 else 0
        
        # Feature 3: Average neighbor degree
        neighbor_degree_avg = np.zeros(num_nodes)
        for i in range(num_nodes):
            neighbors = np.where(A[i] > 0)[0]
            if len(neighbors) > 0:
                neighbor_degree_avg[i] = degree[neighbors].mean()
        neighbor_degree_avg_norm = neighbor_degree_avg / (neighbor_degree_avg.max() + 1e-8)
        
        # Stack local features
        X = np.column_stack([degree_norm, clustering, neighbor_degree_avg_norm])
        
        # Target: Betweenness centrality (requires global graph knowledge)
        G = nx.from_numpy_array(A)
        
        try:
            betweenness = nx.betweenness_centrality(G)
            y = np.array([betweenness[i] for i in range(num_nodes)])
        except:
            # Fallback for disconnected graphs
            y = degree / (num_nodes - 1 + 1e-8)
        
        y = y.reshape(-1, 1)
        
        # Normalize target to [0, 1]
        if y.max() > y.min():
            y = (y - y.min()) / (y.max() - y.min())
        
        return X.astype(np.float32), y.astype(np.float32)
    
    @staticmethod
    def pagerank_based(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate features based on PageRank
        
        Args:
            A: Adjacency matrix
            
        Returns:
            X: Node features (num_nodes x feature_dim)
            y: Node labels (num_nodes x 1)
        """
        num_nodes = A.shape[0]
        G = nx.from_numpy_array(A)
        
        # Feature: degree
        degree = A.sum(axis=1, keepdims=True)
        X = degree / (degree.max() + 1e-8)
        
        # Label: PageRank score
        try:
            pagerank = nx.pagerank(G)
            y = np.array([pagerank[i] for i in range(num_nodes)]).reshape(-1, 1)
            y = y / (y.max() + 1e-8)
        except:
            # Fallback if graph is disconnected
            y = X.copy()
        
        return X.astype(np.float32), y.astype(np.float32)

    @staticmethod
    def homophily_based(A: np.ndarray, num_classes: int = 5, 
                       homophily_strength: float = 0.7, 
                       iterations: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate features with homophily (similar neighbors)
        
        Nodes start with random class probabilities, then smooth based on neighbors.
        Models social phenomena where connected nodes tend to be similar.
        
        Args:
            A: Adjacency matrix (num_nodes x num_nodes)
            num_classes: Number of latent classes/communities
            homophily_strength: How much neighbors influence similarity (0-1)
            iterations: Number of smoothing iterations
            
        Returns:
            X: Node features (num_nodes x num_classes) - class probability distribution
            y: Node labels (num_nodes x 1) - dominant class assignment (for prediction)
        """
        num_nodes = A.shape[0]
        
        # Initialize with random class probabilities (Dirichlet distribution)
        class_probs = np.random.dirichlet(np.ones(num_classes), size=num_nodes)
        
        # Iterative smoothing (homophily effect)
        for _ in range(iterations):
            new_probs = class_probs.copy()
            
            for node in range(num_nodes):
                degree = A[node].sum()
                
                if degree > 0:
                    # Get average neighbor probabilities
                    neighbor_probs = (A[node].reshape(-1, 1) * class_probs).sum(axis=0) / degree
                    
                    # Mix own distribution with neighbor distribution
                    new_probs[node] = (
                        (1 - homophily_strength) * class_probs[node] +
                        homophily_strength * neighbor_probs
                    )
                    
                    # Re-normalize to valid probability distribution
                    new_probs[node] /= new_probs[node].sum()
            
            class_probs = new_probs
        
        # Features: full probability distribution
        X = class_probs
        
        # Label: "community strength" - how strongly node belongs to dominant class
        # Influenced by both random initialization and network smoothing
        dominant_class_prob = class_probs.max(axis=1)
        
        # Factor in network position
        degree_normalised = A.sum(axis=1) / (A.sum(axis=1).max() + 1e-8)
        
        # Community strength = dominant probability * network centrality
        y = (dominant_class_prob * (0.7 + 0.3 * degree_normalised)).reshape(-1, 1)
        
        # Normalize target
        if y.max() > y.min():
            y = (y - y.min()) / (y.max() - y.min())
        
        return X.astype(np.float32), y.astype(np.float32)