from torch_geometric.data import Data, Dataset
from typing import List, Tuple
import numpy as np
import torch
from tqdm import tqdm

class SyntheticGraphDataset(Dataset):
    """PyTorch Geometric Dataset for synthetic graphs with learnable tasks"""
    
    def __init__(
        self,
        num_graphs: int = 1000,
        num_nodes_range: Tuple[int, int] = (30, 50),
        graph_type: str = 'erdos_renyi',
        feature_type: str = 'structural_role',
        edge_prob: float = 0.15,
        seed: int = 42
    ):
        """
        Args:
            num_graphs: Number of graphs to generate
            num_nodes_range: (min, max) range for number of nodes
            graph_type: Type of graph topology
                - 'erdos_renyi': Random graph
                - 'barabasi_albert': Scale-free graph
                - 'watts_strogatz': Small-world graph
            feature_type: Type of node features/task
                - 'structural_role': Local features -> Betweenness centrality
                - 'pagerank': Local features -> PageRank score
                - 'homophily': Class probabilities -> Community strength
            edge_prob: Edge probability (for Erdos-Renyi)
            seed: Random seed for reproducibility
        """
        super().__init__()
        
        self.num_graphs = num_graphs
        self.num_nodes_range = num_nodes_range
        self.graph_type = graph_type
        self.feature_type = feature_type
        self.edge_prob = edge_prob
        self.seed = seed
        
        # Validate inputs
        self._validate_config()
        
        # Import generators
        from GraphGenerator import GraphGenerator
        from NodeFeatureGenerator import NodeFeatureGenerator
        
        self.graph_generator = GraphGenerator(seed=seed)
        self.feature_generator = NodeFeatureGenerator(seed=seed)
        
        # Generate all graphs
        self.data_list = self._generate_all_graphs()
    
    def _validate_config(self):
        """Validate configuration parameters"""
        valid_graph_types = ['erdos_renyi', 'barabasi_albert', 'watts_strogatz']
        valid_feature_types = ['structural_role', 'pagerank', 'homophily']
        
        if self.graph_type not in valid_graph_types:
            raise ValueError(
                f"Invalid graph_type: '{self.graph_type}'. "
                f"Valid options: {valid_graph_types}"
            )
        
        if self.feature_type not in valid_feature_types:
            raise ValueError(
                f"Invalid feature_type: '{self.feature_type}'. "
                f"Valid options: {valid_feature_types}"
            )
        
        if self.graph_type == 'erdos_renyi':
            avg_nodes = (self.num_nodes_range[0] + self.num_nodes_range[1]) / 2
            min_edge_prob = np.log(avg_nodes) / avg_nodes
            if self.edge_prob < min_edge_prob * 0.8:
                print(f" Warning: edge_prob={self.edge_prob:.3f} may be too low!")
    
    def _generate_all_graphs(self) -> List[Data]:
        """Generate all graphs in the dataset"""
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        data_list = []
        
        print(f"\nGenerating {self.num_graphs} {self.graph_type} graphs")
        print(f"Feature type: {self.feature_type}")
        print(f"Node range: {self.num_nodes_range}")
        if self.graph_type == 'erdos_renyi':
            print(f"Edge probability: {self.edge_prob}")
        
        attempts = 0
        max_attempts = self.num_graphs * 3
        failures = {
            'no_edges': 0,
            'feature_error': 0,
            'validation_error': 0,
            'other': 0
        }
        
        with tqdm(total=self.num_graphs, desc="Generating graphs") as pbar:
            while len(data_list) < self.num_graphs and attempts < max_attempts:
                attempts += 1
                
                try:
                    # Random number of nodes
                    num_nodes = np.random.randint(
                        self.num_nodes_range[0],
                        self.num_nodes_range[1] + 1
                    )
                    
                    # Generate graph structure
                    if self.graph_type == 'erdos_renyi':
                        A = self.graph_generator.generate_erdos_renyi(num_nodes, self.edge_prob)
                    elif self.graph_type == 'barabasi_albert':
                        m = min(3, num_nodes - 1)
                        A = self.graph_generator.generate_barabasi_albert(num_nodes, m=m)
                    elif self.graph_type == 'watts_strogatz':
                        k = min(6, num_nodes - 1)
                        if k % 2 == 1:
                            k -= 1
                        k = max(2, k)
                        A = self.graph_generator.generate_watts_strogatz(num_nodes, k=k, p=0.3)
                    
                    # Validate graph has edges
                    if A.sum() == 0:
                        failures['no_edges'] += 1
                        continue
                    
                    # Check minimum connectivity
                    num_edges = int(A.sum() / 2)
                    if num_edges < num_nodes - 1:
                        failures['no_edges'] += 1
                        continue
                    
                    # Generate features and labels based on type
                    if self.feature_type == 'structural_role':
                        X, y = self.feature_generator.structural_role_based(A)
                    elif self.feature_type == 'pagerank':
                        X, y = self.feature_generator.pagerank_based(A)
                    elif self.feature_type == 'homophily':
                        X, y = self.feature_generator.homophily_based(A)
                    else:
                        raise ValueError(f"Unknown feature_type: {self.feature_type}")
                    
                    # Validate features
                    if np.isnan(X).any() or np.isnan(y).any():
                        failures['feature_error'] += 1
                        continue
                    
                    if np.isinf(X).any() or np.isinf(y).any():
                        failures['feature_error'] += 1
                        continue
                    
                    # Check reasonable ranges
                    if X.max() > 100 or y.max() > 100:
                        failures['validation_error'] += 1
                        continue
                    
                    # Check for all zeros (degenerate case)
                    if X.max() == 0 or y.max() == 0:
                        failures['validation_error'] += 1
                        continue
                    
                    # Convert to edge list format
                    edge_index = self.graph_generator.adjacency_to_edge_index(A)
                    
                    if edge_index.size(1) == 0:
                        failures['no_edges'] += 1
                        continue
                    
                    # Create PyG Data object
                    data = Data(
                        x=torch.FloatTensor(X),
                        edge_index=edge_index,
                        y=torch.FloatTensor(y),
                        num_nodes=num_nodes
                    )
                    
                    data_list.append(data)
                    pbar.update(1)
                    
                except Exception as e:
                    failures['other'] += 1
                    if attempts % 100 == 0:
                        print(f"\n  Debug: Error at attempt {attempts}: {str(e)[:50]}")
                    continue
        
        # Report results
        if len(data_list) == 0:
            print(f"Failure to generate any valid graphs")
            print(f"\nFailure breakdown after {attempts} attempts:")
            for reason, count in failures.items():
                if count > 0:
                    print(f"  - {reason}: {count}")
            print(f"\nConfiguration:")
            print(f"  - graph_type: {self.graph_type}")
            print(f"  - feature_type: {self.feature_type}")
            print(f"  - num_nodes_range: {self.num_nodes_range}")
            print(f"  - edge_prob: {self.edge_prob}")
            raise ValueError("Failed to generate dataset. Check configuration above.")
        
        success_rate = len(data_list) / attempts * 100
        print(f" Successfully generated {len(data_list)}/{self.num_graphs} graphs")
        print(f"   Total attempts: {attempts}")
        print(f"   Success rate: {success_rate:.1f}%")
        
        if failures['no_edges'] > 0:
            print(f"   Skipped (no edges): {failures['no_edges']}")
        if failures['feature_error'] > 0:
            print(f"   Skipped (feature errors): {failures['feature_error']}")
        if failures['validation_error'] > 0:
            print(f"   Skipped (validation errors): {failures['validation_error']}")
        if failures['other'] > 0:
            print(f"   Skipped (other errors): {failures['other']}")
        
        # Show sample statistics
        sample = data_list[0]
        print(f"\nSample graph statistics:")
        print(f"   Nodes: {sample.num_nodes}")
        print(f"   Edges: {sample.edge_index.size(1)}")
        print(f"   Features shape: {sample.x.shape}")
        print(f"   Feature range: [{sample.x.min():.3f}, {sample.x.max():.3f}]")
        print(f"   Target range: [{sample.y.min():.3f}, {sample.y.max():.3f}]")
        
        # Show feature-specific info
        if self.feature_type == 'structural_role':
            print(f"   Task: Predict betweenness centrality from local features")
        elif self.feature_type == 'pagerank':
            print(f"   Task: Predict PageRank from degree features")
        elif self.feature_type == 'homophily':
            print(f"   Task: Predict community strength from class probabilities")
        
        return data_list
    
    def len(self) -> int:
        """Return the number of graphs in the dataset"""
        return len(self.data_list)
    
    def get(self, idx: int) -> Data:
        """Get a graph by index"""
        return self.data_list[idx]