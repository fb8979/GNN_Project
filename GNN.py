import argparse
import torch
torch.backends.mkldnn.enabled = False
import GNNExperiment

parser = argparse.ArgumentParser(description="Run GNN Experiment")

parser.add_argument('--num_graphs', type=int, default=250, help="Number of graphs to generate")
parser.add_argument('--num_nodes_min', type=int, default=50, help="Minimum number of nodes per graph")
parser.add_argument('--num_nodes_max', type=int, default=100, help="Maximum number of nodes per graph")
parser.add_argument('--graph_type', type=str, default='barabasi_albert',
                    choices=['erdos_renyi', 'barabasi_albert', 'watts_strogatz'], help="Type of graph")
parser.add_argument('--feature_type', type=str, default='pagerank',
                    choices=['structural_role', 'homophily', 'pagerank'], help="Node feature type")
parser.add_argument('--edge_prob', type=float, default=0.1, help="Edge probability (only for some graph types)")

args = parser.parse_args()

# Configuration
dataset_config = {
    'num_graphs': args.num_graphs,
    'num_nodes_range': (args.num_nodes_min, args.num_nodes_max),
    'graph_type': args.graph_type,
    'feature_type': args.feature_type,
    'edge_prob': args.edge_prob,
    'seed': 42
}

model_configs = {
    'GCN': {
        'hidden_channels': 64,
        'out_channels': 1,
        'num_layers': 3,
        'dropout': 0.5
    },
    'GAT': {
        'hidden_channels': 64,
        'out_channels': 1,
        'num_layers': 3,
        'heads': 4,
        'dropout': 0.5
        },
    'GIN': {
        'hidden_channels': 64,
        'out_channels': 1,
        'num_layers': 3,
        'dropout': 0.5
        }
    }

training_config = {
    'batch_size': 32,
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'epochs': 200,
    'patience': 20
    }

# Run experiment
experiment = GNNExperiment.GNNExperiment(
    dataset_config=dataset_config,
    model_configs=model_configs,
    training_config=training_config,
    seed=42
    )

experiment.run()