import GNNExperiment

def main():
    """Main execution function"""
    
    # Configuration
    dataset_config = {
        'num_graphs': 100,
        'num_nodes_range': (50, 100),
        'graph_type': 'erdos_renyi', 
        'feature_type': 'pagerank',
        'edge_prob': 0.01,
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
    
    experiment = GNNExperiment.GNNExperiment(
        dataset_config=dataset_config,
        model_configs=model_configs,
        training_config=training_config,
        seed=42
    )
    
    experiment.run()

if __name__ == "__main__":
    main()