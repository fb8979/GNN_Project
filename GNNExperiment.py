import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from typing import Dict, Tuple, Optional
import numpy as np
from sklearn.model_selection import train_test_split
import os
import SyntheticGraphDataset
import GCNModel
import GATModel
import GINModel
import Trainer
from Visualiser import Visualiser

class GNNExperiment:
    """Main experiment class for comparing GNN architectures"""
    
    def __init__(
        self,
        dataset_config: Dict,
        model_configs: Dict[str, Dict],
        training_config: Dict,
        device: Optional[torch.device] = None,
        seed: int = 42
    ):
        """
        Args:
            dataset_config: Configuration for dataset generation
            model_configs: Configuration for each model
            training_config: Training hyperparameters
            device: Device to use
            seed: Random seed
        """
        self.dataset_config = dataset_config
        self.model_configs = model_configs
        self.training_config = training_config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.seed = seed
        
        # Set seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Storage for results
        self.results = {}
        self.histories = {}
        self.models = {}
    
    def prepare_data(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Prepare train/val/test dataloaders"""
        print("\nPreparing dataset\n")
        
        # Generate full dataset
        full_dataset = SyntheticGraphDataset.SyntheticGraphDataset(**self.dataset_config)
        
        # Split into train/val/test
        dataset_list = [full_dataset[i] for i in range(len(full_dataset))]
        
        train_data, temp_data = train_test_split(
            dataset_list,
            train_size=0.7,
            random_state=self.seed
        )
        val_data, test_data = train_test_split(
            temp_data,
            train_size=0.5,
            random_state=self.seed
        )
        
        print(f"\nDataset split:")
        print(f"  Train: {len(train_data)} graphs")
        print(f"  Val:   {len(val_data)} graphs")
        print(f"  Test:  {len(test_data)} graphs")
        
        # Create dataloaders
        train_loader = DataLoader(
            train_data,
            batch_size=self.training_config['batch_size'],
            shuffle=True
        )
        val_loader = DataLoader(
            val_data,
            batch_size=self.training_config['batch_size'],
            shuffle=False
        )
        test_loader = DataLoader(
            test_data,
            batch_size=self.training_config['batch_size'],
            shuffle=False
        )
        
        return train_loader, val_loader, test_loader
    
    def train_model(
        self,
        model_name: str,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader
    ) -> Dict:
        """Train a single model"""
        print(f"\nTraining {model_name.upper()}\n")
        
        # Create trainer
        trainer = Trainer.Trainer(
            model=model,
            device=self.device,
            learning_rate=self.training_config['learning_rate'],
            weight_decay=self.training_config['weight_decay']
        )
        
        # Train
        save_dir = "Trained_Models"
        os.makedirs(save_dir, exist_ok=True)  # Creates in current directory
        save_path = f"{save_dir}/{model_name}_{self.dataset_config['graph_type']}_{self.dataset_config['feature_type']}.pt" 
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=self.training_config['epochs'],
            patience=self.training_config['patience'],
            save_path=save_path,
            verbose=True
        )
        
        # Final test evaluation
        test_loss = trainer.evaluate(test_loader)
        
        result = {
            'train_loss': min(history['train_loss']),
            'val_loss': min(history['val_loss']),
            'test_loss': test_loss
        }
        
        print(f"\n {model_name} Results:")
        print(f"   Best Train Loss: {result['train_loss']:.4f}")
        print(f"   Best Val Loss:   {result['val_loss']:.4f}")
        print(f"   Test Loss:       {result['test_loss']:.4f}")
        
        return result, history, trainer.model
    
    def run(self):
        """Run complete experiment"""
        print("\nGraph Neural Network Comparison Experiment")
        print(f"\nDevice: {self.device}")
        print(f"\nModels: {', '.join(self.model_configs.keys())}")
        
        # Prepare data
        train_loader, val_loader, test_loader = self.prepare_data()
        
        # Get input dimension from first batch
        sample_batch = next(iter(train_loader))
        in_channels = sample_batch.x.shape[1]
        
        # Train each model
        for model_name, config in self.model_configs.items():
            # Create model
            if model_name.lower() == 'gcn':
                model = GCNModel.GCNModel(
                    in_channels=in_channels,
                    **config
                )
            elif model_name.lower() == 'gat':
                model = GATModel.GATModel(
                    in_channels=in_channels,
                    **config
                )
            elif model_name.lower() == 'gin':
                model = GINModel.GINModel(
                    in_channels=in_channels,
                    **config
                )
            else:
                raise ValueError(f"Unknown model: {model_name}")
            
            # Train and evaluate
            result, history, trained_model = self.train_model(
                model_name=model_name,
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,

            )
            
            # Store results
            self.results[model_name] = result
            self.histories[model_name] = history
            self.models[model_name] = trained_model
        
        training_plot_save_path = f"Training_plot_{self.dataset_config['feature_type']}"
        Visualiser.plot_training_history(self.histories, training_plot_save_path)
        Visualiser.create_comparison_table(self.results)

        save_dir = "GNN_Plots"
        os.makedirs(save_dir, exist_ok=True) 

        for model_name, model in self.models.items():
          new_save_path = f"{save_dir}/{model_name}_{self.dataset_config['graph_type']}_{self.dataset_config['feature_type']}.png"
          Visualiser.plot_regression_scatter(
            model=model,
            test_loader=test_loader,
            device=self.device,
            save_path=new_save_path
          )
        
        print("Finished!")
    