import networkx as nx
import matplotlib.pyplot as plt
import torch
from typing import Dict
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import numpy as np
import os

class Visualiser:
    """Visualization utilities for GNN results"""
    
    @staticmethod
    def plot_training_history(
        histories: Dict[str, Dict],
        save_path: str = 'training_comparison.png'
    ):
        """
        Plot training histories for multiple models
        
        Args:
            histories: Dict mapping model names to history dicts
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss curves
        for model_name, history in histories.items():
            axes[0].plot(history['train_loss'], label=f'{model_name} Train', alpha=0.7)
            axes[1].plot(history['val_loss'], label=f'{model_name} Val', alpha=0.7)
        
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('MSE Loss', fontsize=12)
        axes[0].set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('MSE Loss', fontsize=12)
        axes[1].set_title('Validation Loss Comparison', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        os.makedirs("GNN_Plots", exist_ok=True) 
        new_save_path = f"GNN_Plots/{save_path}"

        plt.tight_layout()
        plt.savefig(new_save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Saved comparison plot to {save_path}")
    
    @staticmethod
    def plot_sample_prediction(
        model: nn.Module,
        data: Data,
        device: torch.device,
        title: str = "Sample Prediction"
    ):
        """
        Visualize predictions on a sample graph
    
        Args:
            model: Trained model
            data: Graph data
            device: Device
            title: Plot title
        """
        model.eval()
        with torch.no_grad():
            data = data.to(device)
            pred = model(data.x, data.edge_index).cpu().numpy()
    
        true = data.y.cpu().numpy()
    
        # Create networkx graph for visualization
        edge_index = data.edge_index.cpu().numpy()
        G = nx.Graph()
    
        # Add nodes explicitly
        num_nodes = data.num_nodes
        G.add_nodes_from(range(num_nodes))
    
        # Add edges
        G.add_edges_from(edge_index.T)
    
        # Only visualize if graph is not too large
        if num_nodes > 50:
            print(f"Graph too large to visualize ({num_nodes} nodes), skipping...")
            return
    
        pos = nx.spring_layout(G, seed=42)
    
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
        # True values
        nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax1)
        nodes1 = nx.draw_networkx_nodes(
            G, pos,
            node_color=true.flatten(),
            node_size=500,
            cmap='viridis',
            vmin=true.min(),
            vmax=true.max(),
            ax=ax1
        )
        ax1.set_title('True Values', fontsize=14, fontweight='bold')
        plt.colorbar(nodes1, ax=ax1)
        ax1.axis('off')
    
        # Predictions
        nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax2)
        nodes2 = nx.draw_networkx_nodes(
            G, pos,
            node_color=pred.flatten(),
            node_size=500,
            cmap='viridis',
            vmin=pred.min(),
            vmax=pred.max(),
            ax=ax2
        )
        ax2.set_title('Predictions', fontsize=14, fontweight='bold')
        plt.colorbar(nodes2, ax=ax2)
        ax2.axis('off')
    
        fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def create_comparison_table(results: Dict[str, Dict]) -> None:
        """
        Print comparison table of model results
        
        Args:
            results: Dict mapping model names to result dicts
        """
        print("Model Comparison Results:")
        print(f"{'Model':<15} {'Train Loss':<15} {'Val Loss':<15} {'Test Loss':<15}")
        
        for model_name, result in results.items():
            print(f"{model_name:<15} "
                  f"{result['train_loss']:<15.4f} "
                  f"{result['val_loss']:<15.4f} "
                  f"{result['test_loss']:<15.4f}")
        
        # Find best model
        best_model = min(results.items(), key=lambda x: x[1]['test_loss'])
        print(f"\n Best Model: {best_model[0]} (Test Loss: {best_model[1]['test_loss']:.4f})")

    @staticmethod
    def plot_regression_scatter(
        model: nn.Module,
        test_loader: DataLoader,
        device: torch.device,
        save_path: str = 'regression_scatter.png'
    ):
        """Scatter plot of true vs predicted values"""
        model.eval()
        all_true = []
        all_pred = []
    
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                pred = model(batch.x, batch.edge_index)
                all_true.append(batch.y.cpu().numpy())
                all_pred.append(pred.cpu().numpy())
    
        all_true = np.concatenate(all_true)
        all_pred = np.concatenate(all_pred)
    
        # Calculate metrics
        mse = np.mean((all_true - all_pred) ** 2)
        mae = np.mean(np.abs(all_true - all_pred))
        r2 = 1 - (np.sum((all_true - all_pred) ** 2) / 
              np.sum((all_true - all_true.mean()) ** 2))
    
        fig, ax = plt.subplots(figsize=(8, 8))
    
        # Scatter plot
        ax.scatter(all_true, all_pred, alpha=0.5, s=20)
    
        # Perfect prediction line
        min_val = min(all_true.min(), all_pred.min())
        max_val = max(all_true.max(), all_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 
            'r--', linewidth=2, label='Perfect Prediction')
    
        # Add metrics text
        textstr = f'MSE: {mse:.4f}\nMAE: {mae:.4f}\nR²: {r2:.4f}'
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
        ax.set_xlabel('True Values', fontsize=14)
        ax.set_ylabel('Predicted Values', fontsize=14)
        ax.set_title('True vs Predicted Node Values', fontsize=16, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
    
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close(fig)
        print(f"Saved scatter plot to {save_path}")