import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from typing import Optional, Dict
import EarlyStopping

class Trainer:
    """Trainer for GNN models"""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5
    ):
        """
        Args:
            model: GNN model to train
            device: Device to train on
            learning_rate: Initial learning rate
            weight_decay: L2 regularization weight
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.criterion = nn.MSELoss()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10
        )
    
    def train_epoch(self, loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_graphs = 0
        
        for batch in loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            
            out = self.model(batch.x, batch.edge_index)
            loss = self.criterion(out, batch.y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item() * batch.num_graphs
            num_graphs += batch.num_graphs
        
        return total_loss / num_graphs
    
    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> float:
        """Evaluate on validation/test set"""
        self.model.eval()
        total_loss = 0
        num_graphs = 0
        
        for batch in loader:
            batch = batch.to(self.device)
            out = self.model(batch.x, batch.edge_index)
            loss = self.criterion(out, batch.y)
            
            total_loss += loss.item() * batch.num_graphs
            num_graphs += batch.num_graphs
        
        return total_loss / num_graphs
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 250,
        patience: int = 20,
        save_path: Optional[str] = None,
        verbose: bool = True
    ) -> Dict:
        """
        Full training loop with validation
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Maximum number of epochs
            patience: Early stopping patience
            save_path: Path to save best model
            verbose: Whether to print progress
            
        Returns:
            Dictionary with training history
        """
        history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': []
        }
        
        early_stopping = EarlyStopping.EarlyStopping(patience=patience, mode='min')
        best_val_loss = float('inf')
        
        if verbose:
            print(f"Training {self.model.__repr__()}")
            print(f"Device: {self.device}")
            print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(1, epochs + 1):
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss = self.evaluate(val_loader)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Store history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['learning_rates'].append(current_lr)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if save_path:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_loss': val_loss,
                        'history': history
                    }, save_path)
            
            # Print progress
            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch:03d}/{epochs} | "
                      f"LR: {current_lr:.6f} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f}")
            
            # Early stopping
            if early_stopping(val_loss, epoch):
                if verbose:
                    print(f"\n Early stopping at epoch {epoch}")
                    print(f"Best val loss: {best_val_loss:.4f} at epoch {early_stopping.best_epoch}")
                break
        
        # Load best model
        if save_path:
            checkpoint = torch.load(save_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if verbose:
                print(f"\n Loaded best model from epoch {checkpoint['epoch']}")
        
        return history