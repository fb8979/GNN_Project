# Graph Neural Network Comparison Framework

A containerized framework for comparing GNN architectures (GCN, GAT, GIN) on synthetic graph datasets.

## Overview

This project benchmarks three graph neural network models on node-level regression tasks across different graph topologies (Erdős-Rényi, Barabási-Albert, Watts-Strogatz) and feature types (structural roles, degree, PageRank).

## Quick Start

```bash
# Build and run with Docker
docker compose build
docker compose up gnn-training

# Or run locally
pip install -r requirements.txt
python GNN.py
```

## Project Structure

```
├── GNN.py                      # Main experiment script
├── GNNExperiment.py           # Experiment orchestration
├── GraphGenerator.py          # Graph topology generation
├── NodeFeatureGenerator.py    # Node features and labels
├── SyntheticGraphDataset.py   # PyTorch Geometric dataset
├── GCNModel.py                # Graph Convolutional Network
├── GATModel.py                # Graph Attention Network
├── GINModel.py                # Graph Isomorphism Network
├── Trainer.py                 # Training and evaluation
├── Visualiser.py              # Plotting utilities
├── Dockerfile                 # Container configuration
└── docker-compose.yml         # Multi-service orchestration
```

## Configuration

Edit `GNN.py` to customize:

- **Graph type**: `erdos_renyi`, `barabasi_albert`, `watts_strogatz`
- **Features**: `structural_role`, `degree`, `pagerank`, `homophily`
- **Models**: Configure hidden dims, layers, dropout for GCN/GAT/GIN
- **Training**: Batch size, learning rate, epochs, early stopping

## Results

Outputs are saved to:
- `GNN_Plots/` - Training curves and regression plots
- `Trained_Models/` - Model checkpoints (.pt files)
- `results/` - Performance metrics

## Docker Services

```bash
docker compose up gnn-training    # Train all models
```

## Requirements

- Docker Desktop (recommended) or Python 3.10+
- PyTorch, PyTorch Geometric, NetworkX, scikit-learn
