# Detect which docker compose command to use
DOCKER_COMPOSE := $(shell command -v docker-compose 2> /dev/null)
ifndef DOCKER_COMPOSE
	DOCKER_COMPOSE := docker compose
endif

# Build Docker image
build:
	$(DOCKER_COMPOSE) build

# Run basic training
train:
	$(DOCKER_COMPOSE) up gnn-training

# View logs
logs:
	$(DOCKER_COMPOSE) logs -f

# Stop containers
stop:
	$(DOCKER_COMPOSE) down

# Clean generated outputs
clean-outputs:
	rm -rf GNN_Plots/*.png
	rm -rf Trained_Models/*.pth
	rm -rf results/*.json

# Install local dependencies (non-Docker)
install:
	pip install -r requirements.txt

# Run locally without Docker
local:
	python GNN.py --num_graphs 300 --num_nodes_min 50 --num_nodes_max 100 --graph_type erdos_renyi --feature_type structural_role --edge_prob 0.1