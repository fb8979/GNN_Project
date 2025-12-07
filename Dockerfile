# GNN Comparison Project - Dockerfile
FROM --platform=linux/amd64 python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install NumPy 1.x first (PyTorch 2.1 compatibility)
RUN pip install --no-cache-dir "numpy<2.0"

# Install PyTorch (CPU version)
RUN pip install --no-cache-dir torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu

# Install PyTorch Geometric
RUN pip install --no-cache-dir torch-geometric

# Install other requirements
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create output directories
RUN mkdir -p GNN_Plots Trained_Models results

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV TORCH_USE_MKLDNN=0
ENV MKLDNN_DISABLE=1

CMD ["python", "GNN.py"]