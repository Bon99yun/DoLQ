#!/bin/bash

# Environment Name
ENV_NAME="DoLQ"
PYTHON_VERSION="3.11"

echo "=========================================="
echo "Setting up Conda environment: $ENV_NAME"
echo "=========================================="

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Conda is not installed. Please install Conda first."
    exit 1
fi

# Accept Conda Terms of Service if needed
echo "Accepting Conda Terms of Service..."
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

# Create Conda Environment if it doesn't exist
if conda info --envs | grep -q "$ENV_NAME"; then
    echo "Environment '$ENV_NAME' already exists. Skipping creation."
else
    echo "Creating environment with Python $PYTHON_VERSION..."
    conda create -n $ENV_NAME python=$PYTHON_VERSION -y
    if [ $? -ne 0 ]; then
        echo "Failed to create conda environment. Exiting."
        exit 1
    fi
fi

# Activate Conda Environment
# Note: 'conda activate' works best in interactive shells. 
# For scripts, we might need to source conda.sh.
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

echo "Environment activated: $ENV_NAME"

# Install uv
echo "Installing uv..."
pip install uv

# Install dependencies using uv
echo "Installing requirements via uv..."
if [ -f "requirements.txt" ]; then
    # Use --python flag to ensure uv uses the conda environment's Python
    uv pip install -r requirements.txt --python "$(which python)" --index-strategy unsafe-best-match
    if [ $? -eq 0 ]; then
        echo "Dependencies installed successfully!"
    else
        echo "Failed to install dependencies. Trying with pip as fallback..."
        pip install -r requirements.txt
    fi
else
    echo "requirements.txt not found!"
fi

echo "=========================================="
echo "Setup Complete! Run 'conda activate $ENV_NAME' to start."
echo "=========================================="
