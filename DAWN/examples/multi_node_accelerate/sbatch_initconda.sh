#!/bin/bash
#SBATCH --partition=pvc
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --output=initconda.out
#SBATCH --cpus-per-gpu=1

set -o errexit

# Set up the environment.
module load intelpython-conda/2024.0.1.3

# Configure conda shell
conda init

# Install packages
conda env create -f environment.yaml

# Activate conda environment
conda activate torch-env

# Check python version (should be 3.10)
python --version

# Deactivate conda environment
conda deactivate

# To remove the environment
#conda remove --name torch-env --all -y

