#!/bin/bash
#SBATCH --partition=pvc
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --output=initconda.out
#SBATCH --cpus-per-gpu=1

set -o errexit

# Set up the environment.
module purge
module load default-dawn
module load intelpython-conda/2024.0.1.3

# Configure conda shell
conda init

# Install packages
conda env create -f environment.yaml

# To remove the environment
#conda remove --name torch-env --all -y

