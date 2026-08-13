#!/bin/bash
#SBATCH --job-name=setup-k2.6
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2G
#SBATCH --time=03:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# ==============================================================
# STEP 1 of 2: one-time setup for serving Kimi K2.6 on Isambard-AI.
#
# Before running, replace:
#   <<<PROJECT_STORAGE_PATH>>>  a directory on YOUR project storage
#                               (Lustre, not $HOME) where everything
#                               for this deployment will live, e.g.
#                               /home/<project>/<you>/<project>_shared/<you>/kimi26
#
# Then:
#   export HF_TOKEN=hf_xxxxxxxx
#   sbatch setup.sh
#
# Safe to re-run: each step is skipped if already done.
# ==============================================================

set -euo pipefail

module purge
module load brics/default brics/nccl brics/aws-ofi-nccl

WORKDIR=<<<PROJECT_STORAGE_PATH>>>
ENV_DIR=$WORKDIR/env
NVHPC_INSTALL_DIR=$WORKDIR/nvhpc

# Where model weights get cached
: "${HF_HOME:=$WORKDIR/hf_cache}"
export HF_HOME
echo "Using HF_HOME: $HF_HOME"

# Setting uv's default cache in the project storage
: "${UV_CACHE_DIR:=$WORKDIR/.uv-cache}"
export UV_CACHE_DIR

mkdir -p "$WORKDIR" "$HF_HOME" "$UV_CACHE_DIR"

# Checking for HF_TOKEN if the model weights haven't been downloaded yet.
if [ ! -f "$WORKDIR/model_path.txt" ] && [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN is not set." >&2
    echo "Run 'export HF_TOKEN=hf_xxxxxxxx' before submitting this job." >&2
    exit 1
fi

echo "=== [1/3] Python environment ==="
if [ ! -f "$ENV_DIR/bin/activate" ]; then
    # making sure that Lustre's striping is set to 1 for the env dir, 
    # so that many small files don't get spread across OSTs
    mkdir -p "$ENV_DIR"
    lfs setstripe -c 1 "$ENV_DIR" || true
    uv venv --seed --python=3.12 "$ENV_DIR"
fi
source "$ENV_DIR/bin/activate"

uv pip install -U vllm==0.26.0 flashinfer-python ray[default] huggingface_hub \
    --torch-backend=auto \
    --extra-index-url https://wheels.vllm.ai/nightly/vllm

export LD_LIBRARY_PATH=$(find "$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia" -maxdepth 2 -type d -name lib | tr '\n' ':')${LD_LIBRARY_PATH:-}

echo "torch/vllm versions:"
vllm --version
python -c "import torch; print('torch:', torch.__version__)"

# Needed because Isambard-AI's driver targets CUDA 12.x, but this
# vLLM build is compiled against CUDA 13. See:
# https://docs.isambard.ac.uk/user-documentation/guides/gpus_and_cuda/#cuda-forward-compatibility
echo "=== [2/3] NVIDIA HPC SDK ==="
if [ ! -d "$NVHPC_INSTALL_DIR" ]; then
    mkdir -p "$WORKDIR/nvhpc_download"
    cd "$WORKDIR/nvhpc_download"
    # aarch64 tarball -- Isambard-AI is ARM (Grace CPU), not x86_64.
    wget -nc https://developer.download.nvidia.com/hpc-sdk/26.3/nvhpc_2026_263_Linux_aarch64_cuda_13.1.tar.gz
    tar xpzf nvhpc_2026_263_Linux_aarch64_cuda_13.1.tar.gz
    cd nvhpc_2026_263_Linux_aarch64_cuda_13.1
    NVHPC_SILENT="true" NVHPC_INSTALL_DIR="$NVHPC_INSTALL_DIR" NVHPC_INSTALL_TYPE="single" ./install
else
    echo "Already installed, skipping."
fi
echo "NVHPC_ROOT: $NVHPC_INSTALL_DIR/Linux_aarch64/26.3"

echo "=== [3/3] Model weights ==="

MODEL_PATH_FILE="$WORKDIR/k2.6_model_path.txt"
if [ -f "$MODEL_PATH_FILE" ]; then
    echo "Already downloaded, skipping. See $MODEL_PATH_FILE:"
    cat "$MODEL_PATH_FILE"
else
    hf auth login --token "$HF_TOKEN"
    export HF_XET_HIGH_PERFORMANCE=1
    MODEL_PATH=$(hf download moonshotai/Kimi-K2.6 | tee /dev/stderr | grep "path:" | sed 's/.*path: *//')
    echo "$MODEL_PATH" > "$MODEL_PATH_FILE"
    echo "Model downloaded to: $MODEL_PATH"
fi

echo ""
echo "Setup complete. Run 'sbatch serve.sh' next."