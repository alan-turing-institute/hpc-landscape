#!/bin/bash
#SBATCH --job-name=serve-k2.6
#SBATCH --nodes=2
#SBATCH --gpus=8
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# ==============================================================
# STEP 2 of 2: serve Kimi K2.6 across 2 nodes (8 GH200 GPUs) with vLLM.
#
# Before running, replace:
#   <<<PROJECT_STORAGE_PATH>>>  the same path you used in setup.sh
#   <<<ISAMBARD_USERNAME>>>     your Isambard username (only used in
#                               the laptop SSH command printed below)
#   <<<PROJECT_CODE>>>          your project code, e.g. u6fw -- the
#                               part before .aip2.isambard in the ssh
#                               command you normally use to log in
#
# Run once setup.sh has finished:  sbatch serve.sh
# ==============================================================

set -euo pipefail

WORKDIR=<<<PROJECT_STORAGE_PATH>>>
ENV_DIR=$WORKDIR/env
export HF_HOME=$WORKDIR/hf_cache

# CUDA forward compatibility -- lets this CUDA-13-targeted vLLM build
# run on Isambard-AI's CUDA-12.x driver. compat and math_libs must come
# first in LD_LIBRARY_PATH, ahead of the system driver's libcuda.so.
# math_libs (curand.h etc) is needed by flashinfer's JIT compile step.
# Deliberately NOT adding NVHPC's own bundled nccl/nvshmem -- those
# could shadow the Slingshot-tuned brics/nccl module loaded below.
NVHPC_ROOT=$WORKDIR/nvhpc/Linux_aarch64/26.3
export LD_LIBRARY_PATH=$NVHPC_ROOT/cuda/13.1/compat:$NVHPC_ROOT/math_libs/13.1/lib64:${LD_LIBRARY_PATH:-}
export CUDA_HOME=$NVHPC_ROOT/cuda/13.1
export PATH=$CUDA_HOME/bin:$PATH
export CPATH=$NVHPC_ROOT/cuda/13.1/include:$NVHPC_ROOT/math_libs/13.1/include:$NVHPC_ROOT/compilers/include:${CPATH:-}

# Model path was written by setup.sh's download step.
MODEL_PATH=$(cat "$WORKDIR/k2.6_model_path.txt")
MODEL_NAME="moonshotai/Kimi-K2.6"

module load brics/nccl brics/aws-ofi-nccl

# --- Figure out which node is "head" and which are "workers" ---
NODES=($(scontrol show hostnames $SLURM_NODELIST))
HEAD_NODE=${NODES[0]}
HEAD_IP=$(dig +short $HEAD_NODE)
RAY_PORT=6378
export VLLM_HOST_IP=$HEAD_IP

echo "Head node: $HEAD_NODE ($HEAD_IP)"
echo "Model: $MODEL_PATH"
echo ""
echo "From your laptop, once vLLM finishes loading:"
echo "  ssh -L 8000:$HEAD_NODE:8000 <<<ISAMBARD_USERNAME>>>@<<<PROJECT_CODE>>>.aip2.isambard"
echo "  curl http://localhost:8000/v1/models"

# Same CUDA fix as above -- needed again here since each srun step
# below starts with a clean environment.
activate_env() {
    source "$ENV_DIR/bin/activate"
    export LD_LIBRARY_PATH=$NVHPC_ROOT/cuda/13.1/compat:$NVHPC_ROOT/math_libs/13.1/lib64:${LD_LIBRARY_PATH:-}
    export CUDA_HOME=$NVHPC_ROOT/cuda/13.1
    export PATH=$CUDA_HOME/bin:$PATH
    export CPATH=$NVHPC_ROOT/cuda/13.1/include:$NVHPC_ROOT/math_libs/13.1/include:$NVHPC_ROOT/compilers/include:${CPATH:-}
    export LD_LIBRARY_PATH=$(find "$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia" -maxdepth 2 -type d -name lib | tr '\n' ':')$LD_LIBRARY_PATH
}
activate_env

# --- Start Ray on every node: one call for the head, one per worker ---
start_ray_node() {
    local node=$1 ip=$2 extra_args=$3
    srun --nodelist=$node --nodes=1 --gpus=4 --cpus-per-task=72 --ntasks-per-node=1 \
        bash -c "source $ENV_DIR/bin/activate; export LD_LIBRARY_PATH=$NVHPC_ROOT/cuda/13.1/compat:$NVHPC_ROOT/math_libs/13.1/lib64:\${LD_LIBRARY_PATH:-}; export CUDA_HOME=$NVHPC_ROOT/cuda/13.1; export PATH=\$CUDA_HOME/bin:\$PATH; export CPATH=$NVHPC_ROOT/cuda/13.1/include:$NVHPC_ROOT/math_libs/13.1/include:$NVHPC_ROOT/compilers/include:\${CPATH:-}; export LD_LIBRARY_PATH=\$(find \$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia -maxdepth 2 -type d -name lib | tr '\n' ':')\$LD_LIBRARY_PATH; ray start --block --num-gpus=4 --node-ip-address=$ip $extra_args" &
}

start_ray_node "$HEAD_NODE" "$HEAD_IP" "--head --port=$RAY_PORT"
for node in "${NODES[@]:1}"; do
    ip=$(dig +short $node)
    start_ray_node "$node" "$ip" "--address=$HEAD_IP:$RAY_PORT"
done

echo "Waiting for Ray cluster to come up..."
sleep 30

echo "Checking cluster status (should show 8 GPUs total across 2 nodes)..."
srun --overlap --nodelist=$HEAD_NODE --nodes=1 --ntasks=1 --gpus=0 \
    bash -c "source $ENV_DIR/bin/activate; ray status"

# --- Start vLLM, pointed at the Ray cluster we just built ---
# --enforce-eager: disables CUDA graph capture. Required -- vLLM V1 +
# Ray + pipeline-parallelism hits a known illegal-memory-access bug
# during graph capture (see ray-project/ray#51596). Trade-off: slightly
# higher per-token latency, not a correctness issue.
echo "Starting vLLM serve..."
srun --overlap --nodelist=$HEAD_NODE --nodes=1 --gpus=4 --ntasks-per-node=1 \
    bash -c "
        source $ENV_DIR/bin/activate
        export LD_LIBRARY_PATH=$NVHPC_ROOT/cuda/13.1/compat:$NVHPC_ROOT/math_libs/13.1/lib64:\${LD_LIBRARY_PATH:-}
        export CUDA_HOME=$NVHPC_ROOT/cuda/13.1
        export PATH=\$CUDA_HOME/bin:\$PATH
        export CPATH=$NVHPC_ROOT/cuda/13.1/include:$NVHPC_ROOT/math_libs/13.1/include:$NVHPC_ROOT/compilers/include:\${CPATH:-}
        export LD_LIBRARY_PATH=\$(find \$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia -maxdepth 2 -type d -name lib | tr '\n' ':')\$LD_LIBRARY_PATH

        vllm serve $MODEL_PATH \
            --served-model-name $MODEL_NAME \
            --distributed-executor-backend ray \
            --host 0.0.0.0 --port 8000 \
            --trust-remote-code \
            --tensor-parallel-size 4 \
            --pipeline-parallel-size 2 \
            --enforce-eager \
            --enable-auto-tool-choice \
            --tool-call-parser kimi_k2 \
            --reasoning-parser kimi_k2 \
            --mm-encoder-tp-mode data \
            --kv-cache-dtype fp8 \
            --gpu-memory-utilization 0.90 \
            --max-model-len 16384 \
            --max-num-seqs 16 \
            --all2all-backend deepep_v2
    "

wait