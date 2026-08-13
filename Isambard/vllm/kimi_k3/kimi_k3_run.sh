#!/bin/bash
set -e

export HF_HOME=/root/.cache/huggingface

# Enable CUDA forward compatibility so this works on CUDA 12.7
export VLLM_ENABLE_CUDA_COMPATIBILITY=1
# Apptainer binds some LD_LIBRARY files before vllm so we need to preload this file.
export LD_PRELOAD=/usr/local/cuda-13.0/compat/libcuda.so.1

# Apptainer's --nv flag puts libcuda.so.1 in /.singularity.d/libs - tell Triton to use this one.
export TRITON_LIBCUDA_PATH=/.singularity.d/libs

# Route NCCL/Gloo over the Slingshot NIC (not eth0).
# NCCL_SOCKET_IFNAME uses the "hsn" prefix, but GLOO_SOCKET_IFNAME needs the exact interface name "hsn0" or it fails to start.
export NCCL_SOCKET_IFNAME=hsn
export GLOO_SOCKET_IFNAME=hsn0

# NCCL_NET forces the aws-ofi-nccl RDMA plugin so NCCL doesn't silently fall back to slow TCP sockets.
export NCCL_NET="AWS Libfabric"

export NCCL_CROSS_NIC=1
export NCCL_NET_FORCE_FLUSH=0
export FI_PROVIDER=cxi
export FI_CXI_RX_MATCH_MODE=hybrid
export FI_CXI_DISABLE_CQ_HUGETLB=1

# Info level debugging - can change once its working
export NCCL_DEBUG=INFO
export NCCL_DEBUG_FILE=logs/kimi_k3/nccl-%h.%p.log

export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_USE_RUST_FRONTEND=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Each node needs its own triton_cache directory (/tmp is local to node)
export TRITON_CACHE_DIR=/tmp/triton_cache_kimi_k3_${SLURM_JOB_ID}_${SLURM_NODEID}

NODE_RANK=$SLURM_NODEID

# From https://recipes.vllm.ai/moonshotai/Kimi-K3
ARGS=(
  --trust-remote-code
  --tensor-parallel-size 32
  --enable-expert-parallel
  --nnodes "${SLURM_NNODES}"
  --node-rank "${NODE_RANK}"
  --master-addr "${PRIMARY_IP}"
  --gpu-memory-utilization 0.97
  --max-num-seqs 5
  --max-model-len 32768
  --moe-backend marlin
  --disable-custom-all-reduce
  --no-enable-flashinfer-autotune
  --max-num-batched-tokens 4096
  --attention-backend FLASHMLA
  --enable-auto-tool-choice
  --tool-call-parser kimi_k3
  --reasoning-parser kimi_k3
)

if [[ "$NODE_RANK" -eq 0 ]]; then
    echo "[rank 0/head] $(hostname) starting vLLM, master ${PRIMARY_IP}"
    vllm serve moonshotai/Kimi-K3 "${ARGS[@]}" &
    VLLM_PID=$!
    trap 'kill $VLLM_PID 2>/dev/null' EXIT

    # Wait for the REST API to be available
    until curl -s http://localhost:8000/v1/models >/dev/null 2>&1; do
        sleep 60
        echo "Waiting for vLLM to start..."
    done
    echo "vLLM is up -- running a real inference request to confirm end-to-end"

    curl -s http://localhost:8000/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{"model": "moonshotai/Kimi-K3", "messages": [{"role": "user", "content": "What ingredients do I need to bake a cake?."}], "max_tokens": 500}'
    echo

    wait "$VLLM_PID"
else
    echo "[rank ${NODE_RANK}/worker] $(hostname) starting vLLM, master ${PRIMARY_IP}"
    vllm serve moonshotai/Kimi-K3 "${ARGS[@]}" --headless
fi
