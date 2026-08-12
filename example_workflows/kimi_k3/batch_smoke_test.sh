#!/bin/bash
#SBATCH --time 0:10:0
#SBATCH --gpus 1
#SBATCH --job-name kimi_k3_smoke
#SBATCH --output kimi_k3_smoke_%j.log

# Check that CUDA compatibility is working and we can start vllm

export APPTAINERENV_VLLM_ENABLE_CUDA_COMPATIBILITY=1
export APPTAINERENV_LD_PRELOAD=/usr/local/cuda-13.0/compat/libcuda.so.1

apptainer exec --nv container/kimi-k3.sif python3 -c "
import vllm  # must import before torch: sets LD_LIBRARY_PATH for CUDA compat
import torch
print('CUDA available:', torch.cuda.is_available())
print('Device:', torch.cuda.get_device_name(0))
print('Capability:', torch.cuda.get_device_capability(0))
"

apptainer exec --nv container/kimi-k3.sif vllm --version
