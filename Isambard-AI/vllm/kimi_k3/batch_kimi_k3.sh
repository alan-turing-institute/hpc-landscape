#!/bin/bash
# vim: et:ts=4:sts=4:sw=4

#SBATCH --time 2:00:0
#SBATCH --nodes 8
#SBATCH --gpus-per-node 4
#SBATCH --cpus-per-gpu 72
#SBATCH --mem 0
#SBATCH --job-name kimi_k3
#SBATCH --output kimi_k3_%j.log

echo "--------------------------------------"
echo "New job: ${SLURM_JOB_ID}"
echo "--------------------------------------"

cd -P .

module purge
module load brics/default
module load brics/apptainer-multi-node

mkdir -p "$HF_HOME" logs/kimi_k3

# Derive the head node's address for vLLM's --master-addr.
export PRIMARY_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export PRIMARY_IP=$(srun --nodes=1 --ntasks=1 -w "$PRIMARY_HOST" \
    bash -c "ip -4 -o addr show hsn0 | awk '{print \$4}' | cut -d/ -f1 | head -n1")
echo "Primary host: $PRIMARY_HOST ($PRIMARY_IP)"

# Propagate into the container.
export APPTAINERENV_PRIMARY_IP=$PRIMARY_IP

# Apply the patch and set PATCHED_MODEL_PY env variable
source patches/prepare_patched_model_py.sh

# Start container and run kimi_k3_run.sh
# Overwrite model.py with the patched version
srun -N"${SLURM_NNODES}" -n"${SLURM_NNODES}" --ntasks-per-node=1 -l \
    apptainer exec --nv \
    --bind "$PWD":"$PWD","$HF_HOME":/root/.cache/huggingface,"$PATCHED_MODEL_PY":/usr/local/lib/python3.12/dist-packages/vllm/models/kimi_k3/nvidia/model.py \
    container/kimi-k3.sif /host/adapt.sh bash "$PWD/kimi_k3_run.sh"
wait
