#!/bin/bash
#SBATCH --qos turing
#SBATCH --account <PROJECT_ID>
#SBATCH --time 3:00:0
#SBATCH --nodes 4
#SBATCH --gpus-per-node 4
#SBATCH --ntasks-per-node 1
#SBATCH --mem 491520
#SBATCH --job-name llama3-70b-ft

# Execute using:
# sbatch batch-llama3-70b-ft.sh

module purge
module load baskerville
module load bask-apps/live
module load PyTorch/2.1.2-foss-2022b-CUDA-11.8.0

VENV_PATH="./venv"

pushd /bask/homes/o/ovau2564/vjgo8416-ml-workload/llama-factory/LLaMA-Factory

if [ -d "$VENV_PATH" ]; then
    echo "Virtual environment exists, activating"
    . "${VENV_PATH}"/bin/activate
else
    echo "Creating and activating virtual environment"
    python3 -m venv "${VENV_PATH}"
    . "${VENV_PATH}"/bin/activate
    echo "Installing requirements"
    pip install pip --upgrade
    pip install -r requirements.txt
    pip install -e ".[torch,metrics]"
    pip install --upgrade huggingface_hub
fi

export PRIMARY_PORT=12340
export PRIMARY_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export NUM_PROCESSES=$((${SLURM_JOB_NUM_NODES}*${SLURM_GPUS_PER_NODE}))

echo
echo "######################################"
echo "Starting"
echo "######################################"
echo

# Track GPU metrics
stdbuf -o0 nvidia-smi dmon -o TD -s puct -d 1 > dmon.txt &

# Track CPU metrics
stdbuf -o0 vmstat -t 1 -y > cpu.txt &

# Run the task
srun bash -c \
    'accelerate launch \
    --config_file ../config_node02gpu04.yaml \
    --main_process_ip ${PRIMARY_ADDR} \
    --main_process_port ${PRIMARY_PORT} \
    --machine_rank ${SLURM_PROCID} \
    --num_processes ${NUM_PROCESSES} \
    --num_machines ${SLURM_JOB_NUM_NODES} \
    src/train.py examples/train_lora/llama3_lora_sft_70b.yaml'

echo
echo "######################################"
echo "Done"
echo "######################################"
echo

echo "Deactivating virtual environment"
deactivate
