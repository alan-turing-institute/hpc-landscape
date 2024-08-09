#!/bin/bash
#SBATCH --job-name llamafactory
#SBATCH --output job.out
#SBATCH --account <ACCOUNT_ID>
#SBATCH --partition pvc
#SBATCH --time 1:00:0
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 36
#SBATCH --gres gpu:1

# Execute as
# sbatch batch-llamafactory-single.sh

# Don't tolerate errors
set -e errexit

# Directory to move into at the start
pushd <FULL_PATH>/LLaMA-Factory

# Set up modules environment
echo "Load modules"
module purge
module load default-dawn
module load intel-oneapi-tbb/2021.11.0/oneapi/xtkj6nyp
module load intel-oneapi-compilers/2024.0.0/gcc/znjudqsi
module load intel-oneapi-mkl/2024.0.0/oneapi/4n7ruz44
module load intel-oneapi-mpi/2021.11.0/oneapi/h7nq7sah
module load gcc/13.2.0/ic25lr2r

echo "Initialise environment"
pushd /usr/local/dawn/software/spack/spack-views/dawn-test-2023-12-22/
source intel-oneapi-compilers-2024.0.0/gcc-13.2.0/znjudqsiaf6x5u2rxdtymf6ss55nmimw/compiler/2024.0/env/vars.sh
source intel-oneapi-mkl-2024.0.0/oneapi-2024.0.0/4n7ruz44nhbsd5xp4nnz6mgm2z7vqzxs/mkl/2024.0/env/vars.sh
source intel-oneapi-compilers-2024.0.0/gcc-13.2.0/znjudqsiaf6x5u2rxdtymf6ss55nmimw/setvars.sh
popd

# Define the path to the virtual environment relative to the CWD
VENV_PATH="./venv"

if [ -d "$VENV_PATH" ]; then
  echo "Activate virtual environment"
  source "${VENV_PATH}"/bin/activate
  source $(python -c "import oneccl_bindings_for_pytorch as torch_ccl;print(torch_ccl.cwd)")/env/vars.sh
else
  echo "Create virtual environment"
  python3.9 -m venv ${VENV_PATH}
  source ./${VENV_PATH}/bin/activate
  pip install --upgrade pip
  pip install "numpy<2"
  pip install torch==2.0.1a0 \
    torchvision==0.15.2a0 \
    intel-extension-for-pytorch==2.0.120+xpu \
    oneccl-bind-pt==2.0.200 \
    --extra-index-url https://pytorch-extension.intel.com/release-whl-aitools/
  source $(python -c "import oneccl_bindings_for_pytorch as torch_ccl;print(torch_ccl.cwd)")/env/vars.sh
  pip install -e ".[torch,metrics]"
fi

echo "Run LLaMA fine-tuning"

# Following the instructions from
# https://github.com/hiyouga/LLaMA-Factory/tree/main?tab=readme-ov-file

llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml

deactivate
popd
echo "All done"

