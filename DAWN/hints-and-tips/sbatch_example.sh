#!/bin/bash
#SBATCH --job-name litgpu
#SBATCH --output job.out
#SBATCH --account <ACCOUNT_ID>
#SBATCH --partition pvc
#SBATCH --time 0:10:0
#SBATCH --nodes 4
#SBATCH --ntasks-per-node 8
#SBATCH --cpus-per-task 12
#SBATCH --gres gpu:4

# Don't tolerate errors
set -e errexit

# Directory to move into at the start
pushd ${PWD}/.

# Set up modules environment
echo "Load modules"
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

# Avoid too many open file handles error
ulimit -n 1000000

# Avoid mpi failing to init
# See https://oneapi-src.github.io/oneCCL/env-variables.html#ccl-atl-transport
export CCL_ATL_TRANSPORT=ofi
# See https://www.intel.com/content/www/us/en/docs/mpi-library/developer-guide-windows/2021-6/ofi-providers-support.html#id-d3455e188
export FI_PROVIDER=verbs

# Avoids segfaults, for some reason
# https://spec.oneapi.io/level-zero/latest/sysman/PROG.html#environment-variables
export ZES_ENABLE_SYSMAN=1

# Otherwise we're told to
# Se3 https://github.com/oneapi-src/oneCCL/blob/master/man/OneCCL.md#ccl_ze_ipc_exchange
export CCL_ZE_IPC_EXCHANGE=sockets

# So that accelerate loads the ccl backend
# See https://github.com/oneapi-src/oneCCL/blob/master/man/OneCCL.md#ccl_worker_count
#export CCL_WORKER_COUNT=1
# See https://github.com/oneapi-src/oneCCL/blob/4b2c810d31cce3d957a76bcd595d131c7e77c5da/src/common/env/env.cpp#L960
export CCL_WORKER_OFFLOAD=0

# Disable XeTLA based customer kernels for performance
# See https://bigdl.readthedocs.io/en/latest/doc/LLM/Overview/install_gpu.html#id4
export USE_XETLA=OFF

# Use immediate command lists for performance
# See https://www.intel.com/content/www/us/en/developer/articles/guide/level-zero-immediate-command-lists.html
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1

# Define the master communication node for MPI
# See https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.plugins.environments.SLURMEnvironment.html
export MASTER_ADDR=${SLURMD_NODENAME}
# Select a random port in the range [29880, 38072)
export MASTER_PORT=$((29880 + ($RANDOM % 8192)))

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
  #pip install -r requirements.txt
fi

#echo "Start monitoring GPU usage"
#ZE_FLAT_DEVICE_HIERARCH="COMPOSITE" ZE_AFFINITY_MASK="" mpirun \
#  -n ${SLURM_JOB_NUM_NODES} \
#  -ppn 1 \
#  -prepend-rank \
#  bash -c 'stdbuf -o0 xpumcli dump -t 0,1 -m 0,1,2,5 -i 1 > gpu-${PMI_RANK}.out'

# MPI arrangement from SLURM
CONFIG_PROCESSES_TOTAL=$((${SLURM_JOB_NUM_NODES}*${SLURM_NTASKS_PER_NODE}))

echo "Do something meaningful"
mpirun \
  -n ${CONFIG_PROCESSES_TOTAL} \
  -ppn ${SLURM_NTASKS_PER_NODE} \
  -prepend-rank \
  python -c \
  'import os, torch, warnings; \
  warnings.filterwarnings("ignore"); \
  import intel_extension_for_pytorch; \
  print(torch.xpu.get_device_properties(int(os.environ.get("PMI_RANK")) % 8))'

deactivate
popd
echo "All done"
