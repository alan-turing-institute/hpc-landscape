#!/bin/bash
#SBATCH --partition=pvc
#SBATCH --time=6
#SBATCH --nodes=2
#SBATCH --gpus=8
#SBATCH --gpus-per-node=4
#SBATCH --output=job.out
#SBATCH --cpus-per-gpu=10

# Set up the environment.
module load intel-oneapi-tbb/2021.11.0/oneapi/xtkj6nyp
module load intel-oneapi-compilers/2024.0.0/gcc/znjudqsi
module load intel-oneapi-mkl/2024.0.0/oneapi/4n7ruz44
module load intel-oneapi-mpi/2021.11.0/oneapi/h7nq7sah

# Avoid too many open file handles error.
ulimit -n 1000000

# Avoid mpi failing to init.
export CCL_ATL_TRANSPORT=ofi
export FI_PROVIDER=verbs

# Avoids segfaults, for some reason.
export ZES_ENABLE_SYSMAN=1

# Otherwise we're told to.
export CCL_ZE_IPC_EXCHANGE=sockets

# So that accelerate loads the ccl backend.
export CCL_WORKER_COUNT=1

# Check that we can call mpirun with something simple.
mpirun -n 16 -ppn 8 -prepend-rank hostname

# Do training.
/usr/bin/time mpirun -n 16 -ppn 8 -prepend-rank accelerate launch --config_file xpu-multinode.yaml example.py

