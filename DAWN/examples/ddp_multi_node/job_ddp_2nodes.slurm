#!/bin/bash
#SBATCH -ppvc
#SBATCH --time=00:05:00
#SBATCH -N 2
#SBATCH --ntasks-per-node 4
#SBATCH -c 4
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH -o slurm.out


export I_MPI_DEBUG=5
export I_MPI_OFFLOAD=1
export CCL_LOG_LEVEL=info 
export ZES_ENABLE_SYSMAN=1
module load intel-oneapi-mpi
module load intelpython-conda
conda activate pytorch-gpu
ulimit -n 1048575
mpirun -np 1 python ./ddp_demo.py --xpu 0 : -np 1 python ./ddp_demo.py --xpu 2 : -np 1 python ./ddp_demo.py --xpu 4 : -np 1 python ./ddp_demo.py --xpu 6 \
       : -np 1 python ./ddp_demo.py --xpu 0 : -np 1 python ./ddp_demo.py --xpu 2 : -np 1 python ./ddp_demo.py --xpu 4 : -np 1 python ./ddp_demo.py --xpu 6 \

