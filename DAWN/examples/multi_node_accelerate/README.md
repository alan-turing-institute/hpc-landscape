# Multi-Node PyTorch Accelerate Example

## Setup

1. Install miniconda using these instructions.
1. To create an environment, you'll need to either:
   1. Initialise it for the whole of your shell with `conda init`.
   2. Enable it for this shell session with `source ~/miniconda3/etc/profile.d/conda.sh`.
1. Create an environment with the `environment.yaml`.

## Run

1. Re-initialise conda, if you are in a new SSH session.
1. Activate the conda environemnt with `conda activate path/name`.
1. Submit the job with `sbatch sbatch_example.sh`.
1. Check the job's status with `sacct`.
1. View the job's output with `cat job.out`.
   
