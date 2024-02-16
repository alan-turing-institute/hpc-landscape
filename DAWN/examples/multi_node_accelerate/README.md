# Multi-Node PyTorch Accelerate Example

## Setup

Batch execute the script to create the conda environment on a compute note.
This will install conda, create the environment and install the requirements in
the environment.
```
sbatch sbatch_initconda.sh
```

This will create a Python 3.10 environment with the packages specified in
`environment.yaml`. Using these versions is important as (at least some)
versions of Torch 2.1 are known not to work.

## Run

Batch execute the script to run the two-node, eight-GPU PyTorch training
script.
```
sbatch sbatch_example.sh
```

Follow the job's output with `tail -f job.out`. Follow the GPU usage with `tail
-f dmon-*.txt` and the CPU usage with `tail -f cmon-*.txt`. 
