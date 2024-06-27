# Multi-Node PyTorch Accelerate Example

The example in this directory demonstrates model training across two nodes on DAWN and using eight GPUs (four on each node).
Note that each GPU on DAWN is split into two stacks, so we’ll need eight processes and it will look like we’re performing the training across eight GPUs.

The example uses PyTorch accelerate to simplify training and distribution.

Although configure for two nodes and eight GPUs, the code is simple and clear, so it should be self explanatory how you could increase the number of nodes and GPUs.

The training loop can be seen in the `example.py` file, which is where the actual work happens.
Let’s unravel the various execution blocks.

1. Lines  1-41: Set up the environment and configure communication between the nodes.
   This is achieved by reading in the node list from the environment configured by SLURM, then assign various environment variables to ensure they’re set up correctly.
2. Lines 41-48: Set up the model to include three linear transformation layers each comprised of 1000 weights.
   Adam is configured for optimisation.
3. Lines 49-64: Generate some example data.
   Every data point lines on the straight line $y = 0.7x + 0.3$.
   A total of 1000 data points are generated, selected uniformly between 0 and 1 in steps of 0.001.
4. Lines 65:91: Run the training loop.
   This runs a total of 50 epochs

During training `vmstat` and `xpi-smi` are used to collect data bout CPU and GPU usage respectively.

## Setup

Before starting the training script a Conda environment must be set up containing the required dependencies.
This can be done by executing the `sbatch_initconda.sh` script using SLURM:

```
sbatch sbatch_initconda.sh
```

This will install Conda, create the environment and install the requirements in the environment.
The Conda enviroronment will include a Python 3.10 environment with the packages specified in `environment.yaml`.
Using these versions is important as (at least some) versions of Torch 2.1 are known not to work.

Output from the process will be logged to the `initconda.out` file.

## Run

The `sbatch_example.sh` file is used to describe the job and its execution flow.
Here are its constituent pieces.

1. Lines 1-9: SLURM parameters describing what resources are needed to run the job.
2. Line 10: Ensure the process is exited in case of error.
3. Lines 11-19: Configure the environment to ensure the correct system software is available.
4. Line 22: Activate the Conda environment that was already set up using the `sbatch_initconda.sh` script.
5. Lines 24-38: Set parameters and environment variables needed for multi-node training to work correctly on DAWN.
6. Lines 40-43: Set `vmstat` and `xpu-smi` running in the background to capture CPU and GPU stats respectively.
7. Line 46: Run a simple test command to check things are working.
8. Line 49: Run the training script.
   This line kicks off all the real work that happens in `example.py`.

You should batch execute the script to run the two-node, eight-GPU PyTorch training as described above:

```
sbatch sbatch_example.sh
```

The following files will be generated as a consequence of running the script:

1. `job.out`: any text sent to `stdout` from the training process will be logged in this file.
2. `cpu.out`: a log of various CPU-related metrics captured during training.
3. `gpu.out`: a log of various GPU-related metrics captured during training.

It should take only a couple of minutes for the training to complete.
You can follow the job's output while it’s running using `tail -f job.out`.
