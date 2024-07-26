# Hints and Tips

**Last updated: 2024-07-11**

In this file we provide some hints and tips, mostly around submitting jobs via the SLURM scheduler, which we think may be of use to DAWN users.

## SLURM Batch files

DAWN uses the SLURM scheduler to arrange execution of jobs on the system.
There are two common ways to request jobs using SLURM: either by submitting a batch job, or an interactive job.

In this file we’ll focus on batch jobs.
Such jobs are non-interactive and asynchronous.
This means that as a user you create a batch file that stipulates all of the steps involved in your job, which you submit to SLURM.
At some point later, when the resources are available, SLURM will action your job for execution.

In this document we go through some of the common things you might want to include in a batch file to submit to SLURM on DAWN.

### Example batch file

We’ve create an example batch file, [`sbatch_example.sh`](./sbatch_example.sh) to demonstrate the ideas described here.
This script is designed to execute correctly in its current form, running on four nodes with sixteen GPUs.
It doesn’t do anything useful though, simply outputting some info about the available GPUs to the output file.

### Using the example as a template

The example batch file should make a reasonable template for other more advanced tasks.
If you’re using it for this purpose there are a few crucial things you may want to change.
We’ll summarise these here, but go into greater detail in subsequent sections.

1. Line 25: module loading.
   If there are modules your application needs to run, they can be added here.
2. Lines 83-90: initialisation of the virtual environment.
   Your application is likely to have its own Python dependencies.
   Add them to this section to have them added to the virtual environment as it’s initialised.
3. Lines 100-112: task execution.
   The actual work is done here.
   You can remove these lines and add your own task here instead.

### Batch file submission

You’ll find the `sbatch_example.py` file already available in the repository.
The only thing you’ll need to do to get this to work is to change the `<ACCOUNT_ID>` on line 4 for the correct account that you’re using on DAWN.

Having made this change you can then submit the job to SLURM.
The script assumes that it’ll be run from the directory it’s found in, so make sure you move to that directory first before submitting the script to SLURM:

```bash
cd hints-and-tips
sbatch sbatch_example.py
```

You can now monitor the status of the script by checking the queue:

```bash
squeue --me
```

Once the script is running it’ll send all `stdout` output to the file `job.out`.
Once the job is finished you can load this file into your favourite text editor to check it.
You can also follow execution while it’s running if you want to:

```bash
tail -f job.out
```

This is everything you need to run the example, but we want to understand what’s actually going on with this batch script.
We’ll therefore go through all of the elements of it in the upcoming sections.

### SLURM preamble

The SLURM header always lives at the top of your batch file.
It provides details to SLURM about how to run your script, including the account to use, the maximum amount of time to give your task and the hardware to provision for it.

```shell
#!/bin/bash
#SBATCH --job-name gpuinfo
#SBATCH --output job.out
#SBATCH --account <ACCOUNT_ID>
#SBATCH --partition pvc
#SBATCH --time 0:10:0
#SBATCH --nodes 4
#SBATCH --ntasks-per-node 8
#SBATCH --cpus-per-task 12
#SBATCH --gres gpu:4
```

Let’s break this down:
1. `job-name`: this is for the user’s benefit.
    The job will appear in the queue with this name.
2. `output`: the file to save out the stdout output to.
3. `account`: This is the account to run the script on; you should change `<ACCOUNT_ID>` to the account id for your account.
   As a user you may have multiple projects and multiple accounts with different capabilities, so this allows you to choose the correct one.
4. `partition`: This selects the system to run your task on.
   Here `pvc` stands for Ponte Vecchio, the code name for the GPUs used by Dawn, so this tells SLURM to run your script on Dawn.
5. `time`: The maximum time your script is allowed to run, in the format hours:minutes:seconds.
   It doesn’t matter if your script finishes early, but the longer time you request the longer your script may end up waiting in the queue to be allocated resources.
6. `nodes`: The number of nodes to execute on.
   Each node has a maximum of four GPUs.
   Here we’re requesting four nodes so that we can run across sixteen GPUs.
7. `ntasks-per-node`: We want one task per “tile” on the GPU.
   Each GPU has two tiles and each node has four GPUs, so we want 2 * 4 = 8 tasks per node.
8. `cpus-per-task`: Here a CPU actually means a core.
   Each Dawn machine has 96 cores and we’re running 8 tasks per node, so here we’re asking for 8 * 12 = 96 cores.
   In other words, the full machine.
   In general you should scale the cores you need with the number of GPUs you’re requesting.
9. `gres`: “Generic Resource Scheduling”.
   We’re using this to specify the number of GPUs per node.
   So `gpu:4` here means four for each node, a total of 16 GPUs or 32 tiles.
   So many numbers!

Hopefully that clarifies the meaning, but for more detail, see the [recommended SLURM preamble](https://docs.hpc.cam.ac.uk/hpc/user-guide/pvc.html#recommendations-for-running-on-dawn) in the DAWN documentation, or if you really want the detail, check the SLURM documentation for [sbatch](https://slurm.schedmd.com/sbatch.html).

### Preparation

We then set a few arbitrary things up.
If something goes wrong we want our script to bail rather than continue regardless.
This is typically what you want so it’s worth adding this to the top of your batch scripts:

```shell
# Don't tolerate errors
set -e errexit
```

The working directory the script runs from will be the same as the directory you submit it from unless you specify otherwise using the `--chdir` argument when you call it.
We assume you’ll submit the script from the directory it’s in.
The next line then moves into the the same directory.
Yes, in fact it doesn’t really do anything.
But if you needed to enter a different directory, you could do so here.

```shell
# Directory to move into at the start
pushd ${PWD}/.
```

## Loading modules

The DAWN team has made a large number of different tools, libraries and frameworks available from DAWN using modules.
Using a module is a bit like installing a package, except that because the files are already on the system they’re usually really fast and easy to activate.

It’s far better to use a module if it’s available rather than installing the software yourself since the software packaged in modules will be tailored for the system it’s built for.

We should always run `module purge` and load `default-dawn` before loading or installing any other software.

We’re also activating a bunch of Intel OneAPI modules that are needed by PyTorch and to support MPI.

```shell
# Set up modules environment
echo "Load modules"
module purge
module load default-dawn
module load intel-oneapi-tbb/2021.11.0/oneapi/xtkj6nyp
module load intel-oneapi-compilers/2024.0.0/gcc/znjudqsi
module load intel-oneapi-mkl/2024.0.0/oneapi/4n7ruz44
module load intel-oneapi-mpi/2021.11.0/oneapi/h7nq7sah
module load gcc/13.2.0/ic25lr2r
```

See the [DAWN software documentation](https://docs.hpc.cam.ac.uk/hpc/user-guide/pvc.html#software) for more info about using modules on DAWN.

The OneAPI documentation tells us that we need to run some scripts to set up the environment.
Having activated the various modules we still need to do this and the following commands are essentially just following these requirements.
We use `pushd` and `popd` to change directory here so that we can restore our location after these commands have executed.

```shell
echo "Initialise environment"
pushd /usr/local/dawn/software/spack/spack-views/dawn-test-2023-12-22/
source intel-oneapi-compilers-2024.0.0/gcc-13.2.0/znjudqsiaf6x5u2rxdtymf6ss55nmimw/compiler/2024.0/env/vars.sh
source intel-oneapi-mkl-2024.0.0/oneapi-2024.0.0/4n7ruz44nhbsd5xp4nnz6mgm2z7vqzxs/mkl/2024.0/env/vars.sh
source intel-oneapi-compilers-2024.0.0/gcc-13.2.0/znjudqsiaf6x5u2rxdtymf6ss55nmimw/setvars.sh
popd
```

## Socket limit

The Intel [“Collective Communications Library”](https://oneapi-src.github.io/oneCCL/) MPI mechanism that we use to communicate between nodes and processes relies on sockets for communication.
This can result in a very large number of sockets being opened, more in fact than the default allowed by the operating system.
The following command increases the limit to avoid our task exceeding it (which would cause it to fail).

```shell
# Avoid too many open file handles error
ulimit -n 1000000
```

## Environment variables

Next up we set a whole host of environment variables.
Most of these are essentially default settings needed to get the communication library to work.
Each variable has been annotated with a description and link to the documentation (or the closest we could find to documentation) that explains why it’s needed.

```shell
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
```

The last two environment variables are `MASTER_ADDR` and `MASTER_PORT`.
These are the name and port for the master node that’s used to support communication between all of the other nodes.
We’re get the name of the master node from SLURM and choose a random port.

## Setting up the Virtual Environment

It’s convenient to have an environment variable for where we plan to install our virtual environment.
The next line sets this up.

```shell
# Define the path to the virtual environment relative to the CWD
VENV_PATH="./venv"
```

Following this we have a bit of a mess, but it’s actually quite simple.
The commands check whether the Python virtual environment directory already exists.
If it doesn’t it creates the virtual environment and installed all of the requirements inside it.
If it does exist it assumes everything is already correctly installed and just activates the environment instead.

```shell
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
```

This means that the first time we run the batch file a suitable Python virtual environment will be created.
On subsequent runs it will activate the virtual environment instead.

The actual dependencies installed here are designed to suit a basic environment with PyTorch and MPI.
You’ll likely want to have additional dependencies installed and these should be placed inside the `else` condition.
For example, there’s a line commented out to install the dependencies listed in a `requirements.txt` file, which is a common way to provide Python dependencies for a project.

## Monitoring GPU usage

Next you’ll find a block of code that’s been commented out:

```shell
#echo "Start monitoring GPU usage"
#ZE_FLAT_DEVICE_HIERARCH="COMPOSITE" ZE_AFFINITY_MASK="" mpirun \
#  -n ${SLURM_JOB_NUM_NODES} \
#  -ppn 1 \
#  -prepend-rank \
#  bash -c 'stdbuf -o0 xpumcli dump -t 0,1 -m 0,1,2,5 -i 1 > gpu-${PMI_RANK}.out'
```

In theory this should spawn one process per node, each of which will execute the `xpumcli` command to monitor GPU usage on the node.
The results are output to files named `gpu-0.out`, `gpu-1.out`, `gpu-2.out` and `gpu-3.out`; one file for each node.

This code is commented out because at time of writing it unfortunately doesn’t work, due to the unavailability of the `xpumcli` command.
This may change in future if the configuration on DAWN is changed.

## Running the training

We’re now going to execute the command that actually does the work.
For example this could be training a model, performing some inference, or something else.

In our case, since this is just example, we’re going to run a short inline Python script designed to output some info about a single GPU.
We’ll execute this in parallel with one process running for each available GPU (so 32 in total).
If all goes to plan when running this, it will output 32 lines of GPU info to our output file `job.out`.


To do this it’s useful for us to have to hand an environment variable that holds the total number of processes.
SLURM will automatically provide some useful environment variables, but it doesn’t provide this one, so we need to calculate it from the values it *does* provide.
We use the `$((expression))` pattern to ask the shell interpreter to perform the required calculation for us.

```shell
# MPI arrangement from SLURM
CONFIG_PROCESSES_TOTAL=$((${SLURM_JOB_NUM_NODES}*${SLURM_NTASKS_PER_NODE}))
```

Here the total processes is the number of nodes multiplied by the number of tasks per node.
The environment variables that start `SLURM_...` are assigned automatically when SLURM runs the script, so this is a convenient way to set up environment variables that will change automatically depending on the values we put in the SLURM preamble.

Given the SLURM preamble, this environment variables will end up being set to `CONFIG_PROCESSES_TOTAL` = 4 * 8 = 32.
We’ll use this again shortly.

For a full list of the environment variables that SLURM sets when it runs your script, see the [Output Environment Variables](https://slurm.schedmd.com/sbatch.html#SECTION_OUTPUT-ENVIRONMENT-VARIABLES) section of the sbatch documentation.

We’re finally there! The command here executes our script with appropriate parameters.

```shell
mpirun \
  -n ${CONFIG_PROCESSES_TOTAL} \
  -ppn ${SLURM_NTASKS_PER_NODE} \
  -prepend-rank \
  python -c \
  'import os, torch, warnings; \
  warnings.filterwarnings("ignore"); \
  import intel_extension_for_pytorch; \
  print(torch.xpu.get_device_properties(int(os.environ.get("PMI_RANK")) % 8))'
```

There are some important things to note here.
We don’t just call python with the script.
We instead pass the python command to `mpirun`.

The `mpirun` command is used to execute something multiple times across different nodes.
It will set up certain variables in the environments the processes are run in so they can tell each other apart.
For example, `mpirun` will set the `PMI_RANK` variable to a different number for each process and the `PMI_SIZE` environment variable to the total number of processes executed.
You’ll notice that we use `PMI_RANK` to select a GPU to query in the inline script.

Recall from earlier that we set up the `CONFIG_PROCESSES_TOTAL` environment variable to take the value 32.
Consequently we’re calling `mpirun` to execute 32 processes in total split into 8 processes per node.

Depending on what we’re attempting we might also want to pass values that depend on our SLURM configuration into a training script as hyperparameters.
You can see this in action by taking a look at the equivalent [`sbatch_example.sh`](../examples/gpt_lightning/sbatch_example.sh) file in the `gpt_lightning` example.

The output from executing the script will be stored in the `job.out` file, as specified in the SLURM preamble at the top of the batch file.
As noted earlier, you can use `tail` to follow this output as it’s produced, or view it afterwards in your favourite text editor.

## Cleaning up

Once the training step has completed we deactivate our virtual environment and move into the directory we started from.
These steps aren’t strictly necessary because the shell environment will be lost anyway, but it feels right to restore things.

```shell
deactivate
popd
echo "All done"
```

And that’s it! Hopefully walking through this script will have helped make clearer what’s going on and allow you to write suitable batch scripts for your own tasks.
