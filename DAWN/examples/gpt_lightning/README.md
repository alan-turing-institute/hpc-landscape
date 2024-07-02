# PyTorch Lightning on DAWN

PyTorch Lightning isn’t currently supported for use on Intel GPUs.
This example shows how you can take the lit-GPT code &mdash; a version of ming-GPT that’s been amended for use with PyTorch Lightning &mdash; and adjust it so that it works with the Intel GPUs on DAWN.

In the context of PyTorch Lightning, the term XPU is often used in place of GPU if it’s in relation to an Intel GPU.

## Preparation

### Fetch the submodule

First let’s move in to the correct folder.
The exact command you use will depend on where you currently are in the directory hierarchy.
The key point is that we want to end up in the `hpc-landscape/DAWN/examples/gpt_lightning` directory.
```shell
$ cd ./hpc-landscape/DAWN/examples/gpt_lightning
```

Before we get started make sure you’ve initialised and updated all of the submodules.
The `gpt_lightning` example folder contains a submodule which itself contains a submodule, so we must perform the update recursively.
```shell
$ git submodule update --init --recursive
```

### Download the training data

Our training script is designed to load the training data from a file called `shakespeare_input.txt` if it exists, or try to download it from the Internet otherwise. We’ll download it in advance so we only have to do it the once.

We’re going to use the classic “Shakespeare” dataset for training, which includes all of the works of Shakespeare. If we use this dataset, our model will be learning how to generate Shakespeare.

```shell
$ pushd lit-GPT/
$ wget https://cs.stanford.edu/people/karpathy/char-rnn/shakespeare_input.txt
$ popd
```

Note here we use `pushd` to move in to a directory and `popd` to move out of it again. We could have used `cd` for this as well.

## Install the XPU overrides

Since PyTorch lightning doesn’t support Intel GPUs directly, we need to add our own functionality in order to get it to work.
PyTorch Lightning includes a plugin capability for adding new GPUs.
However, as we’ll see, this isn’t quite flexible enough to handle all of the required changes.
Nevertheless it does simplify things a lot.
To this end, most of the capabilities that PyTorch Lightning needs in order to handle Intel GPUs can be found in the `xpu.py` file.
This is a version of the `xpu.py` [developed by Intel](https://github.com/maxwelltsai/rho-diffusion/tree/main) but with support added for BFloat16 precision.

You don’t need to understand the details in this file.
What’s important is that you understand that this file contains a subclass of the `LightningEnvironment` called `IntelMPIEnvironment` and a subclass of `Accelerator` called `XPUAccelerator`. We’ll override the default environment and accelerator using these.

In order to do this, the first thing we must do is make the contents of `xpu.py` available to the training code.
We do this by copying the file into our project.

```shell
$ cp xpu.py gpt_lightning/
```

Now that we have a copy in the same folder as our other code, we can import it as if it were a standard Python library.

## Apply the patch (or don’t)

As already discussed, installing the Lightning overrides isn’t enough to get things to work, we’ll also need to adjust our code slightly in order to make use of it.

We’ve provided a patch that you can apply to the existing project which will amend the code appropriately.
We’re going to work through all of the changes here, so you can either apply this patch and check the changes using `git diff` as we go along, or you can edit the files directly as we work our way through the changes.
I’d recommend the latter, but you’ll just need to be careful not to make any mistakes.
If you get in a mess at any point, you can always perform a `git reset --hard` to clear all of your changes and then apply the patch to sort things out.

You can apply the patch like this:
```shell
$ pushd lit-GPT
$ git apply ../0001-Add-DAWN-specific-changes.patch
$ popd
```

## Update the code manually

You only need to make these changes if you didn’t apply the patch. If you did apply the patch, you may still prefer to walk through these steps without applying them so as to understand what motivates them.

If you didn’t apply the patch, you should apply these changes directly yourself by editing the code.

### Update the imports

First up we need to import the new functionality from `xpy.py` into the `train.py` file. We have to import it *before* we import anything from PyTorch or Lightning, so add this directly after the `URLError` error and before the `lightning` import.

```python
# XPUAccelerator must be imported before PyTorch or Lightning
import xpu
```

There are a few other imports we’ll need, but which should come *after* the Lightning imports. We need some functionality from `torch.distributed` and the Intel-specific bindings for PyTorch. These should be placed after the `lightning` imports but before the `main` function starts.

```python
from torch.distributed import init_process_group, destroy_process_group
import oneccl_bindings_for_pytorch
from lightning.pytorch.strategies import DDPStrategy
```

### Configure the environment

Unfortunately there are a few environment variables that are needed for message passing between nodes, but which aren’t configured automatically by SLURM on DAWN. In particular SLURM will set up the `PMI_RANK` with the rank of the current node (so that each node has a unique number) and the `PMI_SIZE` with the total number of nodes that are available. The Intel implementation of Lightning requires these to be called `RANK` and `WORLD_SIZE` respectively.

Consequently we need to add some code to read in the values and set some environment variables with the expected names. We also need to initialise the message passing backend to `ccl`, the [“Collective Communications Library”](https://oneapi-src.github.io/oneCCL/) which is specific to Intel.

We have to do this near the top of the program, so we put it inside the `main` function, just before the call to `data.CharDataset()`.

Here’s the code to add:

```python
    os.environ["RANK"] = str(int(os.environ["PMI_RANK"]))
    os.environ["WORLD_SIZE"] = str(int(os.environ["PMI_SIZE"]))
    init_process_group(backend='ccl')
```

### Set up the callbacks

The standard `lit-GPT` code has some callbacks added, essentially for the purposes of outputting relevant metrics (e.g. how long the epoch took and how much memory it used).

Unfortunately these are CUDA-specific. We could just skip the CUDA versions, but it’s a bit nicer if we add some alternatives that will work on the Intel hardware of DAWN.

For this we can add the following code to the end of the `lightning_gpt/callbacks.py` file:

```python
class XPUMetricsCallback(Callback):
    def on_train_epoch_start(self, trainer: "Trainer", pl_module: "LightningModule") -> None:
        # Reset the memory use counter
        torch.xpu.reset_peak_memory_stats(self.root_gpu(trainer))
        torch.xpu.synchronize(self.root_gpu(trainer))
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer: "Trainer", pl_module: "LightningModule") -> None:
        torch.xpu.synchronize(self.root_gpu(trainer))
        max_memory = torch.xpu.max_memory_allocated(self.root_gpu(trainer)) / 2**20
        epoch_time = time.time() - self.start_time

        max_memory = trainer.strategy.reduce(max_memory)
        epoch_time = trainer.strategy.reduce(epoch_time)

        rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
        rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")

    def root_gpu(self, trainer: "Trainer") -> int:
        return trainer.strategy.root_device.index
```

We then have to create a hook to add the callbacks in to the main loop. For this we go back to the `train.py` file, where we should add the following just after the line creating the `callback_list` variable.

```python
    if torch.xpu.is_available():
        torch.set_float32_matmul_precision("high")
        callback_list.append(callbacks.XPUMetricsCallback())
```

### Set up the Strategy and Trainer

We now get to probably the most significant change. Ordinarily we would use the `Trainer` class from Lightning, setting the strategy as a parameter, typically set as a string (for example `strategy=“ddp”`).

Lightning allows us to subclass the existing strategies or create new ones and pass them in directly, and this is what we’ll be doing here. However there’s another catch because if we set the precision to `bf16` Lightning will assume we want to use CUDA, which is specific to Nvidia GPUs.

Consequently we also need to use an XPU-specific subclass of the `Trainer` in order to handle BFloat16 precision successfully.

Here’s the code for the trainer from the base code aimed at other platforms:

```python
    trainer = L.Trainer.from_argparse_args(
        args,
        max_epochs=2,
        gradient_clip_val=1.0,
        callbacks=callback_list,
        enable_checkpointing=False,
        accelerator="auto",
    )
```

We’re going to amend this so that it uses our XPU-specific subclasses of the Lightning `DDPStrategy` and `Trainer` classes, both of which can be found in `xpu.py` in case you want to look into them further.

For now though, it means we just have to change the above code so it looks like this:

```python
    strategy = xpu.DDPXPUStrategy(process_group_backend='nccl')

    trainer = xpu.Trainer.from_argparse_args(
        args,
        max_epochs=2,
        gradient_clip_val=1.0,
        callbacks=callback_list,
        enable_checkpointing=False,
        strategy=strategy,
    )
```

Note how we’ve changed the way the `Trainer` is instantiated and are now passing in a bespoke `DDPXPUStrategy` object to the trainer.

At this point you may also want to change the number of training epochs. The value of 2 will allow the training to complete rapidly, but won’t give good results.

### Code changes

That’s all of the changes we need to make to the code. We’ve gone through it in detail in case you want to apply a similar approach to other models.

There are three main purposes of the code changes here.

First we’re making the it compatible for use with the Intel GPUs (“XPUs”) that DAWN provides. The majority of the changes are to support the alternative underlying Intel OneAPI needed to use the Intel GPUs. The Intel OneAPI is based on SYCL and is an alternative to CUDA. Before running this code we’re going to install both the drivers and a version of PyTorch that’s been adapted for use with OneAPI. We’ll get on to this in the coming sections.

Second we’re providing support for the Intel OneAPI MPI framework that allows us to distribute training across multiple processes, multiple GPUs and multiple nodes. If we were only running on a single GPU or a single node we might have been able to avoid some of this, but we’ve tried to keep things as general as possible to support different use cases here.

Third we’ve added support for BFloat16. The Intel Max 1550 GPUs on DAWN are optimised for use with BFloat16 and in our experiments we experienced a significant speed-up over FP32 or FP16 when using BFloat16.

Having made all of these changes we can now run the code, but to do this we’ll want to use SLURM to submit the job. We’ll also need to set up our environment careful in order for MPI to work, so we’ll need to write a SLURM batch script. We’ll go through this in the next sections.

## Batch file

### Batch file submission

We’ve got our code in order, but there’s more to it: we also need to worry about drivers, the environment and dependencies. Our SLURM batch script will handle all of this for us.

You’ll find the `sbatch_example.py` file already available in the repository. The only thing you’ll need to do to get this to work is to change the `<ACCOUNT_ID>` on line 4 for the correct account that you’re using on DAWN.

Having made this change you can then submit the job to SLURM. The script assumes that it’ll be run from the directory it’s found in, so make sure you move to that directory first before submitting the script to SLURM:

```bash
cd gpt_lightning
sbatch sbatch_example.py
```

You can now monitor the status of the script by checking the queue:

```bash
squeue --me
```

Once the script is running it’ll send all `stdout` output to the file `job.out`. Once the job is finished you can load this file into your favourite text editor to check it. You can also follow execution while it’s running if you want to:

```bash
tail -f job.out
```

Although this is everything you need to run the example training, it’ll also be useful to examine the batch file in more detail. We’ll go through all of the elements of it in the upcoming sections.

### SLURM preamble

The SLURM header always lives at the top of your batch file. It provides details to SLURM about how to run your script, including the account to use, the maximum amount of time to give your task and the hardware to provision for it.

```shell
#!/bin/bash
#SBATCH --job-name litgpu
#SBATCH --output job.out
#SBATCH --account <ACCOUNT_ID>
#SBATCH --partition pvc
#SBATCH --time 1:00:0
#SBATCH --nodes 4
#SBATCH --ntasks-per-node 8
#SBATCH --cpus-per-task 12
#SBATCH --gres gpu:4
```

Let’s break this down:
1. `job-name`: this is for the user’s benefit. The job will appear in the queue with this name.
2. `output`: the file to save out the stdout output to.
3. `account`: This is the account to run the script on; you should change `<ACCOUNT_ID>` to the account id for your account. As a user you may have multiple projects and multiple accounts with different capabilities, so this allows you to choose the correct one.
4. `partition`: This selects the system to run your task on. Here `pvc` stands for Ponte Vecchio, the code name for the GPUs used by Dawn, so this tells SLURM to run your script on Dawn.
5. `time`: The maximum time your script is allowed to run, in the format hours:minutes:seconds. It doesn’t matter if your script finishes early, but the longer time you request the longer your script may end up waiting in the queue to be allocated resources.
6. `nodes`: The number of nodes to execute on. Each node has a maximum of four GPUs. Here we’re requesting four nodes so that we can run across sixteen GPUs.
7. `ntasks-per-node`: We want one task per “tile” on the GPU. Each GPU has two tiles and each node has four GPUs, so we want 2 * 4 = 8 tasks per node.
8. `cpus-per-task`: Here a CPU actually means a core. Each Dawn machine has 96 cores and we’re running 8 tasks per node, so here we’re asking for 8 * 12 = 96 cores. In other words, the full machine. In general you should scale the cores you need with the number of GPUs you’re requesting.
9. `gres`: “Generic Resource Scheduling”. We’re using this to specify the number of GPUs per node. So `gpu:4` here means four for each node, a total of 16 GPUs or 32 tiles. So many numbers!

Hopefully that clarifies the meaning, but for more detail, see the [recommended SLURM preamble](https://docs.hpc.cam.ac.uk/hpc/user-guide/pvc.html#recommendations-for-running-on-dawn) in the DAWN documentation, or if you really want the detail, check the SLURM documentation for [sbatch](https://slurm.schedmd.com/sbatch.html).

### Preparation

We then set a few arbitrary things up. If something goes wrong we want our script to bail rather than continue regardless. This is typically what you want so it’s worth adding this to the top of your batch scripts:

```shell
# Don't tolerate errors
set -e errexit
```

The working directory the script runs from will be the same as the directory you submit it from unless you specify otherwise using the `--chdir` argument when you call it. We assume you’ll submit the script from the directory it’s in. The next line then moves into the `lig-GPT` directory that’s contained with it.

```shell
# Directory to move into at the start
pushd ${PWD}/lit-GPT
```

It’s convenient to have an environment variable for where we plan to install our virtual environment. The next line sets this up.

```shell
# Define the path to the virtual environment relative to the CWD
VENV_PATH="./venv"
```

There are some numbers that it’s useful to have to hand because we need to use them when we make some calls later. We use the `$((expression))` pattern to ask the shell interpreter to perform some calculations for us.

For example, here the total processes is the number of nodes multiplied by the number of tasks per node. The environment variables that start `SLURM_...` will be set up automatically when SLURM runs the script, so this is a convenient way to set up environment variables that will change automatically depending on the values we put in the SLURM preamble.

```shell
# MPI arrangement from SLURM
CONFIG_PROCESSES_TOTAL=$((${SLURM_JOB_NUM_NODES}*${SLURM_NTASKS_PER_NODE}))
CONFIG_GPUS_PER_PROCESS=$((${SLURM_GPUS_ON_NODE}*2/${SLURM_NTASKS_PER_NODE}))
CONFIG_CPUS_PER_NODE=$((${SLURM_CPUS_PER_TASK}*${SLURM_NTASKS_PER_NODE}))
```

Given the SLURM preamble, these environment variables will end up being set as follows:

1. `CONFIG_PROCESSES_TOTAL` = 4 * 8 = 32
2. `CONFIG_GPUS_PER_PROCESS` = 4 * 2 / 8 = 1
3. `CONFIG_CPUS_PER_NODE` = 12 * 8 = 96

We’ll use these later.

For a full list of the environment variables that SLURM sets when it runs your script, see the [Output Environment Variables](https://slurm.schedmd.com/sbatch.html#SECTION_OUTPUT-ENVIRONMENT-VARIABLES) section of the sbatch documentation.

### Loading modules

The DAWN team have made a large number of different tools, libraries and frameworks available from DAWN using modules. Using a module is a bit like installing a package, except that because the files are already on the system they’re usually really fast and easy to activate.

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

### Socket limit

The Intel [“Collective Communications Library”](https://oneapi-src.github.io/oneCCL/) MPI mechanism that we use to communicate between nodes and processes relies on sockets for communication. This can result in a very large number of sockets being opened, more in fact than the default allowed by the operating system. The following command increases the limit to avoid our task exceeding it (which would cause it to fail).

```shell
# Avoid too many open file handles error
ulimit -n 1000000
```

### Environment variables

Next up we set a whole host of environment variables. Most of these are essentially default settings needed to get the communication library to work. Each variable has been annotated with a description and link to the documentation (or the closest we could find to documentation) that explains why it’s needed.

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

The last two environment variables are `MASTER_ADDR` and `MASTER_PORT`. These are the name and port for the master node that’s used to support communication between all of the other nodes. We’re get the name of the master node from SLURM and choose a random port.

### Setting up the build environment

The OneAPI documentation tells us that we need to run some scripts to set up the environment. Having activated the various modules we still need to do this and the following commands are essentially just following these requirements.

```shell
echo "Initialise environment"
pushd /usr/local/dawn/software/spack/spack-views/dawn-test-2023-12-22/
source intel-oneapi-compilers-2024.0.0/gcc-13.2.0/znjudqsiaf6x5u2rxdtymf6ss55nmimw/compiler/2024.0/env/vars.sh
source intel-oneapi-mkl-2024.0.0/oneapi-2024.0.0/4n7ruz44nhbsd5xp4nnz6mgm2z7vqzxs/mkl/2024.0/env/vars.sh
source intel-oneapi-compilers-2024.0.0/gcc-13.2.0/znjudqsiaf6x5u2rxdtymf6ss55nmimw/setvars.sh
popd
```

### Setting up the Virtual Environment

This next section is a bit of a mess, but it’s actually quite simple. The commands check whether the Python virtual environment directory already exists. If it doesn’t it creates the virtual environment and installed all of the requirements inside it, based on the list of packages in the various `pip` requirements files. If it does exist it assumes everything is already correctly installed and just activates the environment instead. 

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
  pip install torch==2.0.1a0 torchvision==0.15.2a0 intel-extension-for-pytorch==2.0.120+xpu oneccl-bind-pt==2.0.200 --extra-index-url https://pytorch-extension.intel.com/release-whl-aitools/
  source $(python -c "import oneccl_bindings_for_pytorch as torch_ccl;print(torch_ccl.cwd)")/env/vars.sh
  pip install -e .
  pip install -r requirements.txt
  pip install -r requirements/nanogpt.txt
fi
```

This means that the first time we run the batch file a suitable Python virtual environment will be created. On subsequent runs it will activate the virtual environment instead.

### Running the training

We’re finally there! The command here executes our script with appropriate parameters. There are some important things to note here.

We don’t just calling python with the script. We instead pass the python command to `mpirun`.

The `mpirun` command is used to execute something multiple times across different nodes. It will set up certain variables in the environments the processes are run so they can tell each other apart. For example, `mpirun` will set the `PMI_RANK` variable to a different number for each process and the `PMI_SIZE` environment variable to the total number of processes executed. You’ll notice that we used these in our `train.py` script earlier.

Recall from earlier that we set the following environment variables:

1. `CONFIG_PROCESSES_TOTAL` = 32
2. `CONFIG_GPUS_PER_PROCESS` = 1
3. `CONFIG_CPUS_PER_NODE` = 96

Consequently we’re calling `mpirun` to execute 32 processes and 8 processes per node.

We’re also passing in a bunch of parameters to our script which will get interpreted by PyTorch lightning. The values we’re setting are 32 nodes, 1 GPU per process and 96 workers. We also request BFloat16 precision, set the size of the model and the training hyperparameters.

```shell
echo "[Richard O'Brien] Will you start the training please!"
# See https://youtu.be/Wx4ExtpBbXc?feature=shared&t=2863
mpirun -n ${CONFIG_PROCESSES_TOTAL} -ppn ${SLURM_NTASKS_PER_NODE} -prepend-rank python train.py \
  --enable_progress_bar 0 \
  --num_nodes ${CONFIG_PROCESSES_TOTAL} \
  --devices ${CONFIG_GPUS_PER_PROCESS} \
  --num_workers $((${SLURM_CPUS_PER_TASK})) \
  --implementation "mingpt" \
  --model_type "None" \
  --strategy "ddp" \
  --n_layer 12 \
  --n_head 12 \
  --n_embd 768 \
  --precision "bf16" \
  --batch_size 16 \
  --max_epochs 2
```

At this point the training will start and hopefully everything will work successfully. This will generate a lot of output, most of which can be ignored.

At the end of each training epoch the time taken and memory usage is output. These come from the callbacks we set up earlier.

Finally, once training is complete, an inference step will be run and the model we’ve trained will generate some (probably very bad) “pretend Shakespeare”.

### Cleaning up

Once the training step has completed we deactivate our virtual environment and move into the directory we started from. These steps aren’t strictly necessary because the shell environment will be lost anyway, but it feels right to restore things.

```shell
deactivate
popd
echo "All done"
```

And that’s it! Hopefully walking through this can help understand what’s going on and translate the changes for other training tasks.
