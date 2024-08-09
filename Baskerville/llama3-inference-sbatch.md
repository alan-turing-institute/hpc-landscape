# Llama 3 inference on Baskerville using `sbatch`

Before you can make use of them you’ll need to download the models.
See [llama-download.md](llama-download.md) for how to go about doing this on Baskerville.

## Llama 3 8B inference, single node, single GPU

The 8B parameter Llama 3 model will run with a single process on a single node with a single GPU.
The checkpoint provided for use with the example will, in fact, only work with a single process.

We found at least 32 GiB of RAM was needed to run the model.
With too little RAM the process will be sent a SIGKILL termination signal, the process will end with an exit code of -9 and an OOM report will be generated.

The RAM, node, GPU and process quantities must be explicitly configured through `sbatch` header for things to work.

## 1. Get the full path of your llama3 directory.

You should already have the llama3 git repository cloned to Baskerville.
You’ll need the full path of where you cloned it to for the script.
You can find this by moving into the directory and then running the `pwd` (print working directory) command:

```shell
$ cd llama3
$ pwd
```

## 2. Create the batch script.

Using your favourite text editor, create a batch file with the contents below.

In this script you’ll need to replace `<PROJECT_ID>` with the name of your project and `<LLAMA3_PATH>` with the full path of your cloned llama3 repository from Step 1.

If it doesn't already exist, a virtual environment will be created and configured automatically in a directory called `venv` immediately inside the directory you set `<LLAMA3_PATH>` to.

Make sure you call the script `batch-llama3-8b-inf.sh`.
Technically you can give it any name, but I’ve assumed it’s called this in the steps that follow.

```shell
#!/bin/bash
#SBATCH --qos turing
#SBATCH --account <PROJECT_ID>
#SBATCH --time 0:30:0
#SBATCH --nodes 1
#SBATCH --gpus 1
#SBATCH --mem 32768
#SBATCH --job-name llama3-8b-inf

# Execute using:
# sbatch -o stdout-%A_%a.out ./batch-llama3-8b-inf.sh

module purge
module load baskerville
module load bask-apps/live
module load PyTorch/2.1.2-foss-2022b-CUDA-11.8.0

VENV_PATH="./venv"

pushd <LLAMA3_PATH>

if [ -d "$VENV_PATH" ]; then
    echo "Virtual environment exists, activating"
    . "${VENV_PATH}"/bin/activate
else
    echo "Creating and activating virtual environment"
    python3 -m venv "${VENV_PATH}"
    . "${VENV_PATH}"/bin/activate
    echo "Installing requirements"
    pip install pip --upgrade
    pip install -e .
    pip install -r requirements.txt
fi

export OMP_NUM_THREADS=1

echo
echo "######################################"
echo "Starting"
echo "######################################"
echo

python -m torch.distributed.run \
    --standalone \
    --nnodes 1 \
    --nproc_per_node 1 \
    example_chat_completion.py \
    --ckpt_dir Meta-Llama-3-8B-Instruct/ \
    --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model \
    --max_seq_len 512 \
    --max_batch_size 6

echo
echo "######################################"
echo "Done"
echo "######################################"
echo

echo "Deactivating virtual environment"
deactivate
popd
```

You’ll either need to create this batch file directly on Baskerville, or transfer it to Baskerville.
It doesn’t matter where you store it on Baskerville.

## 3. Schedule the job file to run.

```shell
$ sbatch -o stdout-%A_%a.out ./batch-llama3-8b-inf.sh
```

## 4. Check the status of your job.

Use the following command to check the status of your job.
Once it’s completed it will disappear from the queue.

```shell
$ squeue --me
```

This is a small job so it shouldn’t take too long to schedule and run (it took less than five minutes for me, but will depend on how busy Baskerville is when you run it).

## 5. View the results in realtime.

The console output from the job will be stored in a file with a name of the form `stdout-%A_%a.out` where the `%A` and `%a` are replaced by the job ID and array index respectively.
For example, for me the file was called `stdout-763144_4294967294.out`.

If you want to follow the progress of the execution in real time &mdash; while it’s running that is &mdash; you can execute the following command to follow the output.
If you have multiple output files you may need to specify the filename explicitly, rather than relying on the `*` wildcard.

```shell
tail -f stdout-*.out
```

## 6. View the results after completion.

Once the job has completed you can view open the `stdout-%A_%a.out` file to see the results.
It should contain something like this:

```text
[...]
==================================

System: Always answer with Haiku

User: I am going to Paris, what should I see?

> Assistant: Eiffel's iron kiss
River Seine's gentle whispers
Art in every stone

==================================
[...]
```

## Llama 3 70B inference, two nodes, eight GPUs

The 70B parameter LLama 3 model requires considerably more resources to execute compared to the 8B parameter model.
The checkpoint provided for the example script requires eight processes and eight GPUs.
On Baskerville this means we’ll need to use two nodes, each with four GPUs.

Thankfully it’s considerably easier to do this using `sbatch` compared to `srun`.
Let’s do it now.

## 7. Create the batch script.

Using your favourite text editor, create a batch file with the contents below.

In this script you’ll need to replace `<PROJECT_ID>` with the name of your project and `<LLAMA3_PATH>` with the full path of your cloned llama3 repository from Step 1.

Make sure you call the script `batch-llama3-70b-inf.sh`.
Technically you can give it any name, but I’ve assumed it’s called this in the steps that follow.

```shell
#!/bin/bash
#SBATCH --qos turing
#SBATCH --account vjgo8416-ml-workload
#SBATCH --time 0:30:0
#SBATCH --nodes 2
#SBATCH --gpus-per-node 4
#SBATCH --ntasks-per-node 1
#SBATCH --mem 131072
#SBATCH --job-name llama3-70b-inf

# Execute using:
# sbatch -o stdout-%A_%a.out ./batch-llama3-70b-inf.sh

module purge
module load baskerville
module load bask-apps/live
module load PyTorch/2.1.2-foss-2022b-CUDA-11.8.0

VENV_PATH="./venv"

pushd /bask/projects/v/vjgo8416-ml-workload/llama/llama3

if [ -d "$VENV_PATH" ]; then
    echo "Virtual environment exists, activating"
    . "${VENV_PATH}"/bin/activate
else
    echo "Creating and activating virtual environment"
    python3 -m venv "${VENV_PATH}"
    . "${VENV_PATH}"/bin/activate
    echo "Installing requirements"
    pip install pip --upgrade
    pip install -e .
    pip install -r requirements.txt
fi

export PRIMARY_PORT=12340
export PRIMARY_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export OMP_NUM_THREADS=1

echo
echo "######################################"
echo "Starting"
echo "######################################"
echo

srun bash -c \
    ‘python -m torch.distributed.run \
    --nnodes ${SLURM_NNODES} \
    --nproc_per_node ${SLURM_GPUS_PER_NODE} \
    --master_addr ${PRIMARY_ADDR} \
    --master_port ${PRIMARY_PORT} \
    --node_rank ${SLURM_NODEID} \
    example_chat_completion.py \
    --ckpt_dir Meta-Llama-3-70B-Instruct/ \
    --tokenizer_path Meta-Llama-3-70B-Instruct/tokenizer.model \
    --max_seq_len 512 \
    --max_batch_size 6’

echo
echo "######################################"
echo "Done"
echo "######################################"
echo

echo "Deactivating virtual environment"
deactivate
popd
```

Once again, you’ll need to either create this script directly on Baskerville or copy it over.

There are a couple of important things to note about this script.
First, because we’re running across multiple nodes using `torchrun` we have to pass our training command through `srun`.
This will ensure multiple processes get spawned, one on each node.
The syntax for `srun` is to execute the command passed as a parameter on each of the nodes.

We also need to pass various parameters to our Python script, some of which are taken from environment variables.
The second point to note is about these parameters.
Most of the parameters can be set to the same value for each process, but there’s one exception, which is the value passed as the node rank: `--node_rank ${SLURM_NODEID}`.
This needs to take a different value on each node so that the nodes can communicate with each other successfully.

The `SLURM_NODEID` environment variable will automatically be set to a different value on each node by `srun`.
However this means we need to be careful about how we pass the command to `srun`.
If we passed the `python` command directly, this `SLURM_NODEID` will be resolved on the spawning node and its value will then end up the same on each node.

Consequently we have to create a string out of our entire command and use `bash` to run the command.
That way `srun` will spawn the `bash` shell on each node, which will then interpret the `python` command on that node.
The `SLURM_NODEID` environment variable won’t then get resolved until the command is run on the node, which is what we need.

## 8. Schedule the job file to run.

```shell
$ sbatch -o stdout-%A_%a.out ./batch-llama3-70b-inf.sh
```

## 9. Follow the results.

As before you can check the status of your job using the following:

```shell
squeue --me
```

You can also check the output of the `stdout-%A_%a.out` file to see the results, either by following it during execution or opening the file afterwards.

## 10. View the results after completion.

Once the job has completed you can view open the `stdout-%A_%a.out` file to see the results. It should contain something like this:

```text
[...]
==================================

System: Always answer with Haiku

User: I am going to Paris, what should I see?

> Assistant: Eiffel's iron lace
River Seine's gentle whisper
Montmartre's charm waits

==================================
[...]
```

