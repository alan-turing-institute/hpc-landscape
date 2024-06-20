# Llama 2 inference on Baskerville using `sbatch`

Before you can make use of them you’ll need to download the models. See [llama-download.md](llama-download.md) for how to go about doing this on Baskerville.

## Llama 2 7B inference, single node, single GPU

The 7B parameter Llama 2 model will run with a single process on a single node with a single GPU. The checkpoint provided for use with the example will, in fact, only work with a single process.

We found at least 16 GiB of RAM was needed to run the model. With too little RAM the process will be sent a SIGKILL termination signal, the process will end with an exit code of -9 and an OOM report will be generated.

The RAM, node, GPU and process quantities must be explicitly configured through `sbatch` header for things to work.

## 1. Get the full path of your llama directory.

You should already have the llama git repository cloned to Baskerville. You’ll need the full path of where you cloned it to for the script. You can find this by moving into the directory and then running the `pwd` (print working directory) command:

```shell
$ cd llama
$ pwd
```

## 2. Create the batch script.

Using your favourite text editor, create a batch file with the contents below.

In this script you’ll need to replace `<PROJECT_ID>` with the name of your project and `<LLAMA2_PATH>` with the full path of your cloned llama repository from Step 1.

Make sure you call the script `batch-llama2-7b-inf.sh`. Technically you can give it any name, but I’ve assumed it’s called this in the steps that follow.

```shell
#!/bin/bash
#SBATCH --qos turing
#SBATCH --account <PROJECT_ID>
#SBATCH --time 0:30:0
#SBATCH --nodes 1
#SBATCH --gpus 1
#SBATCH --mem 16384
#SBATCH --job-name llama2-7b-inf

# Execute using:
# sbatch -o stdout-%A_%a.out ./batch-llama2-7b-inf.sh

module purge
module load baskerville
module load bask-apps/live
module load PyTorch/2.1.2-foss-2022b-CUDA-11.8.0

VENV_PATH="./venv"

pushd <LLAMA2_PATH>

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
    --ckpt_dir llama-2-7b-chat/ \
    --tokenizer_path tokenizer.model \
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

You’ll either need to create this batch file directly on Baskerville, or transfer it to Baskerville. It doesn’t matter where you store it on Baskerville.

## 3. Schedule the job file to run.

```shell
$ sbatch -o stdout-%A_%a.out ./batch-llama2-7b-inf.sh
```

## 4. Check the status of your job.

Use the following command to check the status of your job. Once it’s completed it will disappear from the queue.

```shell
$ squeue --me
```

This is a small job so it shouldn’t take too long to schedule and run (it took less than five minutes for me, but will depend on how busy Baskerville is when you run it).

## 5. View the results in realtime.

The console output from the job will be stored in a file with a name of the form `stdout-%A_%a.out` where the `%A` and `%a` are replaced by the job ID and array index respectively. For example, for me the file was called `stdout-763144_4294967294.out`.

If you want to follow the progress of the execution in real time &mdash; while it’s running that is &mdash; you can execute the following command to follow the output. If you have multiple output files you may need to specify the filename explicitly, rather than relying on the `*` wildcard.

```shell
tail -f stdout-*.out
```

## 6. View the results after completion.

Once the job has completed you can view open the `stdout-%A_%a.out` file to see the results. It should contain something like this:

```shell

```

## Llama 2 13B inference, single node, two GPUs

The 13B parameter Llama 2 model will run with tow processes on a single node with two GPUs. The checkpoint provided requires this configuration.

We found at least 16 GiB of RAM was needed to run the model. With too little RAM the process will be sent a SIGKILL termination signal, the process will end with an exit code of -9 and an OOM report will be generated.

The RAM, node, GPU and process quantities must be explicitly configured through `sbatch` header for things to work.

## 7. Create the batch script.

Using your favourite text editor, create a batch file with the contents below.

In this script you’ll need to replace `<PROJECT_ID>` with the name of your project and `<LLAMA2_PATH>` with the full path of your cloned llama repository from Step 1.

Make sure you call the script `batch-llama2-13b-inf.sh`. Technically you can give it any name, but I’ve assumed it’s called this in the steps that follow.

```shell
#!/bin/bash
#SBATCH --qos turing
#SBATCH --account <PROJECT_ID>
#SBATCH --time 0:30:0
#SBATCH --nodes 1
#SBATCH --gpus 2
#SBATCH --mem 16384
#SBATCH --job-name llama2-13b-inf

# Execute using:
# sbatch -o stdout-%A_%a.out ./batch-llama2-13b-inf.sh

module purge
module load baskerville
module load bask-apps/live
module load PyTorch/2.1.2-foss-2022b-CUDA-11.8.0

VENV_PATH="./venv"

pushd <LLAMA2_PATH>

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
    --nproc_per_node 2 \
    example_chat_completion.py \
    --ckpt_dir llama-2-13b-chat/ \
    --tokenizer_path tokenizer.model \
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

You’ll either need to create this batch file directly on Baskerville, or transfer it to Baskerville. It doesn’t matter where you store it on Baskerville.

## 3. Schedule the job file to run.

```shell
$ sbatch -o stdout-%A_%a.out ./batch-llama2-13b-inf.sh
```

## 4. Check the status of your job.

Use the following command to check the status of your job. Once it’s completed it will disappear from the queue.

```shell
$ squeue --me
```

This is a small job so it shouldn’t take too long to schedule and run (it took less than five minutes for me, but will depend on how busy Baskerville is when you run it).

## 5. View the results in realtime.

The console output from the job will be stored in a file with a name of the form `stdout-%A_%a.out` where the `%A` and `%a` are replaced by the job ID and array index respectively. For example, for me the file was called `stdout-763144_4294967294.out`.

If you want to follow the progress of the execution in real time &mdash; while it’s running that is &mdash; you can execute the following command to follow the output. If you have multiple output files you may need to specify the filename explicitly, rather than relying on the `*` wildcard.

```shell
tail -f stdout-*.out
```

## 6. View the results after completion.

Once the job has completed you can view open the `stdout-%A_%a.out` file to see the results. It should contain something like this:

```shell

```

## Llama 2 70B inference, two nodes, eight GPUs

The 70B parameter LLama 2 model requires considerably more resources to execute compared to the 8B parameter model. The checkpoint provided for the example script requires eight processes and eight GPUs. On Baskerville this means we’ll need to use two nodes, each with four GPUs.

Thankfully it’s considerably easier to do this using `sbatch` compared to `srun`. Let’s do it now.

## 7. Create an execution wrapper.

Because we’re using `torchrun` we have to create an execution wrapper so that we can pass in the node ranks. SLURM will set a different value to `${SLURM_NODEID}` for each wrapper running on each node. We need to pass these different values on to `torchrun`.

You should create this script in the root of your `llama` repository and call it `wrapper.sh`. You can either create it directly on Baskerville or create it somewhere else (on your local machine?) and transfer it over.

```shell
#!/bin/bash

python -m torch.distributed.run \
    --nnodes ${SLURM_NNODES} \
    --nproc_per_node ${SLURM_GPUS_PER_NODE} \
    --master_addr ${PRIMARY_ADDR} \
    --master_port ${PRIMARY_PORT} \
    --node_rank ${SLURM_NODEID} \
    example_chat_completion.py \
    --ckpt_dir llama-2-70b-chat/ \
    --tokenizer_path tokenizer.model \
    --max_seq_len 512 \
    --max_batch_size 6
```

## 8. Create the batch script.

Using your favourite text editor, create a batch file with the contents below.

In this script you’ll need to replace `<PROJECT_ID>` with the name of your project and `<LLAMA2_PATH>` with the full path of your cloned llama repository from Step 1.

Make sure you call the script `batch-llama2-70b-inf.sh`. Technically you can give it any name, but I’ve assumed it’s called this in the steps that follow.

```shell
#!/bin/bash
#SBATCH --qos turing
#SBATCH --account vjgo8416-ml-workload
#SBATCH --time 0:30:0
#SBATCH --nodes 2
#SBATCH --gpus-per-node 4
#SBATCH --ntasks-per-node 1
#SBATCH --mem 131072
#SBATCH --job-name llama2-llama2-70b-inf

# Execute using:
# sbatch -o stdout-%A_%a.out ./batch-llama2-70b-inf.sh

module purge
module load baskerville
module load bask-apps/live
module load PyTorch/2.1.2-foss-2022b-CUDA-11.8.0

VENV_PATH="./venv"

pushd /bask/projects/v/vjgo8416-ml-workload/llama/llama

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

srun wrapper.sh

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

## 9. Schedule the job file to run.

```shell
$ sbatch -o stdout-%A_%a.out ./batch-llama2-70b-inf.sh
```

## 10. Follow the results.

As before you can check the status of your job using the following:

```shell
squeue --me
```

You can also check the output of the `stdout-%A_%a.out` file to see the results, either by following it during execution or opening the file afterwards.
