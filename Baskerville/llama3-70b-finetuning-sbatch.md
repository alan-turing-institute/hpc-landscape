# Llama 3 70B finetuning on Baskerville

For this walkthrough we'll use LLaMA-Factory to fine tune the LLama 3 70b model using 4 nodes and 16 GPUs on Baskerville.

[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory.git) will do all of the hard work, including downloading and deploying the model from Hugging Face.
It uses `accelerate` to distribute the compute across nodes and GPUs and there's a little configuration needed to get this set up.

At time of writing the 70b parameter Llama 3 model is still considered relatively large (at least in terms of open source models).
Finetuning the model therefore requires a decent allocation of resources.
We'll allocate four nodes on Baskerville for this, each offering four A100 40 GiB GPUs.
It's also possible to train on two nodes each with four A100 80 GiB GPUs, but the size of the model necessitates the equivalent of 320 GiB of GPU RAM to support training.

Elsewhere we've walked through how to perform [finetuning of the LLama3 3b model](../DAWN/examples/llama-factory/llamafactory-finetuning-sbatch-multi.md) on DAWN and [inference using the Llama3 70b model](./llama3-inference-sbatch.md) on Baskerville.
The former, using the smaller model, used just one node, four GPUs and 512 GiB of GPU RAM.
The latter being the larger model but only concerned with inference required two nodes, eight GPUs and 320 GiB of GPU RAM.
Finetuning of the larger model requires scaling up beyond these.

## 1. Redirecting your Hugging Face cache

Since the model is quite large you'll need to store it in project space on Baskerville rather than your home directory.
Throughout this walkthrough we'll use `<PROJECT_PATH>` to refer to your project directory; you should replace it with the full path.

We're going to create a directory `llama-factory` inside the project directory for our work.
You're welcome to use a different location, in which case you'll just need to adjust the commands accordingly.
The only constraint is that as just discussed this directory should be in your project space (or at least somewhere with enough space to store all the data).

```shell
$ mkdir -p <PROJECT_PATH>/llama-factory
```

Next up we're going to create a symlink from the cache directory used by Hugging Face in our home directory to the new directory in our project space.
This is where the model is downloaded to so will ensure it ends up in our project space and not our home directory.
If you've already configured Hugging Face to store its models appropriately you can skip this step.

```shell
$ ln -s <PROJECT_PATH>/llama-factory/cache/huggingface ~/.cache/huggingface/
```

## 2. Clone the LLaMA-Factory repository

If you don't already have it you'll need to clone the LLaMA-Factory repository to your project folder.

```shell
$ cd <PROJECT_PATH>/llama-factory
$ git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
```

## 3. Download scripts

In order to get this to work we're going to need an `accelerate` configuration and a batch file to pass to `sbatch`.
We'll download them now so they're ready for use later.

We'll place them in the directory that we just cloned LLaMA-Factory in to, so no need to change directory.

```shell
$ curl -O https://raw.githubusercontent.com/alan-turing-institute/hpc-landscape/main/Baskerville/scripts/config_node02gpu04.yaml
$ curl -O https://raw.githubusercontent.com/alan-turing-institute/hpc-landscape/main/Baskerville/scripts/batch-llama3-70b-ft.sh
```

The [config file](./scripts/config_node02gpu04.yaml) looks like this and was created using `accelerate config` on a compute node and then tailored for the particular task:

```yaml
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: FSDP
downcast_bf16: 'no'
enable_cpu_affinity: false
fsdp_config:
  fsdp_activation_checkpointing: false
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch: BACKWARD_PRE
  fsdp_cpu_ram_efficient_loading: true
  fsdp_forward_prefetch: false
  fsdp_offload_params: false
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_sync_module_states: true
  fsdp_use_orig_params: true
main_training_function: main
mixed_precision: 'bf16'
num_machines: 4
num_processes: 16
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

We'll look at the batch file in more detail later.

## 4. Obtain a Hugging Face token

For the first of these steps you'll need to visit the [Meta-Llama-3.1-70B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-70B-Instruct) repository on Hugging Face and request access to the model.
I found it took around an hour for me request to be granted.

You should then create an access token via the [Tokens page](https://huggingface.co/settings/tokens) on Hugging Face.
Create a Read Only token and take a copy of it, then run the commands below.
We'll create a virtual environment as part of this, which should be done from inside the `LLaMA-Factory` directory you cloned in the previous step.

```shell
$ cd LLaMA-Factory
$ python3 -m venv venv
$ . ./venv/bin/activate
$ pip install --upgrade huggingface_hub
$ huggingface-cli login
```

You'll be requested to enter the token at the command line.
When it asks you whether to add the token as a git credential you can answer "no".

Once you've done this you can deactivate delete your virtual environment.

```shell
$ deactivate
$ rm -rf venv
```

## 5. Finetuning using `srun`

Using `sbatch` to finetune the model is a lot easier than using `srun`.
On the other hand understanding what's going on is far clearer using `srun` because you have to complete more of the steps manually.
If you just want to go ahead and finetune the module using `sbatch` you can skip ahead to Section 6 below.

### 5.1. Allocate resources

To start we'll need to allocate us some nodes.
We'll use two nodes each with four 80 GiB A100 GPUs.
If we were to use the 40 GiB variants we'd need more nodes which adds complexity because we need to run commands from each node separately.
When we use `sbatch` in the next section we'll use four 40 GiB nodes instead to avoid the potentially longer queue times with the 80 GiB model.

The following command will request these nodes.
You'll need to replace `<PROJECT_ID>` with the name of your project.

```shell
$ cd <PROJECT_PATH>/llama-factory/LLaMA-Factory
$ srun --account <PROJECT_ID> --qos turing --time 2:00:0 --nodes=2 --gpus=8 --mem 491520 --constraint a100_80 --pty /bin/bash
```

Some things to note about this command:
1. We've allocated two hours for the sake of demonstration, but in practice training will require a lot longer than this.
2. We've requested 480 GiB of CPU memory.
   We'll need it all.
3. The `constraint` parameter is where we specify the need for the 80 GiB variant of the A100 as `a100_80`.
   See the [Baskerville docs](https://docs.baskerville.ac.uk/jobs/#available-gpus) for more info on this.

You may have to wait a while before your allocation is granted.
To check the status of your allocation you can run the following.

```shell
$ squeue --me
```

Sadly there's no command to make the queue clear more quickly.

Once your request has been allocated you'll immediately be logged in to the first node allocated.
The command prompt will change from the name of the login node to the name of a compute node, something like this:

```shell
[user1234@bask-pg-login01 LLaMA-Factory]$ srun --account project --qos turing --time 2:00:0 --nodes=2 --gpus=8 --mem 491520 --constraint a100_80 --pty /bin/bash
srun: job 891180 queued and waiting for resources
srun: job 891180 has been allocated resources
[user1234@bask-pg0308u15a LLaMA-Factory]$ 
```

### 5.2. Configure the environment on the primary node

Once you've gained access to a compute node you should run these commands from inside it to load in the appropriate modules needed for running the finetuning:

```shell
$ module purge
$ module load baskerville
$ module load bask-apps/live
$ module load PyTorch/2.0.1-foss-2022a-CUDA-11.7.0
```

You'll also need to create a virtual environment and install dependencies in it:

```shell
$ python3 -m venv venv
$ . ./venv/bin/activate
$ pip install pip --upgrade
$ pip install -r requirements.txt
$ pip install -e ".[torch,metrics]"
$ pip install --upgrade huggingface_hub
```

On future occasions you can just activate the virtual environment rather than having to reinstall everything.

We also want to set up a couple of environment variables which we'll use for MPI to allow communication between nodes.

```shell
$ export PRIMARY_PORT=12340
$ export PRIMARY_ADDR=$(scontrol show hostnames $(squeue --me -h -o "%N" -t R | grep $(hostname)) | head -n 1)
```

### 5.3. Open a terminal on the secondary node

You requested and will have been allocated two compute nodes.
So far we've only performed steps on one of the nodes and before we set the finetuning running we'll need to set up the second node as well.

To find the name of your nodes run the following command on the compute node you're already logged in to:

```shell
$ scontrol show hostnames "$SLURM_JOB_NODELIST"
$ $(scontrol show hostnames $(squeue --me -h -o "%N" -t R | grep $(hostname)) | head -n 1)
```

You'll get a comma separated list of nodes.
For example, I get the following:

```shell
bask-pg0308u25a,bask-pg0308u33a
```

We'll refer to these as `<PRIMARY_NODE>` and `<SECONDARY_NODE>`, which you should replace below with the actual name of your nodes.
You may find it useful to take a note of the names for later, but if at any time you forget, just run `squeue --me` to see them again.

Open a second terminal on Baskerville (either using gnu `screen`, `tmux` or by opening another terminal and logging in there).
Use this second terminal to SSH into the second node:

```shell
$ ssh <SECONDARY_NODE>
```

### 5.4. Configure the environment on the secondary node

Since this is a completely separate device we'll need to set up the dependencies on it again:

```shell
$ module purge
$ module load baskerville
$ module load bask-apps/live
$ module load PyTorch/2.0.1-foss-2022a-CUDA-11.7.0
```

However as we do share directory space we can reuse the virtual environment from the primary node.

```shell
$ cd <PROJECT_PATH>/llama-factory/LLaMA-Factory
$ . ./venv/bin/activate
```

We also need to set up the environment variables.
Note that the values of these variables will be the same on both nodes: that's intentional.

```shell
$ export PRIMARY_PORT=12340
$ export PRIMARY_ADDR=$(scontrol show hostnames $(squeue --me -h -o "%N" -t R | grep $(hostname)) | head -n 1)
```

### 5.5 Execute on the primary node

Now we're ready to execute the finetuning.
In the terminal for your primary node execute the following command:

```shell
$ accelerate launch \
    --config_file ../config_node02gpu04.yaml \
    --main_process_ip ${PRIMARY_ADDR} \
    --main_process_port ${PRIMARY_PORT} \
    --machine_rank 0 \
    --num_processes 8 \
    --num_machines 2 \
    src/train.py examples/train_lora/llama3_lora_sft_70b.yaml
```

Execution will pause until you also start execution on the secondary node in the next step.

### 5.6. Execute on the secondary node

In the terminal for your secondary node execute the following command:

```shell
$ accelerate launch \
    --config_file ../config_node02gpu04.yaml \
    --main_process_ip ${PRIMARY_ADDR} \
    --main_process_port ${PRIMARY_PORT} \
    --machine_rank 1 \
    --num_processes 8 \
    --num_machines 2 \
    src/train.py examples/train_lora/llama3_lora_sft_70b.yaml
```

Note that this is almost identical to the command used on the primary node.
The difference is that we've changed the `machine_rank` from `0` to `1`.

Execution will now continue on both nodes.
Eventually either the finetuning will complete or your session time will expire.
In the later case your processes will be killed and finetuning halted as a result.
You can go through the same steps again, amending things and with a longer session time as necessary, to then complete the full finetuning training.
However in practice we wouldn't recommend using this `srun` approach for real work, it's really only useful for demonstration purposes.
To properly finetune the model you should use the `sbatch` approach described below.

## 6. Finetuning using `sbatch`

In Section 5 we looked at how to finetune the model manually by allocating a couple of nodes and executing the training code on both.
In this section we'll do the same but using `sbatch` so that the job can be added to the queue and will run automatically as soon as the resources become available.

The benefit of this approach is that it requires no manual intervention and so:
1. We don't have to hang around to wait for our job to reach the top of the queue, we can just let it wait and leave it to run.
2. Since the process doesn't require manual intervention, we can distribute the task across many more nodes much more easily.

The trick is in crafting an appropriate batch script to do all this.
Note that our batch script makes use of the `config_node02gpu04.yaml` configuration file that we downloaded and examined in Section 3 above.

### 6.1. Examine the batch script

Below is the [batch file](./scripts/batch-llama3-70b-ft.sh) we'll be using to run the LLaMA-Factory finetuning.

We'll examine it in a bit more detail, but before you use it you must edit it to replace the `<PROJECT_ID>` and `<PROJECT_PATH>` with your project identifier and project path respectively.


```shell
#!/bin/bash
#SBATCH --qos turing
#SBATCH --account <PROJECT_ID>
#SBATCH --time 3:00:0
#SBATCH --nodes 4
#SBATCH --gpus-per-node 4
#SBATCH --ntasks-per-node 1
#SBATCH --mem 491520
#SBATCH --job-name llama3-70b-ft

# Execute using:
# sbatch batch-llama3-70b-ft.sh

module purge
module load baskerville
module load bask-apps/live
module load PyTorch/2.1.2-foss-2022b-CUDA-11.8.0

VENV_PATH="./venv"

pushd <PROJECT_PATH>/llama-factory/LLaMA-Factory

if [ -d "$VENV_PATH" ]; then
    echo "Virtual environment exists, activating"
    . "${VENV_PATH}"/bin/activate
else
    echo "Creating and activating virtual environment"
    python3 -m venv "${VENV_PATH}"
    . "${VENV_PATH}"/bin/activate
    echo "Installing requirements"
    pip install pip --upgrade
    pip install -r requirements.txt
    pip install -e ".[torch,metrics]"
    pip install --upgrade huggingface_hub
fi

export PRIMARY_PORT=12340
export PRIMARY_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export NUM_PROCESSES=$((${SLURM_JOB_NUM_NODES}*${SLURM_GPUS_PER_NODE}))

echo
echo "######################################"
echo "Starting"
echo "######################################"
echo

# Track GPU metrics
stdbuf -o0 nvidia-smi dmon -o TD -s puct -d 1 > dmon.txt &

# Track CPU metrics
stdbuf -o0 vmstat -t 1 -y > cpu.txt &

# Run the task
srun bash -c \
    'accelerate launch \
    --config_file ../config_node02gpu04.yaml \
    --main_process_ip ${PRIMARY_ADDR} \
    --main_process_port ${PRIMARY_PORT} \
    --machine_rank ${SLURM_PROCID} \
    --num_processes ${NUM_PROCESSES} \
    --num_machines ${SLURM_JOB_NUM_NODES} \
    src/train.py examples/train_lora/llama3_lora_sft_70b.yaml'

echo
echo "######################################"
echo "Done"
echo "######################################"
echo

echo "Deactivating virtual environment"
deactivate
```

Let's look at what this file is doing.

Lines 1-9 are the SLURM header.
These specify the resources required: four nodes, each with four GPUs, 480 GiB CPU RAM per node and with a running time of three hours.

Lines 14-17 load in the required modules.

Lines 19-35 set up and populate the virtual environment, or just activate it if there already is one.

Lines 37-set up environment variables.
Specifically the commands pull out the hostname of the primary node, set the port to use and calculate the total number of processes needed based on the number of nodes and number of GPUs allocated.

The module loading, virtual environment configuration and setting of environment variables are equivalent to the steps from Sections 5.2 and 5.4 above.

Lines 47-48 run `nvidia-smi` to monitor GPU usage.
Data are collected once per second and output to the `dmon.txt` file.

Lines 50-51 run `vmstat` to monitor CPU usage.
Data are collected once per second and output to the `cpu.txt` file.

Lines 53-62 call `accelerate` to run the finetuning script.
Something to note about this command is that we're using `srun` to ensure it gets executed on *all four nodes* and we send the command as a parameter to a separate `bash` shell so that the environment variables are dereferenced separately on each node rather than being dereferenced before being distributed.
This is crucial to ensure that `${SLURM_PROCID}` takes a different value for each node.

This execution step is equivalent to the steps from Sections 5.5 and 5.6 above.

### 6.2. Schedule the job file to run

Once we've updated the values in the batch file and are happy with the configuration we can schedule it to run using `sbatch`:

```shell
$ sbatch batch-llama3-70b-ft.sh
```

### 6.3. Check the status of your job

Use the following command to check the status of your job.
Once it’s completed it will disappear from the queue.

```shell
$ squeue --me
```

This is quite a large job so it could take quite a while before it actually runs.
I had to leave it to run overnight.

### 6.4. View the results in realtime

The console output from the job will be stored in a file with a name of the form `stdout-*.out` where the `*` will be replaced by the job ID.
For example, for me the file was called `stdout-890671.out`.

If you want to follow the progress of the execution in real time &mdash; while it’s running that is &mdash; you can execute the following command to follow the output.
If you have multiple output files you may need to specify the filename explicitly, rather than relying on the `*` wildcard.

```shell
$ tail -f stdout-*.out
```

### 6.5. View the results after completion

Once the job has completed you can view open the `stdout-*.out` file to see the results.
If all has gone well it should contain something like this:

```log
[...]
[INFO|trainer.py:2134] 2024-09-17 23:35:34,372 >> ***** Running training *****
[INFO|trainer.py:2135] 2024-09-17 23:35:34,372 >>   Num examples = 981
[INFO|trainer.py:2136] 2024-09-17 23:35:34,372 >>   Num Epochs = 3
[INFO|trainer.py:2137] 2024-09-17 23:35:34,372 >>   Instantaneous batch size per device = 1
[INFO|trainer.py:2140] 2024-09-17 23:35:34,372 >>   Total train batch size (w. parallel, distributed & accumulation) = 128
[INFO|trainer.py:2141] 2024-09-17 23:35:34,372 >>   Gradient Accumulation steps = 8
[INFO|trainer.py:2142] 2024-09-17 23:35:34,372 >>   Total optimization steps = 21
[INFO|trainer.py:2143] 2024-09-17 23:35:34,409 >>   Number of trainable parameters = 6,471,680
  0%|          | 0/21 [00:00<?, ?it/s]
  5%|▍         | 1/21 [49:47<16:35:57, 2987.90s/it]
 10%|▉         | 2/21 [1:40:05<15:51:43, 3005.47s/it]
 14%|█▍        | 3/21 [2:28:37<14:48:50, 2962.79s/it]
slurmstepd: error: *** STEP 890671.0 ON bask-pg0309u04a CANCELLED AT 2024-09-18T02:12:23 DUE TO TIME LIMIT ***
[...]
```

As you can see the expected duration for running a single epoch is around 17 hours.
Three epochs would therefore run for over 50 hours and here you can see the job was terminated before this, after running for three hours.

## 7. Wrap Up

In this walkthrough we looked at how to perform finetuning of the Llama 3 70b model on Baskerville.
We first looked at how to do this manually using `srun` followed by how to do so automatically using `sbatch`.

In our tests finetuning was slow and it looked like there was a lot of scope to optimise the process.
Our aim here is just to show how it can be made to run; the optimisation is left to the reader.

