# Llama 2 inference on Baskerville using `srun`

Before you can make use of them you’ll need to download the models.
See [llama-download.md](llama-download.md) for how to go about doing this on Baskerville.

## Llama 2 7B inference, single node, single GPU

The 7B parameter Llama 2 model will run with a single process on a single node with a single GPU.
The checkpoint provided for use with the example will, in fact, only work with a single process.

We found at least 16 GiB of RAM was needed to run the model.
With too little RAM the process will be sent a SIGKILL termination signal, the process will end with an exit code of -9 and an OOM report will be generated.

The RAM, node, GPU and process quantities must be explicitly configured using `srun` for things to work.

### 1. Log in to a single node with a single GPU and a suitable configuration:

You’ll need to replace `<PROJECT_ID>` with the name of your project.

```shell
$ srun --qos turing \
    --account <PROJECT_ID> \
    --time 0:30:0 \
    --nodes 1 \
    --gpus 1 \
    --mem 16384 \
    --pty bash
```

### 2. Set up the environment on the node:

Note that the very first command here moves in to the `llama` directory, which is the directory containing the `example_text_completion.py` file.
We have to be inside this directory for things to work.

```shell
$ cd llama
$ module purge
$ module load baskerville
$ module load bask-apps/live
$ module load PyTorch/2.1.2-foss-2022b-CUDA-11.8.0
$ python -m venv venv
$ . ./venv/bin/activate
$ pip install pip --upgrade
$ pip install -e .
$ pip install -r requirements.txt
```

### 3. Run the example script that performs a bunch of inference steps:

```shell
$ python -m torch.distributed.run \
    --standalone \
    --nnodes 1 \
    --nproc_per_node 1 \
    example_chat_completion.py \
    --ckpt_dir llama-2-7b-chat/ \
    --tokenizer_path tokenizer.model \
    --max_seq_len 512 \
    --max_batch_size 6
```

You should see some output that looks something like this:

```text
[...]
Loaded in 20.01 seconds
[...]
==================================

System: Always answer with Haiku

User: I am going to Paris, what should I see?

> Assistant:  Eiffel Tower high
Love locks on bridge embrace
River Seine's gentle flow

==================================
[...]
```

One things to note is that in the Llama docs it [recommends](https://github.com/meta-llama/llama?tab=readme-ov-file#quick-start) to use `torchrun` to execute the script.
When `torchrun` is used directly in this way it doesn’t support Python virtual environments because it defaults to the system-installed Python.
To circumvent this we’ve switched `torchrun` out for `python -m torch.distributed.run` instead.

### 4. Tidy up

Once you’re done don’t forget to log out of the node so that it can be released for access by someone else.
If you forget to do this the node will automatically close your session once the full 30 minutes you requested are up.

```
$ deactivate
$ exit
```

## Llama 2 13B inference, single node, two GPUs

The 13B parameter Llama 2 model will run with two processes on a single node with two GPUs.
This arrangement is a requirement for the checkpoint to work.

We found at least 32 GiB of RAM was needed to run the model.
With too little RAM the process will be sent a SIGKILL termination signal, the process will end with an exit code of -9 and an OOM report will be generated.

The RAM, node, GPU and process quantities must be explicitly configured using `srun` for things to work.

### 5. Log in to a single node with a single GPU and a suitable configuration:

You’ll need to replace `<PROJECT_ID>` with the name of your project.

```shell
$ srun --qos turing \
    --account <PROJECT_ID> \
    --time 0:30:0 \
    --nodes 1 \
    --gpus 2 \
    --mem 32768 \
    --pty bash
```

### 6. Set up the environment on the node:

We assume you’re already in the `llama` directory as in step 2 above.
If not you should use `cd` to move there first.

```shell
$ module purge
$ module load baskerville
$ module load bask-apps/live
$ module load PyTorch/2.1.2-foss-2022b-CUDA-11.8.0
$ . ./venv/bin/activate
```

We’re also assuming here that the virtual environment was already created in step 2 above, so we just activate it.
If you’ve not already done this then you’ll need to run the commands there to create it as well.

### 7. Run the example script that performs a bunch of inference steps:

```shell
$ python -m torch.distributed.run \
    --standalone \
    --nnodes 1 \
    --nproc_per_node 2 \
    example_chat_completion.py \
    --ckpt_dir llama-2-13b-chat/ \
    --tokenizer_path tokenizer.model \
    --max_seq_len 512 \
    --max_batch_size 6
```

You should see some output that looks something like this:

```text
[...]
Loaded in 27.98 seconds
[...]
==================================

System: Always answer with Haiku

User: I am going to Paris, what should I see?

> Assistant:  Eiffel Tower high
River Seine's gentle flow
Art and love in air

==================================
[...]
```

### 8. Tidy up

Once you’re done don’t forget to log out of the node so that it can be released for access by someone else.
If you forget to do this the node will automatically close your session once the full 30 minutes you requested are up.

```
$ deactivate
$ exit
```

## Llama 2 70B inference, two nodes, eight GPUs

The 70B parameter LLama 2 model requires considerably more resources to execute compared to the smaller Llama 2 models.
The checkpoint provided for the example script requires eight processes and eight GPUs.
On Baskerville this means we’ll need to use two nodes, each with four GPUs.

This complicates matters when using `srun` because we’ll need to manually run `tourch.distributed.run` on each node.
We can do this, it just means we need to log in to two nodes simultaneously.
Using `screen` or `tmux` will make your life a lot easier when doing this.
If you’re not comfortable with either of these, as an alternative you can open two terminals and log in to Baskerville twice.
You’ll end up with a similar experience.

### 9. Provision two nodes:

Here you’ll again need to replace `<PROJECT_ID>` with the name of your project.

```shell
srun --qos turing \
    --account <PROJECT_ID> \
    --time 0:30:0 \
    --nodes 2 \
    --gpus 8 \
    --mem 131072 \
    --pty bash
```

In the above command we’re provisioning two nodes, eight GPUs and 32 GiB of RAM.
We’ll need all of this computing power to run the model (the checkpoint requires eight processes).
The `srun` command above not only provisions the node, it also logs you in to one of them.

### 10. Find the names of the nodes.

```shell
$ squeue --me --format="%N" --noheader
```

This should output the names of two nodes separated by a comma.
You’ll need to pick one of them to be the primary node (often referred to as the Master Node) and the other to be the secondary node.
It doesn’t matter which is which.

In future steps we’ll use `<PRIMARY_NODE>` to refer to one node and `<SECONDARY_NODE>` to refer to the other node.

For example, I get the following output:

```shell
bask-pg0308u05a,bask-pg0308u06a
```

So I’d replace any instance of `<PRIMARY_NODE>` in the commands below with `bask-pg0308u05a` and any instance of `<SECONDARY_NODE>` with `bask-pg0308u06a`.

### 11. Configure the primary node.

```shell
$ cd llama
$ module purge
$ module load baskerville
$ module load bask-apps/live
$ module load PyTorch/2.1.2-foss-2022b-CUDA-11.8.0
$ . ./venv/bin/activate
```

Note that we’re not creating or configuring the virtual environment here because we assume you already did that in step 2.
If you’ve deleted the virtual environment, you can create it again using the instructions provided there.

Since we now have two nodes to worry about, we’ll need to configure the second node as well.

### 12. Log in to the secondary node.

If you’re using gnu screen or tmux you can switch to a new shell.
If you’ve opened two terminals, move to the other terminal.
Either way, make sure you leave the shell on the primary node running.

```shell
$ ssh <SECONDARY_NODE>
```

### 13. Configure the secondary node.

This is the same as for the primary node, but we have to do it on both.

```shell
$ cd llama
$ module purge
$ module load baskerville
$ module load bask-apps/live
$ module load PyTorch/2.1.2-foss-2022b-CUDA-11.8.0
$ . ./venv/bin/activate
```

### 14. Execute the example script on the primary node:

Move back to the shell on the primary node.
You can now start the example script.
Don’t forget to replace `<PRIMARY_NODE>` in the command below with the name of the node.

```shell
$ python -m torch.distributed.run \
    --nnodes 2 \
    --nproc_per_node 4 \
    --node_rank 0 \
    --master_addr <PRIMARY_NODE> \
    --master_port 6224 \
    example_chat_completion.py \
    --ckpt_dir llama-2-70b-chat/ \
    --tokenizer_path tokenizer.model \
    --max_seq_len 512 \
    --max_batch_size 6
```

This command runs the script using `torchrun`, telling it to use two nodes and four processes per node (a total of eight processes, which is what we require).
Since this is only running on a single node it will only start up four of the eight processes.
The script will wait until the other four processes have started (which we’ll do in the next step) before continuing.

### 15. Execute the example script on the secondary node:

Now move to the terminal on the secondary node and run this similar &mdash; but not *quite* identical &mdash; command.

```shell
$ python -m torch.distributed.run \
    --nnodes 2 \
    --nproc_per_node 4 \
    --node_rank 1 \
    --master_addr <PRIMARY_NODE> \
    --master_port 6224 \
    example_chat_completion.py \
    --ckpt_dir llama-2-70b-chat/ \
    --tokenizer_path tokenizer.model \
    --max_seq_len 512 \
    --max_batch_size 6
```

### 16. Await the results

Eventually the script will generate output using the model which is output to the console.
You should get very similar output on both nodes.

```text
[...]
Loaded in 55.34 seconds
[...]
==================================

System: Always answer with Haiku

User: I am going to Paris, what should I see?

> Assistant:  Eiffel Tower high
Art museums and fashion streets
Romance in the air

==================================
[...]
```

### 17. Tidy up

Make sure you log out of both nodes once you’re done by running the following commands on both nodes.
This will release the resources and make them available for use by others.

```
$ deactivate
$ exit
```


