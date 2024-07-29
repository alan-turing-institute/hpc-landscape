# Llama 3 inference on Baskerville using `srun`

Before you can make use of them you’ll need to download the models.
See [llama-download.md](llama-download.md) for how to go about doing this on Baskerville.

# Llama 3 8B inference, single node, single GPU

The 8B parameter Llama 3 model will run with a single process on a single node with a single GPU.
The checkpoint provided for use with the example will, in fact, only work with a single process.

We found at least 32 GiB of RAM was needed to run the model.
With too little RAM the process will be sent a SIGKILL termination signal, the process will end with an exit code of -9 and an OOM report will be generated.

The RAM, node, GPU and process quantities must be explicitly configured using `srun` for things to work.

## 1. Log in to a single node with a single GPU and a suitable configuration:

You’ll need to replace `<PROJECT_ID>` with the name of your project.

```shell
$ srun --qos turing \
    --account <PROJECT_ID> \
    --time 0:30:0 \
    --nodes 1 \
    --gpus 1 \
    --mem 32768 \
    --pty bash
```

## 2. Set up the environment on the node:

```shell
$ cd llama3
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

## 3. Run the example script that performs a bunch of inference steps:

```shell
$ python -m torch.distributed.run \
    --standalone \
    --nnodes 1 \
    --nproc_per_node 1 \
    example_chat_completion.py \
    --ckpt_dir Meta-Llama-3-8B-Instruct/ \
    --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model \
    --max_seq_len 512 \
    --max_batch_size 6
```

You should see some output that looks something like this:

```text
[...]
Loaded in 10.39 seconds
[...]
==================================

System: Always answer with Haiku

User: I am going to Paris, what should I see?

> Assistant: Eiffel's iron kiss
River Seine's gentle whispers
Art in every stone

==================================
```

One things to note is that in the Llama 3 docs it [recommends](https://github.com/meta-llama/llama3?tab=readme-ov-file#quick-start) to use `torchrun` to execute the script.
When `torchrun` is used directly in this way it doesn’t support Python virtual environments because it defaults to the system-installed Python.
To circumvent this we’ve switched `torchrun` out for `python -m torch.distributed.run` instead.

## 4. Tidy up

Once you’re done don’t forget to log out of the node so that it can be released for access by someone else.
If you forget to do this the node will automatically close your session once the full 30 minutes you requested are up.

```
$ deactivate
$ exit
```

# Llama 3 70B inference, two nodes, eight GPUs

The 70B parameter LLama 3 model requires considerably more resources to execute compared to the 8B parameter model.
The checkpoint provided for the example script requires eight processes and eight GPUs.
On Baskerville this means we’ll need to use two nodes, each with four GPUs.

This complicates matters when using `srun` because we’ll need to manually run `tourch.distributed.run` on each node.
We can do this, it just means we need to log in to two nodes simultaneously.
Using `screen` or `tmux` will make your life a lot easier when doing this.
If you’re not comfortable with either of these, as an alternative you can open two terminals and log in to Baskerville twice.
You’ll end up with a similar experience.

## 5. Provision two nodes:

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

In the above command we’re provisioning two nodes, eight GPUs and 128 GiB of RAM.
We’ll need all of this computing power to run the model (the checkpoint requires eight processes).
The `srun` command above not only provisions the node, it also logs you in to one of them.

## 6. Find the names of the nodes.

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

## 7. Configure the primary node.

```shell
$ cd llama3
$ module purge
$ module load baskerville
$ module load bask-apps/live
$ module load PyTorch/2.1.2-foss-2022b-CUDA-11.8.0
$ . ./venv/bin/activate
```

Note that we’re not creating or configuring the virtual environment here because we assume you already did that in step 2.
If you’ve deleted the virtual environment, you can create it again using the instructions provided there.

Since we now have two nodes to worry about, we’ll need to configure the second node as well.

## 8. Log in to the secondary node.

If you’re using gnu screen or tmux you can switch to a new shell.
If you’ve opened two terminals, move to the other terminal.
Either way, make sure you leave the shell on the primary node running.

```shell
$ ssh <SECONDARY_NODE>
```

## 9. Configure the secondary node.

This is the same as for the primary node, but we have to do it on both.

```shell
$ cd llama3
$ module purge
$ module load baskerville
$ module load bask-apps/live
$ module load PyTorch/2.1.2-foss-2022b-CUDA-11.8.0
$ . ./venv/bin/activate
```

## 10. Execute the example script on the primary node:

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
    --ckpt_dir Meta-Llama-3-70B-Instruct/ \
    --tokenizer_path Meta-Llama-3-70B-Instruct/tokenizer.model \
    --max_seq_len 512 \
    --max_batch_size 6
```

This command runs the script using `torchrun`, telling it to use two nodes and four processes per node (a total of eight processes, which is what we require).
Since this is only running on a single node it will only start up four of the eight processes.
The script will wait until the other four processes have started (which we’ll do in the next step) before continuing.

## 11. Execute the example script on the secondary node:

Now move to the terminal on the secondary node and run this similar &mdash; but not *quite* identical &mdash; command.

```shell
$ python -m torch.distributed.run \
    --nnodes 2 \
    --nproc_per_node 4 \
    --node_rank 1 \
    --master_addr <PRIMARY_NODE> \
    --master_port=6224 \
    example_chat_completion.py \
    --ckpt_dir Meta-Llama-3-70B-Instruct/ \
    --tokenizer_path Meta-Llama-3-70B-Instruct/tokenizer.model \
    --max_seq_len 512 \
    --max_batch_size 6
```

## 12. Await the results

Eventually the script will generate output using the model which is output to the console.
You should get very similar output on both nodes.

```text
[...]
Loaded in 30.96 seconds
[...]
==================================

System: Always answer with Haiku

User: I am going to Paris, what should I see?

> Assistant: Eiffel's iron lace
River Seine's gentle whisper
Montmartre's charm waits

==================================
```

## 13. Tidy up

Make sure you log out of both nodes once you’re done by running the following commands on both nodes.
This will release the resources and make them available for use by others.

```
$ deactivate
$ exit
```

