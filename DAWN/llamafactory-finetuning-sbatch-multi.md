# LLaMA-Factory on DAWN using four XPUs and sbatch

According to the [README on GitHub](https://github.com/hiyouga/LLaMA-Factory/), LLaMA-Factory is easy way to fine-tune large language models. "Easy and Efficient LLM Fine-Tuning" it says. When running the examples on a CUDA-supported device this turns out to be true. LLaMA-Factory supports multiple backends, so in theory it should also be possible to get it running well using the Intel XPUs on DAWN. In this walkthrough we'll do exactly that running it as a batch job.

In this example we'll perform supervised LoRA (Low-Rank Adaptation of LLMs) fine-tuning of Meta's Llama3 large-language model.

We cover executing the same task using an interactive shell and `srun` in the [`llamafactory-finetuning-srun-single.md`](llamafactory-finetuning-sbatch-single.md) walkthrough. The steps here assume you've already worked through the interactive version, so if you've not already read it I recommend working through that first.

In this walkthrough we'll run the training using all four of the XPUs available on a single node. This is a very fast and efficient way to perform fine-tuning and will run between five and six times faster than running on just a single tile, making full use of a node, so it's a good use of time and resources.

We cover running the fine-tuning on just a single tile in the [`llamafactory-finetuning-sbatch-multi.md`](llamafactory-finetuning-sbatch-multi.md) walkthrough. Executing on a single tile is slightly simpler than running across multiple XPUs, but the walkthroughs are otherwise quite similar.

## Setup

We're going to assume you've already completed the interactive walkthrough. This will mean you've already completed the following steps:

1. Cloned the repository.
2. Edited the `llama3_lora_sft.yaml` file to support the Intel IPEX environment.
3. Configured a token for access to Hugging Face on DAWN.

The other steps, including setting up the environment, creating the virtual environment and installing dependencies will all be performed automatically by our batch script, which will use an existing virtual environment if you already set one up.

The code will also automatically download the model from Hugging Face if it's not already available locally. However note that this takes a *long* time, so if you're planning to run this without first having worked through the `srun` version, you'll need to increase the length of time your batch script requests to account for this. I'd recommend changing it from the one hour it's currently set up for to at least five hours.

## Investigate the batch file

The relevant batch script we'll be using is `batch-llamafactory-multi.sh`. Before running anything it'll be helpful to understand what's going on inside this script, so open it inside your favourite text editor to take a look.

There are essentially six sections to this batch file:

1. Lines 1-10 are the SLURM header. These specify the environment you're requesting from SLURM: how many nodes, how many GPUs, how long for and so on.
2. Lines 21-36 configure the environment. These load and configure the relevant modules that are already available on DAWN. These make available the SYCL compiler and Intel Extensions for PyTorch.
3. Lines 38-66 set environment variables. When using `accelerate` with multiple XPUs we have to set up various environment variables. These are specific to DAWN, but necessary to avoid the script failing.
4. Lines 68-88 create the virtual environment. These will use an existing virtual environment if there is one. If not it creates the environment and installs all of the required dependencies.
5. Lines 90-95 perform the fine-tuning. This is the bit that actually does the work we're interested in.
6. Lines 97-99 tidy things up. Once the task is complete we deactivate the virtual environment and restore our original directory. None of this is necessary, but it helps provide closure.

## Transfer the batch file

You'll need to run the script directly from DAWN so will need to copy it over to the system. We recommend using `scp` for this, but you can use whatever your preferred mechanism is. You can run this on your local machine to transfer the file over.

```sh
scp batch-llamafactory-multi.sh <USERNAME>@login-dawn-dev.hpc.cam.ac.uk:<DIRECTORY_TO_STORE_FILE>
```

In addition to the batch file, for multi-XPU training you'll also need to use a bespoke configuration for `accelerate`. We've provided the `config_node01xpu04.yaml` file for this, so you'll need to transfer that over as well.

```sh
scp config_node01xpu04.yaml <USERNAME>@login-dawn-dev.hpc.cam.ac.uk:<DIRECTORY_TO_STORE_FILE>
```

You should copy the `config_node01xpu04.yaml` file to the same directory that you cloned the `LLaMA-Factory` repository into (i.e. not *inside* the `LLaMA-Factory` directory but at the same level as it).

## Edit the batch file

There are two account-specific parts of the batch file you'll need to edit before you can use it.

1. `<ACCOUNT_ID>`: replace this with your account information.
2. `<FULL_PATH>`: replace this with the full path of the directory where you cloned the `LLaMA-Factory` repo to.

If you've stored the `config_node01xpu04.yaml` file to somewhere other than the directory containing the `LLaMA-Factory` clone, you may also need to amend the path pointing to it on line 95.

To find the full path of a particular directory, move into it using `cd` and run `pwd`. The full path will be output to the console.

## Perform fine-tuning

Everything is now set up as required, we can now run the fine-tuning step. Move to the directory where you have the batch file stored, then run the following:

```sh
sbatch batch-llamafactory-multi.sh
```

As we're running this as a batch job it'll be queued before being executed. You can check its status by running the following:

```sh
squeue --me
```

For multi-XPU training we can no longer use `llamafactory-cli` because we need to pass our bespoke configuration to the `accelerate` backend. Instead we call `accelerate` passing in details of the configuration file, training code and the Hugging Face configuration.

The training will happen considerably faster running across four XPUs compared to running on a single XPU tile. It just over five minutes with four XPUs compared to over 30 minutes when running on just a single tile of one XPU. There is also some set up time required and an inference step at the end, so the overall time for the run should be between five and ten minutes.

## View the output

All of the output sent to `stdout` during the training will be saved out to a file called `job-multi.out`. You can load this into your favourite text editor to see the results. If everything completes successfully you should see the fine-tuning training complete, followed by the short inference step.

While the execution is running you can, if you want, follow its progress in real-time by running the following:

```sh
tail -f job-multi.out
```

And that's it! Congratulations, you can now fine tune models in batches on DAWN using up to eight XPUs simultaneously.

