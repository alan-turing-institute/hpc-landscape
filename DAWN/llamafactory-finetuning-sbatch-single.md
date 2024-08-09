# LLaMA-Factory on DAWN using a single XPU and sbatch

According to the [README on GitHub](https://github.com/hiyouga/LLaMA-Factory/), LLaMA-Factory is easy way to fine-tune large language models. "Easy and Efficient LLM Fine-Tuning" it says. When running the examples on a CUDA-supported device this turns out to be true. LLaMA-Factory supports multiple backends, so in theory it should also be possible to get it running well using the Intel XPUs on DAWN. In this walkthrough we'll do exactly that running it as a batch job.

In this example we'll perform supervised LoRA (Low-Rank Adaptation of LLMs) fine-tuning of Meta's Llama3 large-language model.

We cover executing the same task using an interactive shell and `srun` in the [`llamafactory-finetuning-srun-single.md`](llamafactory-finetuning-sbatch-single.md) walkthrough. The steps here assume you've already worked through the interactive version, so if you've not already read it I recommend working through that first.

In this walkthrough we'll run the training on just a single tile of a single XPU. It's also possible to execute it on more, up to eight tiles running across four XPUs on a single DAWN node. We cover this case, which is a little more complex in the [`llamafactory-finetuning-sbatch-multi.md`](llamafactory-finetuning-sbatch-multi.md) walkthrough.

## Setup

We're going to assume you've already completed the interactive walkthrough. This will mean you've already completed the following steps:

1. Cloned the repository.
2. Edited the `llama3_lora_sft.yaml` file to support the Intel IPEX environment.
3. Configured a token for access to Hugging Face on DAWN.

The other steps, including setting up the environment, creating the virtual environment and installing dependencies will all be performed automatically by our batch script, which will use an existing virtual environment if you already set one up.

The code will also automatically download the model from Hugging Face if it's not already available locally. However note that this takes a *long* time, so if you're planning to run this without first having worked through the `srun` version, you'll need to increase the length of time your batch script requests to account for this. I'd recommend changing it from the one hour it's currently set up for to at least five hours.

## Investigate the batch file

The relevant batch script we'll be using is `batch-llamafactory-single.sh`. Before running anything it'll be helpful to understand what's going on inside this script, so open it inside your favourite text editor to take a look.

There are essentially five sections to this batch file:

1. Lines 1-10 are the SLURM header. These specify the environment you're requesting from SLURM: how many nodes, how many GPUs, how long for and so on.
2. Lines 21-36 configure the environment. These load and configure the relevant modules that are already available on DAWN. These make available the SYCL compiler and Intel Extensions for PyTorch.
3. Lines 38-58 create the virtual environment. These will use an existing virtual environment if there is one. If not it creates the environment and installs all of the required dependencies.
4. Lines 60-54 perform the fine-tuning. This is the bit that actually does the work we're interested in.
5. Lines 67-69 tidy things up. Once the task is complete we deactivate the virtual environment and restore our original directory. None of this is necessary, but it helps provide closure.

## Transfer the batch file

You'll need to run the script directly from DAWN so will need to copy it over to the system. We recommend using `scp` for this, but you can use whatever your preferred mechanism is. You can run this on your local machine to transfer the file over.

```sh
scp batch-llamafactory-single.sh <USERNAME>@login-dawn-dev.hpc.cam.ac.uk:<DIRECTORY_TO_STORE_FILE>
```

## Edit the batch file

There are two account-specific parts of the batch file you'll need to edit before you can use it.

1. `<ACCOUNT_ID>`: replace this with your account information.
2. `<FULL_PATH>`: replace this with the full path of the directory where you cloned the `LLaMA-Factory` repo to.

If you move into the directory containing `LLaMA_Factory` you can find the full path by running `pwd` from the command line.

## Perform fine-tuning

Everything is now set up as required, we can now run the fine-tuning step. Move to the directory where you have the batch file stored, then run the following:

```sh
sbatch batch-llamafactory-single.sh
```

As we're running this as a batch job it'll be queued before being executed. You can check its status by running the following:

```sh
squeue --me
```

The call to `llamafactory-cli` in the batch file will calling the Hugging Face `accelerate` code in the background to perform the training. This will run the training on only a single tile of a single XPU (in effect one eighth of the total XPUs available on the node). If we wanted to run with more than one tile we'd need to call `accelerate` directly and we cover this in the [`llamafactory-finetunning-sbatch-multi.md`](llamafactory-finetunning-multi.md) walkthrough.

The training takes around 30 minutes to complete, but there are some preliminary steps and an inference step at the end, so the total task will run for around 35 minutes once it's been allocated resources.

## View the output

All of the output sent to `stdout` during the training will be saved out to a file called `job.out`. You can load this into your favourite text editor to see the results. If everything completes successfully you should see the fine-tuning training complete, followed by the short inference step.

While the execution is running you can, if you want, follow its progress in real-time by running the following:

```sh
tail -f job.out
```

And that's it! Congratulations, you can now fine tune models in batches on DAWN.

