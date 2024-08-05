# LLaMA-Factory on DAWN using a single XPU and srun

According to the [README on GitHub](https://github.com/hiyouga/LLaMA-Factory/), LLaMA-Factory is easy way to fine-tune large language models. "Easy and Efficient LLM Fine-Tuning" it says. When running the examples on a CUDA-supported device this turns out to be true. LLaMA-Factory supports multiple backends, so in theory it should also be possible to get it running well using the Intel XPUs on DAWN. In this walkthrough we'll do exactly that, working through the steps using an interactive shell spawned using `srun`.

In this example we'll perform supervised LoRA (Low-Rank Adaptation of LLMs) fine-tuning of Meta's Llama3 large-language model.

We cover executing the steps below using `sbatch` in the [`llamafactory-finetuning-sbatch-single.md`](llamafactory-finetuning-sbatch-single.md) and [`llamafactory-finetuning-sbatch-multi.md`](llamafactory-finetuning-sbatch-multi.md) walkthroughs. Since those walkthroughs build on this one, I'd recommend working through this before moving on to these other two.

## Setup

To kick things off we need the LLaMA-Factory code, which we can download from GitHub. So we should log in to DAWN and run the following commands in a suitable directory. There's no need to allocate resources for this, we can run this on a login node.

```sh
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
```

## Edit the configuration file

The training we're going to perform is described in the `examples/train_lora/llama3_lora_sft.yaml` configuration file. We can't perform the training on DAWN using the example training configuration in its default form, so we have to make a small adjustment to it.

On line 32 there's a configuration option that sets whether or not `bf16` should be used for the training. We're going to switch that from `true` to `false`. Here's the diff of the file to highlight exactly what needs to be changed:

```diff
diff --git a/examples/train_lora/llama3_lora_sft.yaml b/examples/train_lora/llama3_lora_sft.yaml
index 55a8077..fed80cb 100644
--- a/examples/train_lora/llama3_lora_sft.yaml
+++ b/examples/train_lora/llama3_lora_sft.yaml
@@ -29,7 +29,7 @@ learning_rate: 1.0e-4
 num_train_epochs: 3.0
 lr_scheduler_type: cosine
 warmup_ratio: 0.1
-bf16: true
+bf16: false
 ddp_timeout: 180000000
 
 ### eval
```

If we leave this value as `true` the Intel Extensions for PyTorch (IPEX) code will attempt to optimise the model to use mixed-precision BFLOAT16 calculations. This optimisation will fail. If we disable the `bf16` option then the model will be used in its original format and the optimisation step succceeds.

On the login node you should either apply the above patch or edit the file directly to make the change.

## Start an interactive session

Now we're ready to start performing compute-node specific tasks, so we'll need to set up an interactive shell on a compute-node to work in. In the following command you'll need to adjust `<ACCOUNT_ID>` to match your account. We've given an hour of time to work on this, but you might prefer to adjust this if you think you'll need more or less. Bear in mind that the actual training step will take around 35 minutes to complete.

One further thing to bear in mind is that on the very first run the training script will attempt to download the model data from Hugging Face. This can take *several hours* and you should increase the time to accommodate for this. I'd recommend requesting 5 hours on your first run.

```sh
srun --account <ACCOUNT_ID> --partition pvc --time 1:00:0 --nodes 1 --gres gpu:1 --pty bash
```

## Set up the environment

Next up we need to configure the packages available in our environment by loading various modules and running various scripts available on DAWN. Here we load in various Intel OneAPI modules to provide a SYCL compiler and libraries.

```sh
module purge
module load default-dawn
module load intel-oneapi-tbb/2021.11.0/oneapi/xtkj6nyp
module load intel-oneapi-compilers/2024.0.0/gcc/znjudqsi
module load intel-oneapi-mkl/2024.0.0/oneapi/4n7ruz44
module load intel-oneapi-mpi/2021.11.0/oneapi/h7nq7sah
module load gcc/13.2.0/ic25lr2r

pushd /usr/local/dawn/software/spack/spack-views/dawn-test-2023-12-22/
source intel-oneapi-compilers-2024.0.0/gcc-13.2.0/znjudqsiaf6x5u2rxdtymf6ss55nmimw/compiler/2024.0/env/vars.sh
source intel-oneapi-mkl-2024.0.0/oneapi-2024.0.0/4n7ruz44nhbsd5xp4nnz6mgm2z7vqzxs/mkl/2024.0/env/vars.sh
source intel-oneapi-compilers-2024.0.0/gcc-13.2.0/znjudqsiaf6x5u2rxdtymf6ss55nmimw/setvars.sh
popd
```

## Create a virtual environment

The modules won't give us everything we need, so we should also create a virtual environment and install the remaining dependencies into it. We'll set up a virtual environment in a directory called `venv` inside the `LLaMA-Factory` directory. If you already have an environment set up there, you'll either need to delete it or change the name of the one we're using.

If you're already created and set up this environment previously then you can just activate it on future occasions.

```sh
python3.9 -m venv venv
. ./venv/bin/activate

pip install --upgrade pip
pip install "numpy<2"
pip install torch==2.0.1a0 torchvision==0.15.2a0 intel-extension-for-pytorch==2.0.120+xpu oneccl-bind-pt==2.0.200 --extra-index-url https://pytorch-extension.intel.com/release-whl-aitools/
source $(python -c "import oneccl_bindings_for_pytorch as torch_ccl;print(torch_ccl.cwd)")/env/vars.sh
```

We;ve biw reached the point where we can start to follow the instructions from the [LLaMA-Factory source repo](https://github.com/hiyouga/LLaMA-Factory/tree/main?tab=readme-ov-file). That means we should also install the dependencies described there. Note that we'll already have installed most of what's needed by this point already, but this will mop up the remaining pices.

```sh
pip install -e ".[torch,metrics]"
```

## Download the model from Hugging Face

When you run the fine-tuning training script for the first time it will attempt to download the model data from Hugging Face. In order for this to succeed you'll have had to:

1. Arrange access to the model.
2. Configure access to your Hugging Face account.

For the first of these steps you'll need to visit the [Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) repository on Hugging Face and request access to the model. I found it took around an hour for me request to be granted. In the meantime you can continue on to the second step.

For the second step you'll need to create an access token via the [Tokens page](https://huggingface.co/settings/tokens) on Hugging Face. Create a Read Only token and take a copy of it, then run the following:

```sh
pip install --upgrade huggingface_hub
huggingface-cli login
```

When running the second of these you'll be requested to enter the token at the command line. When it asks you whether to add the token as a git credential you can answer `n` for No.

Once these steps are complete you'll have set up access to download the model from Hugging Face to your DAWN account.

## Perform fine-tuning

Everything is now set up as required, we can now run the fine-tuning step. The LLaMA-Factory code is doing all the hard work here, calling the Hugging Face `accelerate` code in the background to perform the training. We use the `llamafactory-cli` wrapper for this as it makes things simpler. This will run the training on only a single tile of a single XPU (in effect one eighth of the total XPUs available on the node). If we wanted to run with more than one tile we'd need to call `accelerate` directly and we cover this in the [`llamafactory-finetunning-sbatch-multi.md`](llamafactory-finetunning-multi.md) walkthrough.

The training takes around 30 minutes to complete, but there are some preliminary steps and an inference step at the end, so the total task will run for around 35 minutes.

```sh
llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml
```

If everything has gone well you should see the fine-tuning training complete, followed by the short inference step.

## Tidy up

When you're done, deactivate the virtual environment and drop out of the interactive session

```sh
deactivate
exit
```

And that's it! Congratulations, you've now successfully fine-tuned Llama3-8b on DAWN.
