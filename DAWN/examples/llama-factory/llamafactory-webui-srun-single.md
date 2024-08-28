# LLaMA-Factory WebUI on DAWN using a single XPU and srun

LLaMA-Factory provides wide-ranging functionality across a large collection of different pre-trained models. While that's useful in itself, for many users the Web-based user-interface is one of its best features, offering an accessible route to training, fine-tuning and interacting with models.

HPCs aren't typically first-choice platforms for Web-based material, given their focus on time-bounded and scheduled jobs. Nevertheless, that doesn't mean you can't use them to spin up a resource with a Web-based user interface for use during a fixed time-period.

In this walkthrough we'll see how we can do this using the LLaMA-Factory Web UI. We'll create an interactive shell, install and run LLaMA-Factory and then port forward the user interface so that we can get GPU-accelerated inference on DAWN via the browser on our local machine.

For more information about LLaMA-Factory, see its [README on GitHub](https://github.com/hiyouga/LLaMA-Factory/).

This repository contains other LLaMA-Factory walkthroughs for use on DAWN, including how to perform fine-tuning using `sbatch` with a [single XPU](llamafactory-finetuning-sbatch-single.md) or [multiple XPUs](llamafactory-finetuning-sbatch-multi.md) walkthroughs.


## Setup

To kick things off we need the LLaMA-Factory code, which we can download from GitHub. So we should log in to DAWN and run the following commands in a suitable directory. There's no need to allocate resources for this, we can run this on a login node.

```sh
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
```

## Update the inference code with XPU support

We've discussed how to update the configuration to support training and fine-tuning on Intel XPUs [elsewhere](llamafactory-finetuning-sbatch-single.md). The training code makes use of the HuggingFace `accelerate` libraries which have already been built to support Intel XPU accelerators.

The LLaMA-Factory inference code doesn't use `accelerate` and doesn't have XPU support out-of-the-box, so we need to tweak it just slightly to get it to work effectively on DAWN. It'll run without these changes, but will only run on CPU and so will be really quite slow. This small change increases the speed considerably, even though it ends up only using one tile of an XPU (effectively one half of a single GPU).

In the next steps you'll need to edit one of the source files. You'll need to edit the file directly on DAWN, but you don't need to be using a compute node for this, it's fine to use a login node.

The file we need to edit is the `src/llamafactory/model/loader.py` in the LLaMA-Factory folder. Open it using your favourite text editor and find line 184, which should look like this:

```python
        model.eval()
```

Delete this line and replace it with the following three lines of code:

```python
        if is_torch_xpu_available():
            model = model.eval().to("xpu")
            torch.autocast(device_type="xpu", enabled=True, dtype=param.data.dtype)
```

This code will check whether there's an XPU available and, if there is, ensure the model is uploaded to it, rather than leaving it on the CPU. We also need to add in import to the top of the file. On line 29 add the following:

```python
from transformers.utils import is_torch_xpu_available
```

This new import will ensure we have the functionality to check whether or not we're on a system with XPU support. After making these two changes you can save out your file. The following diff summarises the changes made:

```diff
diff --git a/src/llamafactory/model/loader.py b/src/llamafactory/model/loader.py
index fe700d5..5abdba2 100644
--- a/src/llamafactory/model/loader.py
+++ b/src/llamafactory/model/loader.py
@@ -26,6 +26,7 @@ from .model_utils.mod import convert_pretrained_model_to_mod, load_mod_pretraine
 from .model_utils.unsloth import load_unsloth_pretrained_model
 from .model_utils.valuehead import load_valuehead_params
 from .patcher import patch_config, patch_model, patch_tokenizer, patch_valuehead_model
+from transformers.utils import is_torch_xpu_available


 if TYPE_CHECKING:
@@ -181,7 +182,9 @@ def load_model(
             if param.data.dtype == torch.float32 and model_args.compute_dtype != torch.float32:
                 param.data = param.data.to(model_args.compute_dtype)

-        model.eval()
+        if is_torch_xpu_available():
+            model = model.eval().to("xpu")
+            torch.autocast(device_type="xpu", enabled=True, dtype=param.data.dtype)
     else:
         model.train()

```

## Transfer the configuration file

You'll need a DAWN-specific configuration that recognises the single XPU available. We've provided one in the repository, called [`config_node01xpu01.yaml`](./scripts/config_node01xpu01.yaml) but you'll need to transfer it over to DAWN to use it.

We recommend using `scp` for this, but you can use whatever mechanism you prefer. You can run this on your local machine to transfer the file over.

```sh
scp ./scripts/config_node01xpu01.yaml <USERNAME>@login-dawn-dev.hpc.cam.ac.uk:<DIRECTORY_TO_STORE_FILE>
```

You should copy the `config_node01xpu01.yaml` file to the same directory that you cloned the `LLaMA-Factory` repository into (i.e. not *inside* the `LLaMA-Factory` directory but at the same level as it).

## Start an interactive session

Now we're ready to start performing compute-node specific tasks, so we'll need to set up an interactive shell on a compute-node to work on. You'll need to adjust `<ACCOUNT_ID>` in the following to match your account. We've given an hour of time to work on this, but you might prefer to increase this if you plan to work with the Web UI for longer.

One further thing to bear in mind is that on the very first run the training script will attempt to download the model data from Hugging Face. This can take *several hours*. Subsequently it will load much more quickly.

```sh
srun --account TURING-DAWN-GPU --partition pvc --time 1:00:0 --nodes 1 --gres gpu:1 --mem 131072 --pty bash
```

We've requested 128 GiB of RAM. This is essential in order for inference to work with the LLaMA3-8B-Chat model. Otherwise the process will be killed with an OOM (-9) error.

Once logged in the command prompt will show the computer you've been allocated. It's likely to be in the form `pvc-s-NNN` where `NNN` is a number. This is the name of your compute node and you'll need it later, so you should take note, or better yet even take a note, of it.

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

We've reached the point where we can start to follow the instructions from the [LLaMA-Factory source repo](https://github.com/hiyouga/LLaMA-Factory/tree/main?tab=readme-ov-file). That means we should also install the dependencies described there. Note that we'll already have installed most of what's needed by this point already, but this will mop up the remaining pices.

```sh
pip install -e ".[torch,metrics]"
```

## Configure your Hugging Face account

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

## Run the Web UI

Everything is now set up as required, we can now run the Web UI. Notice that we're maing use of the `accelerate` configuration file and the edited example file.

```sh
accelerate launch --config_file ../config_node01xpu01.yaml src/webui.py
```

The process will take a little time to initialise. When ready it will open the ELinks text-based Web browser to a page that shows the local address. It should state something like this:

```
http://localhost:7860/
```

You shouldn't yet interact with this page, but you will need to take note of the port number (in this case 7860) as we'll need to use it in the next step when we set up the port forwarding.

## Configure port forwarding

In order to access the Web UI using the browser on your local machine, you'll need to configure port forwarding. This will forward any data sent to a specific port on your local machine to a (potentially different) port on a remote machine. For DAWN we'll need to set this up quite carefully because the compute nodes aren't directly accessible over the Internet. We'll therefore need to perform some acrobatics, in the form of a proxy jump across the login node to get to the compute node.

In order to do this you'll need three pieces of information:

1. `<USERNAME>`: Your username. This is shown on the command prompt when you log in to DAWN.
2. `<COMPUTE_NODE>`: The name of the compute node you're using. We saw this at the command prompt when we first logged in to the compute note.
3. `<PORT>`: The port the Web UI is exposed on. This was displayed by ELinks in the previous step.

In the command below you'll need to replace these placeholders with their correct values.

Run the following command *on your local machine*.

```sh
ssh -J <USERNAME>@login-icelake.hpc.cam.ac.uk -L 8080:localhost:<PORT> <USERNAME>@<COMPUTE_NODE>
```

You'll probably be asked to enter an MFA token at least once, possibly twice, depending on your configuration. If you're asked twice you'll need to use different tokens each time, so you should enter one token, then wait for the TOTP timeout (i.e. until the token is refreshed) and then use the next token for the second request.

Once through the authentication steps you'll be left with a login shell. You don't need to do anything further with this, just keep it open in a terminal until the end.

## Open the Web UI

You're now ready to open the Web UI. Go to your favourite browser on your local machine. The enter the following address into the address bar and hit enter:

```text
http://localhost:8080/
```

The beautiful LLaMA-Factory Web interface, developed using Gradio, should appear in your browser.

## Chat!

Let's load up the LLaMA3-8B-Chat model and interact with it. This is using model inference (no training or fine-tuning) and can all be done through the Web UI.

In the first box at the top of the page the language should be set to English ("en") and the model name should be set to "LLaMA3-8B-Chat". If not, select these from the drop-down lists.

In the next box it shows the Finetuning info. You can ignore this.

Then there are Advanced configuration options. These can be left at their default values.

Below this you'll see four tabs: "Train", "Evaluate & Predict", "Chat" and "Export".

Select the "Chat" tab.

The items below will change. Make sure you have "huggingface" set as the inference engine and "auto" set as the inference data type. Then select the "Load model" button.

If you now go back to your DANW console you'll see some activity happening there as the model loads.

Once it's loaded, you can now enter text in the "Input..." box and press "Submit" to send it to the model. The model's response will then appear in the "Chatbox" box.

If the backend is correctly using an XPU for the inference, the response should print out at a pretty decent rate (about reading speed).


## Tidy up

When you're done you should go back to the DAWN console and exit the application by pressing Ctrl-C. Then deactivate the virtual environment and drop out of the interactive session:

```sh
deactivate
exit
```

And that's it! Congratulations, I hope you enjoyed your conversation with LLaMA3-8B on DAWN.

