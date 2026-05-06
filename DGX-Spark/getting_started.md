# Getting started on DGX Spark

## What is a DGX Spark?

The **[NVIDIA DGX Spark](https://www.nvidia.com/en-gb/products/workstations/dgx-spark/?nvid=nv-int-solr-571968)** (announced January 2025) [1,2] is a compact desktop AI supercomputer built on the **GB10 Grace Blackwell Superchip** — the same architecture as NVIDIA's data center GPUs but in a personal form. 

The purpose of DGX Spark is to run large AI models (up to ~200B parameters) locally. It is designed for researchers, developers, and enterprises who need data center-class inference on a desk.

### How does it compare to other hardware?

The closest available comparison for us is the **GH200 Grace Hopper Superchip** used in [Isambard-AI](https://www.isambard.ac.uk/), which is one GPU generation behind (Hopper vs. Blackwell) but has a similar architectural design (CPU + GPU on the same die), and of course is a data center blade rather than a desktop workstation. Below is a comparison of the key specs between the two:

**Table 1.** Hardware comparison between the NVIDIA DGX Spark (GB10 Grace Blackwell) and the NVIDIA GH200 Grace Hopper as deployed on Isambard-AI.

| Spec | DGX Spark (GB10) [2] | GH200 (Isambard-AI) |
|------|-----------------|---------------------|
| GPU Architecture | Blackwell (2025) | Hopper (2023) |
| Streaming Multiprocessors (SMs) | 84 | 132 |
| CUDA Cores | 10,752 | 16,896 |
| CPU Architecture | 20-core ARM | 72-core ARM (Grace / Neoverse V2) |
| FP4 (Tensor) | 1 PFLOPS | N/A |
| FP8 (Tensor) | Not reported | ~2 PFLOPS |
| FP16 (Tensor) | Not reported | ~1 PFLOPS |
| TF32 (Tensor) | Not reported | ~0.5 PFLOPS |
| FP32 (CUDA) | Not reported | 67 TFLOPS |
| GPU Memory | 128 GB unified (LPDDR5X) | 96 GB HBM3e |
| CPU Memory | unified (shared 128 GB) | 480 GB LPDDR5X |
| Memory Bandwidth | ~273 GB/s | ~4 TB/s (HBM3e) |
| CPU–GPU Interconnect | NVLink-C2C (900 GB/s) | NVLink-C2C (900 GB/s) |
| Form Factor | Desktop | Data centre blade |

## Getting access

The Turing owns a small number of DGX Sparks for researchers. The Sparks are owned by specific grand challenges so you should contact your grand challenge lead to find out if your team has access to one. If your team does have access, you will need to carry out the following steps to get access:

1. **Join the `#spark-users` slack channel** on Turing slack. This should have some useful information
2. **Find out who your Spark administrator is.** This will be the person responsible for managing the DGX Spark and granting access to users. You can find out who your Spark administrator is by contacting your grand challenge lead.
3. **Contact your Spark administrator to request a user account.**
4. **Contact IT to grant you access to the DGX Spark's IP address.** The DGX Spark is on a private network accessible via the Turing VPN. Even on the VPN, access to the Spark's IP address is blocked by default so IT need to explicitly allowlist your account before you can connect.

Then, once the above are complete:

5. **Connect to the Turing VPN.** The DGX Spark is on a private network accessible via the Turing VPN, so you will need to connect to the VPN before you can access the Spark.
6. **SSH into the DGX Spark.** You should be able to SSH into the Spark using the username, password and IP address provided by your Spark administrator.
7. **Reset your password.** After logging in for the first time, you should reset your password using the `passwd` command.

## Using the Spark

We currently have no user workload management system set up on the DGX Sparks, so you should coordinate with your team to ensure that only one person is using the Spark's GPU at any time. 

For now, we are using [nvitop](https://nvitop.readthedocs.io/en/latest/) to help with this. It is a CLI tool that shows you the current GPU usage on the Spark, what processes are running and which user launched those processes. Use this before running any workloads to check if the GPU is currently in use by someone else. If it is, please coordinate with them and the rest of your team re. when it will be free to use.

We are planning to set up a more robust user workload management system in the future.

## Installing Python packages

The DGX Spark has CUDA 13.0 and CUDA [compute capability 12.1](https://developer.nvidia.com/cuda/gpus) (sm_121). Like Isambard-AI's GH200 GPUs, it also has `aarch64` architecture. Since this is a newer and less common architecture, some Python packages may not have pre-built wheels available on PyPI and so you may need to build these from source.

### UV

You can install uv on the spark following the instructions [here](https://docs.astral.sh/uv/).

### Torch

To install torch/torchvision/torchaudio:

```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
uv run python "import torch; print(torch.cuda.is_available())"
```

### Flash Attention

[Flash Attention](https://github.com/Dao-AILab/flash-attention) is a common package that does not provide `aarch64` wheels. To install it, first install torch as above, then run the following commands to build flash attention from source:

```bash
export TORCH_CUDA_ARCH_LIST="12.0"
export FLASH_ATTN_CUDA_ARCHS="120"
export MAX_JOBS=16

uv pip install --no-build-isolation -v flash-attn
```

**Note:** This may take a while to build! 

## vLLM

In general, the NVIDIA docs recommend to use their vLLM container for running vLLM on the DGX Spark. Their instructions are [here](https://build.nvidia.com/spark/vllm/instructions).

If however, you'd prefer to install without using a container, you should first install torch as above, then you can install vLLM using the following command:

```bash
# update the version number as needed
uv pip install -U vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/0.17.0/vllm
```

## Containers - Docker vs Podman

The NVIDIA docs recommend using Docker for running containers, however, since this requires root access, we have set up [Podman](https://podman.io/docs) as an alternative. 

In general, the commands are the same between the two, so just replace `docker` with `podman` in the NVIDIA instructions if you want to use Podman instead.

## Getting support

If you have any issues using the DGX Sparks, message the `#spark-users` channel on Turing slack and someone will try to help you out.

Otherwise, NVIDIA have fairly comprehensive documentation for the DGX Spark found [here](https://build.nvidia.com/spark). This is a good place to start if you get stuck. 

**References:**

[1] [NVIDIA DGX Spark announcement](https://nvidianews.nvidia.com/news/nvidia-dgx-spark-personal-ai-supercomputer)
[2] [NVIDIA DGX Spark](https://www.nvidia.com/en-gb/products/workstations/dgx-spark/)
[3] [GH200 Grace Hopper Superchip](https://www.nvidia.com/en-gb/data-center/grace-hopper-superchip/)
[4] [Isambard-AI announcement](https://www.isambard.ac.uk/news/isambard-ai-launches-worlds-first-hpc-ai-supercomputer)
[5] [Isambard-AI](https://www.isambard.ac.uk/)