# Kimi-K3 on Isambard-AI

Runs `vllm/vllm-openai:kimi-k3` on 8 nodes (32x GH200 GPUs) using apptainer.

## How to run

1. Build the container. This must be done on a compute node, not the login node. The easiest way is to ask for an interactive session:

```bash
srun -N 1 --gpus 1 --mem=0 --time=01:00:00 --pty bash
```

Then, on the compute node in the project directory, run:

```bash
mkdir -p container
apptainer build container/kimi-k3.sif docker://vllm/vllm-openai:kimi-k3
```

2. Set `HF_HOME` (if not set already in `.bashrc`)

3. Run `sbatch batch_kimi_k3.sh`

4. Check it worked - there should be `kimi_k3_<jobid>.log` (slurm logs) and `logs/kimi_k3/nccl-*.log` (nccl logs)

## Notes

- `VLLM_ENABLE_CUDA_COMPATIBILITY=1` is required since the image is built for CUDA 13 (Isambard's driver only supports up to CUDA 12.7).
- The original kimi-k3 container with hash `sha256:e90e2603b2781936651ba019804137714367c69e10a7b25a2e57b46995225616` has a bug that means you get OOM error. Need to apply a patch to fix this if using this container version.
