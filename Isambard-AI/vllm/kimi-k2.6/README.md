# Serving Kimi K2.6 on Isambard-AI

Two scripts to deploy [Kimi K2.6](https://huggingface.co/moonshotai/Kimi-K2.6) as an OpenAI-compatible API server on Isambard-AI, using vLLM.

| | |
|---|---|
| **Hardware** | 2 nodes × 4 GH200 GPUs |
| **Parallelism** | TP=4 (within node) × PP=2 (across nodes) |
| **Context length** | 16,384 tokens |
| **Known limitation** | CUDA graphs disabled (`--enforce-eager`) |

## Requirements

| | |
|---|---|
| **uv** | Must be installed already |
| **`<<<PROJECT_STORAGE_PATH>>>`** | Replace in both scripts — a directory on project Lustre storage, not `$HOME` |
| **`HF_TOKEN`** | `export HF_TOKEN=hf_xxx` before running `setup.sh` — Isambard shares outbound IPs, so unauthenticated downloads get rate-limited |

## `setup.sh` — one-time environment setup

| Step | Detail |
|---|---|
| Python env | uv venv + vLLM/Ray, from vLLM's nightly wheel index (not on PyPI yet) |
| NVIDIA HPC SDK | CUDA forward-compatibility shim — driver is CUDA 12.x, this vLLM build targets CUDA 13 |
| Weights | ~650GB download from Hugging Face Hub |

Safe to re-run — each step skips if already done.

## `serve.sh` — starts the server

| Step | Detail |
|---|---|
| Ray cluster | Spans both nodes via Slurm `srun` — head + worker |
| Parallelism | TP=4 within node (fast NVLink all-reduce); PP=2 across nodes (only passes activations at layer boundaries, tolerates the slower cross-node link) |
| `--enforce-eager` | Disables CUDA graph capture — works around a Ray + pipeline-parallelism illegal-memory-access bug ([ray#51596](https://github.com/ray-project/ray/issues/51596)); costs some latency, not correctness |
| CUDA env | Forward-compat variables re-set in every `srun` step, since each starts with a clean environment |

**Stack:** Slurm, uv, vLLM, Ray, NVIDIA HPC SDK, Hugging Face Hub.