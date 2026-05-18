# How does DGX Spark (GB10) compare vs Isambard-AI (GH200) and Baskerville (A100)?

## vLLM inference with [Qwen3.6 35B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)

| System | Scenario | Requests (60s) | Req Lat Mdn (s) | TTFT Mdn (ms) | TPOT Mdn (ms) | Output tok/s | Total tok/s |
|--------|----------|----------------|-----------------|----------------|----------------|--------------|-------------|
| GB10 (DGX Spark) | 512p + 256o | 7 | 8.6 | 0.0 | 33.8 | 34.5 | 105.0 |
| GH200 (Isambard-AI) | 512p + 256o | 43 | 1.4 | 0.0 | 5.3 | 191.8 | 582.9 |
| GB10 (DGX Spark) | 2048p + 512o | 4 | 17.6 | 0.0 | 34.3 | 38.9 | 195.3 |
| GH200 (Isambard-AI) | 2048p + 512o | 22 | 2.7 | 0.0 | 5.3 | 197.1 | 989.2 |
| GB10 (DGX Spark) | 128p + 1024o | 2 | 33.8 | 13282.8 | 33.0 | 51.9 | 58.9 |
| GH200 (Isambard-AI) | 128p + 1024o | 12 | 5.1 | 2593.7 | 5.0 | 218.1 | 247.5 |

i.e. DGX Spark handles 5-6x fewer requests and generates tokens around 6x slower across all scenarios.

For detailed information on the inference benchmarking setup and results see `vllm_inference_benchmark_results/results.md`.

## Finetuning [Gemma 3 4b](https://huggingface.co/google/gemma-3-4b-it)

| System                         | GPUs  | steps | epochs | times (s) |  it/s | sm% (mean) | sm% (max) | mem% (mean) | mem% (max) |
|--------------------------------|-------|-------|--------|-----------|-------|------------|-----------|-------------|------------|
| GB10 (DGX spark) |     1 |  1798 |      1 |     15144 |  0.12 |      94.30 |     96.00 |         N/A |        N/A |
| GH200 (Isambard-AI) |     1 |  1798 |      1 |      1961 |  0.92 |      80.33 |    100.00 |       46.16 |      76.00 |
| GH200 (Isambard-AI) |     2 |  1798 |      1 |      1136 |  1.59 |      86.97 |    100.00 |       40.90 |      80.00 |
| GH200 (Isambard-AI) |     4 |  1800 |      1 |       516 |  3.48 |      85.66 |    100.00 |       39.00 |      76.00 |

i.e. DGX Spark is around 7x slower vs using 1 GPU on Isambard-AI or 29x slower if using 4 GPUs on Isambard-AI.

## Finetuning [Gemma 3 270M](https://huggingface.co/google/gemma-3-270m-it)

| System                         | GPUs  | steps | epochs | times (s) |  it/s | sm% (mean) | sm% (max) | mem% (mean) | mem% (max) |
|--------------------------------|-------|-------|--------|-----------|-------|------------|-----------|-------------|------------|
| GB10 (DGX spark) |     1 |  1798 |      1 |     10553 |  0.17 |      87.04 |     96.00 |         N/A |        N/A |
| GH200 (Isambard-AI) |     1 |  1798 |      1 |      1490 |  1.21 |      69.82 |    100.00 |       46.34 |      74.00 |

i.e. DGX Spark is around 7x slower vs using 1 GPU on Isambard-AI.

## Finetuning Anemoi (FASTNET) 

| System | GPUs | steps | epochs | times (s) |  it/s | sm% (mean) | sm% (max) | mem% (mean) | mem% (max) |
|--------|------|-------|--------|-----------|-------|------------|-----------|-------------|------------|
| GB10 (DGX spark) |   1  |    128 |      1 |      1104 |  0.12 |      90.05 |     96.00 |        0.00 |       0.00 |
| A100 (Baskerville) | 1  |   128 |      1 |       234 |  0.55 |      62.02 |    100.00 |       29.45 |      62.00 |
| A100 (Baskerville) | 2  |   128 |      1 |       142 |  0.90 |      28.70 |    100.00 |       11.07 |      53.00 |
| A100 (Baskerville) | 4 |   128 |      1 |       153 |  0.83 |      51.00 |    100.00 |        9.02 |      40.00 |

i.e. DGX Spark is around 5x slower vs using 1 GPU on Baskerville or around 7-8x slower vs using 2-4 GPUs on Baskerville.
