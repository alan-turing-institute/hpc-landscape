# vLLM inference benchmarks (Qwen3.6 35B)

## Running the benchmarks

The Qwen3.6 model was served using vLLM 0.19.2 using the following command:

```bash
vllm serve Qwen/Qwen3.6-35B-A3B \
    --served-model-name qwen3.6-35b \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 262144 \
    --enable-prefix-caching \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3
```

The benchmarks were then run using [guidellm](https://github.com/vllm-project/guidellm#benchmark-controls) bencharking tool:

```bash
TARGET="http://localhost:8000"
MODEL="qwen3.6-35b"
PROCESSOR="Qwen/Qwen3.6-35B-A3B"
RESULTS_DIR="results/qwen36_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$RESULTS_DIR"

echo "=== GuideLLM Benchmark: $MODEL @ $TARGET ==="
echo "Results will be saved to: $RESULTS_DIR"
echo

# Wait for the server to be ready
echo "Checking server health..."
until curl -sf "$TARGET/health" > /dev/null 2>&1; do
    echo "  Waiting for server at $TARGET ..."
    sleep 5
done
echo "  Server is ready."
echo

run_benchmark() {
    local name="$1"
    local -a extra_args=("${@:2}")
    local out="$RESULTS_DIR/$name"
    mkdir -p "$out"
    echo "--- Running: $name ---"
    guidellm benchmark \
        --target "$TARGET" \
        --model "$MODEL" \
        --processor "$PROCESSOR" \
        --output-path "$out" \
        "${extra_args[@]}"
    echo
}

# Synchronous (1 concurrent request) — baseline latency, 512 prompt / 256 output
run_benchmark "synchronous_512_256" \
    --profile synchronous \
    --max-seconds 60 \
    --data "prompt_tokens=512,output_tokens=256"

# Longer context: 2048 prompt tokens
run_benchmark "synchronous_2048_512" \
    --profile synchronous \
    --max-seconds 60 \
    --data "prompt_tokens=2048,output_tokens=512"

# Short prompt / long output (generation heavy)
run_benchmark "synchronous_128_1024" \
    --profile synchronous \
    --max-seconds 60 \
    --data "prompt_tokens=128,output_tokens=1024"

echo "=== All benchmarks complete ==="
echo "Results: $RESULTS_DIR"
```

## GB10 (DGX spark)**

### 512 prompt + 256 output tokens, 1 stream, 1 request per stream, 60s duration

ℹ Run Summary Info
|=============|==========|==========|======|======|======|========|=====|=====|========|=====|=====|
| Benchmark   | Timings                              ||||| Input Tokens     ||| Output Tokens    |||
| Strategy    | Start    | End      | Dur  | Warm | Cool | Comp   | Inc | Err | Comp   | Inc | Err |
|             |          |          | Sec  | Sec  | Sec  | Tot    | Tot | Tot | Tot    | Tot | Tot |
|-------------|----------|----------|------|------|------|--------|-----|-----|--------|-----|-----|
| synchronous | 08:54:58 | 08:55:58 | 60.0 | 0.0  | 0.0  | 3654.0 | 0.0 | 0.0 | 1792.0 | 0.0 | 0.0 |
|=============|==========|==========|======|======|======|========|=====|=====|========|=====|=====|


ℹ Text Metrics Statistics (Completed Requests)
|=============|=======|=======|=======|=======|=======|=======|=======|=======|========|========|========|========|
| Benchmark   | Input Tokens               |||| Input Words                |||| Input Characters               ||||
| Strategy    | Per Request  || Per Second   || Per Request  || Per Second   || Per Request    || Per Second     ||
|             | Mdn   | p95   | Mdn   | Mean  | Mdn   | p95   | Mdn   | Mean  | Mdn    | p95    | Mdn    | Mean   |
|-------------|-------|-------|-------|-------|-------|-------|-------|-------|--------|--------|--------|--------|
| synchronous | 522.0 | 522.0 | 60.4  | 70.4  | 404.0 | 413.0 | 46.8  | 54.7  | 2739.0 | 2811.0 | 318.1  | 368.9  |
|=============|=======|=======|=======|=======|=======|=======|=======|=======|========|========|========|========|
| Benchmark   | Output Tokens              |||| Output Words               |||| Output Characters              ||||
| Strategy    | Per Request  || Per Second   || Per Request  || Per Second   || Per Request    || Per Second     ||
|             | Mdn   | p95   | Mdn   | Mean  | Mdn   | p95   | Mdn   | Mean  | Mdn    | p95    | Mdn    | Mean   |
|-------------|-------|-------|-------|-------|-------|-------|-------|-------|--------|--------|--------|--------|
| synchronous | 256.0 | 256.0 | 29.6  | 34.5  | --    | --    | --    | --    | --     | --     | --     | --     |
|=============|=======|=======|=======|=======|=======|=======|=======|=======|========|========|========|========|


ℹ Request Token Statistics (Completed Requests)
|=============|=======|=======|=======|=======|=======|=======|=======|=======|=========|========|
| Benchmark   | Input Tok    || Output Tok   || Total Tok    || Stream Iter  || Output Tok      ||
| Strategy    | Per Req      || Per Req      || Per Req      || Per Req      || Per Stream Iter ||
|             | Mdn   | p95   | Mdn   | p95   | Mdn   | p95   | Mdn   | p95   | Mdn     | p95    |
|-------------|-------|-------|-------|-------|-------|-------|-------|-------|---------|--------|
| synchronous | 522.0 | 522.0 | 256.0 | 256.0 | 778.0 | 778.0 | 259.0 | 259.0 | 256.0   | 256.0  |
|=============|=======|=======|=======|=======|=======|=======|=======|=======|=========|========|


ℹ Request Latency Statistics (Completed Requests)
|=============|=========|========|=====|=====|=====|=====|======|======|
| Benchmark   | Request Latency || TTFT     || ITL      || TPOT       ||
| Strategy    | Sec             || ms       || ms       || ms         ||
|             | Mdn     | p95    | Mdn | p95 | Mdn | p95 | Mdn  | p95  |
|-------------|---------|--------|-----|-----|-----|-----|------|------|
| synchronous | 8.6     | 9.6    | 0.0 | 0.0 | 0.0 | 0.0 | 33.8 | 37.4 |
|=============|=========|========|=====|=====|=====|=====|======|======|


ℹ Server Throughput Statistics (All Requests)
|=============|=======|======|=========|==============|===============|==============|
| Benchmark   | Requests             ||| Input Tokens | Output Tokens | Total Tokens |
| Strategy    | Concurrency || Per Sec | Per Sec      | Per Sec       | Per Sec      |
|             | Mdn   | Mean | Mean                                               ||||
|-------------|-------|------|---------|--------------|---------------|--------------|
| synchronous | 1.0   | 1.0  | 0.1     | 70.4         | 34.5          | 105.0        |
|=============|=======|======|=========|==============|===============|==============|


--- 


### 2048 prompt + 512 output tokens, 1 stream, 1 request per stream, 60s duration

ℹ Run Summary Info
|=============|==========|==========|======|======|======|========|=====|=====|========|=====|=====|
| Benchmark   | Timings                              ||||| Input Tokens     ||| Output Tokens    |||
| Strategy    | Start    | End      | Dur  | Warm | Cool | Comp   | Inc | Err | Comp   | Inc | Err |
|             |          |          | Sec  | Sec  | Sec  | Tot    | Tot | Tot | Tot    | Tot | Tot |
|-------------|----------|----------|------|------|------|--------|-----|-----|--------|-----|-----|
| synchronous | 08:56:06 | 08:57:06 | 60.0 | 0.0  | 0.0  | 8232.0 | 0.0 | 0.0 | 2048.0 | 0.0 | 0.0 |
|=============|==========|==========|======|======|======|========|=====|=====|========|=====|=====|


ℹ Text Metrics Statistics (Completed Requests)
|=============|========|========|=======|=======|========|========|=======|=======|=========|=========|========|========|
| Benchmark   | Input Tokens                 |||| Input Words                  |||| Input Characters                 ||||
| Strategy    | Per Request    || Per Second   || Per Request    || Per Second   || Per Request      || Per Second     ||
|             | Mdn    | p95    | Mdn   | Mean  | Mdn    | p95    | Mdn   | Mean  | Mdn     | p95     | Mdn    | Mean   |
|-------------|--------|--------|-------|-------|--------|--------|-------|-------|---------|---------|--------|--------|
| synchronous | 2058.0 | 2058.0 | 117.3 | 156.4 | 1615.0 | 1629.0 | 92.9  | 123.0 | 10774.0 | 10957.0 | 619.6  | 823.8  |
|=============|========|========|=======|=======|========|========|=======|=======|=========|=========|========|========|
| Benchmark   | Output Tokens                |||| Output Words                 |||| Output Characters                ||||
| Strategy    | Per Request    || Per Second   || Per Request    || Per Second   || Per Request      || Per Second     ||
|             | Mdn    | p95    | Mdn   | Mean  | Mdn    | p95    | Mdn   | Mean  | Mdn     | p95     | Mdn    | Mean   |
|-------------|--------|--------|-------|-------|--------|--------|-------|-------|---------|---------|--------|--------|
| synchronous | 512.0  | 512.0  | 29.2  | 38.9  | --     | --     | --    | --    | --      | --      | --     | --     |
|=============|========|========|=======|=======|========|========|=======|=======|=========|=========|========|========|


ℹ Request Token Statistics (Completed Requests)
|=============|========|========|=======|=======|========|========|=======|=======|=========|========|
| Benchmark   | Input Tok      || Output Tok   || Total Tok      || Stream Iter  || Output Tok      ||
| Strategy    | Per Req        || Per Req      || Per Req        || Per Req      || Per Stream Iter ||
|             | Mdn    | p95    | Mdn   | p95   | Mdn    | p95    | Mdn   | p95   | Mdn     | p95    |
|-------------|--------|--------|-------|-------|--------|--------|-------|-------|---------|--------|
| synchronous | 2058.0 | 2058.0 | 512.0 | 512.0 | 2570.0 | 2570.0 | 515.0 | 515.0 | 512.0   | 512.0  |
|=============|========|========|=======|=======|========|========|=======|=======|=========|========|


ℹ Request Latency Statistics (Completed Requests)
|=============|=========|========|=====|=====|=====|=====|======|======|
| Benchmark   | Request Latency || TTFT     || ITL      || TPOT       ||
| Strategy    | Sec             || ms       || ms       || ms         ||
|             | Mdn     | p95    | Mdn | p95 | Mdn | p95 | Mdn  | p95  |
|-------------|---------|--------|-----|-----|-----|-----|------|------|
| synchronous | 17.6    | 17.7   | 0.0 | 0.0 | 0.0 | 0.0 | 34.3 | 34.5 |
|=============|=========|========|=====|=====|=====|=====|======|======|


ℹ Server Throughput Statistics (All Requests)
|=============|=======|======|=========|==============|===============|==============|
| Benchmark   | Requests             ||| Input Tokens | Output Tokens | Total Tokens |
| Strategy    | Concurrency || Per Sec | Per Sec      | Per Sec       | Per Sec      |
|             | Mdn   | Mean | Mean                                               ||||
|-------------|-------|------|---------|--------------|---------------|--------------|
| synchronous | 1.0   | 1.0  | 0.1     | 156.4        | 38.9          | 195.3        |
|=============|=======|======|=========|==============|===============|==============|


---


### 128 prompt + 1024 output tokens, 1 stream, 1 request per stream, 60s duration

ℹ Run Summary Info
|=============|==========|==========|======|======|======|=======|=====|=====|========|=====|=====|
| Benchmark   | Timings                              ||||| Input Tokens    ||| Output Tokens    |||
| Strategy    | Start    | End      | Dur  | Warm | Cool | Comp  | Inc | Err | Comp   | Inc | Err |
|             |          |          | Sec  | Sec  | Sec  | Tot   | Tot | Tot | Tot    | Tot | Tot |
|-------------|----------|----------|------|------|------|-------|-----|-----|--------|-----|-----|
| synchronous | 08:57:22 | 08:58:22 | 60.0 | 0.0  | 0.0  | 276.0 | 0.0 | 0.0 | 2048.0 | 0.0 | 0.0 |
|=============|==========|==========|======|======|======|=======|=====|=====|========|=====|=====|


ℹ Text Metrics Statistics (Completed Requests)
|=============|=======|=======|======|======|=======|=======|======|======|=======|========|======|======|
| Benchmark   | Input Tokens               |||| Input Words              |||| Input Characters            ||||
| Strategy    | Per Request  || Per Second || Per Request  || Per Second || Per Request   || Per Second   ||
|             | Mdn   | p95   | Mdn  | Mean | Mdn   | p95   | Mdn  | Mean | Mdn   | p95    | Mdn   | Mean  |
|-------------|-------|-------|------|------|-------|-------|------|------|-------|--------|-------|-------|
| synchronous | 138.0 | 138.0 | 8.2  | 8.2  | 99.0  | 100.0 | 5.9  | 5.9  | 648.0 | 679.0  | 38.4  | 39.3  |
|=============|=======|=======|======|======|=======|=======|======|======|=======|========|======|======|
| Benchmark   | Output Tokens              |||| Output Words             |||| Output Characters           ||||
| Strategy    | Per Request  || Per Second || Per Request  || Per Second || Per Request   || Per Second   ||
|             | Mdn   | p95   | Mdn  | Mean | Mdn   | p95   | Mdn  | Mean | Mdn   | p95    | Mdn   | Mean  |
|-------------|-------|-------|------|------|-------|-------|------|------|-------|--------|-------|-------|
| synchronous | 1024.0| 1024.0| 60.6 | 60.6 | 127.0 | 435.0 | 7.5  | 16.6 | 779.0 | 2656.0 | 46.1  | 101.7 |
|=============|=======|=======|======|======|=======|=======|======|======|=======|========|======|======|


ℹ Request Token Statistics (Completed Requests)
|=============|=======|=======|========|========|========|========|========|========|=========|========|
| Benchmark   | Input Tok    || Output Tok     || Total Tok      || Stream Iter    || Output Tok      ||
| Strategy    | Per Req      || Per Req        || Per Req        || Per Req        || Per Stream Iter ||
|             | Mdn   | p95   | Mdn    | p95    | Mdn    | p95    | Mdn    | p95    | Mdn     | p95    |
|-------------|-------|-------|--------|--------|--------|--------|--------|--------|---------|--------|
| synchronous | 138.0 | 138.0 | 1024.0 | 1024.0 | 1162.0 | 1162.0 | 1026.0 | 1026.0 | 1.7     | 5.9    |
|=============|=======|=======|========|========|========|========|========|========|=========|========|


ℹ Request Latency Statistics (Completed Requests)
|=============|=========|========|=========|=========|======|======|======|======|
| Benchmark   | Request Latency || TTFT             || ITL        || TPOT       ||
| Strategy    | Sec             || ms               || ms         || ms         ||
|             | Mdn     | p95    | Mdn     | p95     | Mdn  | p95  | Mdn  | p95  |
|-------------|---------|--------|---------|---------|------|------|------|------|
| synchronous | 33.8    | 33.8   | 13282.8 | 28057.1 | 5.6  | 20.0 | 33.0 | 33.0 |
|=============|=========|========|=========|=========|======|======|======|======|


ℹ Server Throughput Statistics (All Requests)
|=============|=======|======|=========|==============|===============|==============|
| Benchmark   | Requests             ||| Input Tokens | Output Tokens | Total Tokens |
| Strategy    | Concurrency || Per Sec | Per Sec      | Per Sec       | Per Sec      |
|             | Mdn   | Mean | Mean                                               ||||
|-------------|-------|------|---------|--------------|---------------|--------------|
| synchronous | 1.0   | 1.0  | 0.0     | 14.5         | 51.9          | 58.9         |
|=============|=======|======|=========|==============|===============|==============|


## GH200 (Isambard-AI)

### 512 prompt + 256 output tokens, 1 stream, 1 request per stream, 60s duration

ℹ Run Summary Info
|=============|==========|==========|======|======|======|=========|=====|=====|=========|=====|=====|
| Benchmark   | Timings                              ||||| Input Tokens      ||| Output Tokens     |||
| Strategy    | Start    | End      | Dur  | Warm | Cool | Comp    | Inc | Err | Comp    | Inc | Err |
|             |          |          | Sec  | Sec  | Sec  | Tot     | Tot | Tot | Tot     | Tot | Tot |
|-------------|----------|----------|------|------|------|---------|-----|-----|---------|-----|-----|
| synchronous | 12:42:46 | 12:43:46 | 60.0 | 0.0  | 0.0  | 22442.0 | 0.0 | 0.0 | 11008.0 | 0.0 | 0.0 |
|=============|==========|==========|======|======|======|=========|=====|=====|=========|=====|=====|


ℹ Text Metrics Statistics (Completed Requests)
|=============|=======|=======|=======|=======|=======|=======|=======|=======|========|========|========|========|
| Benchmark   | Input Tokens               |||| Input Words                |||| Input Characters               ||||
| Strategy    | Per Request  || Per Second   || Per Request  || Per Second   || Per Request    || Per Second     ||
|             | Mdn   | p95   | Mdn   | Mean  | Mdn   | p95   | Mdn   | Mean  | Mdn    | p95    | Mdn    | Mean   |
|-------------|-------|-------|-------|-------|-------|-------|-------|-------|--------|--------|--------|--------|
| synchronous | 522.0 | 522.0 | 384.8 | 391.1 | 403.0 | 410.0 | 297.9 | 302.4 | 2702.0 | 2770.0 | 1990.8 | 2023.1 |
|=============|=======|=======|=======|=======|=======|=======|=======|=======|========|========|========|========|
| Benchmark   | Output Tokens              |||| Output Words               |||| Output Characters              ||||
| Strategy    | Per Request  || Per Second   || Per Request  || Per Second   || Per Request    || Per Second     ||
|             | Mdn   | p95   | Mdn   | Mean  | Mdn   | p95   | Mdn   | Mean  | Mdn    | p95    | Mdn    | Mean   |
|-------------|-------|-------|-------|-------|-------|-------|-------|-------|--------|--------|--------|--------|
| synchronous | 256.0 | 256.0 | 188.7 | 191.8 | --    | --    | --    | --    | --     | --     | --     | --     |
|=============|=======|=======|=======|=======|=======|=======|=======|=======|========|========|========|========|


ℹ Request Token Statistics (Completed Requests)
|=============|=======|=======|=======|=======|=======|=======|=======|=======|=========|========|
| Benchmark   | Input Tok    || Output Tok   || Total Tok    || Stream Iter  || Output Tok      ||
| Strategy    | Per Req      || Per Req      || Per Req      || Per Req      || Per Stream Iter ||
|             | Mdn   | p95   | Mdn   | p95   | Mdn   | p95   | Mdn   | p95   | Mdn     | p95    |
|-------------|-------|-------|-------|-------|-------|-------|-------|-------|---------|--------|
| synchronous | 522.0 | 522.0 | 256.0 | 256.0 | 778.0 | 778.0 | 259.0 | 259.0 | 256.0   | 256.0  |
|=============|=======|=======|=======|=======|=======|=======|=======|=======|=========|========|


ℹ Request Latency Statistics (Completed Requests)
|=============|=========|========|=====|=====|=====|=====|=====|=====|
| Benchmark   | Request Latency || TTFT     || ITL      || TPOT     ||
| Strategy    | Sec             || ms       || ms       || ms       ||
|             | Mdn     | p95    | Mdn | p95 | Mdn | p95 | Mdn | p95 |
|-------------|---------|--------|-----|-----|-----|-----|-----|-----|
| synchronous | 1.4     | 1.4    | 0.0 | 0.0 | 0.0 | 0.0 | 5.3 | 5.3 |
|=============|=========|========|=====|=====|=====|=====|=====|=====|


ℹ Server Throughput Statistics (All Requests)
|=============|=======|======|=========|==============|===============|==============|
| Benchmark   | Requests             ||| Input Tokens | Output Tokens | Total Tokens |
| Strategy    | Concurrency || Per Sec | Per Sec      | Per Sec       | Per Sec      |
|             | Mdn   | Mean | Mean                                               ||||
|-------------|-------|------|---------|--------------|---------------|--------------|
| synchronous | 1.0   | 1.0  | 0.7     | 391.1        | 191.8         | 582.9        |
|=============|=======|======|=========|==============|===============|==============|


--- 


### 2048 prompt + 512 output tokens, 1 stream, 1 request per stream, 60s duration

ℹ Run Summary Info
|=============|==========|==========|======|======|======|=========|=====|=====|=========|=====|=====|
| Benchmark   | Timings                              ||||| Input Tokens      ||| Output Tokens     |||
| Strategy    | Start    | End      | Dur  | Warm | Cool | Comp    | Inc | Err | Comp    | Inc | Err |
|             |          |          | Sec  | Sec  | Sec  | Tot     | Tot | Tot | Tot     | Tot | Tot |
|-------------|----------|----------|------|------|------|---------|-----|-----|---------|-----|-----|
| synchronous | 12:44:26 | 12:45:26 | 60.0 | 0.0  | 0.0  | 45276.0 | 0.0 | 0.0 | 11264.0 | 0.0 | 0.0 |
|=============|==========|==========|======|======|======|=========|=====|=====|=========|=====|=====|


ℹ Text Metrics Statistics (Completed Requests)
|=============|========|========|=======|=======|========|========|=======|=======|=========|=========|========|========|
| Benchmark   | Input Tokens                 |||| Input Words                  |||| Input Characters                 ||||
| Strategy    | Per Request    || Per Second   || Per Request    || Per Second   || Per Request      || Per Second     ||
|             | Mdn    | p95    | Mdn   | Mean  | Mdn    | p95    | Mdn   | Mean  | Mdn     | p95     | Mdn    | Mean   |
|-------------|--------|--------|-------|-------|--------|--------|-------|-------|---------|---------|--------|--------|
| synchronous | 2058.0 | 2058.0 | 756.7 | 792.1 | 1616.0 | 1629.0 | 594.4 | 621.8 | 10832.0 | 10957.0 | 3984.9 | 4171.0 |
|=============|========|========|=======|=======|========|========|=======|=======|=========|=========|========|========|
| Benchmark   | Output Tokens                |||| Output Words                 |||| Output Characters                ||||
| Strategy    | Per Request    || Per Second   || Per Request    || Per Second   || Per Request      || Per Second     ||
|             | Mdn    | p95    | Mdn   | Mean  | Mdn    | p95    | Mdn   | Mean  | Mdn     | p95     | Mdn    | Mean   |
|-------------|--------|--------|-------|-------|--------|--------|-------|-------|---------|---------|--------|--------|
| synchronous | 512.0  | 512.0  | 188.2 | 197.1 | --     | --     | --    | --    | --      | --      | --     | --     |
|=============|========|========|=======|=======|========|========|=======|=======|=========|=========|========|========|


ℹ Request Token Statistics (Completed Requests)
|=============|========|========|=======|=======|========|========|=======|=======|=========|========|
| Benchmark   | Input Tok      || Output Tok   || Total Tok      || Stream Iter  || Output Tok      ||
| Strategy    | Per Req        || Per Req      || Per Req        || Per Req      || Per Stream Iter ||
|             | Mdn    | p95    | Mdn   | p95   | Mdn    | p95    | Mdn   | p95   | Mdn     | p95    |
|-------------|--------|--------|-------|-------|--------|--------|-------|-------|---------|--------|
| synchronous | 2058.0 | 2058.0 | 512.0 | 512.0 | 2570.0 | 2570.0 | 515.0 | 515.0 | 512.0   | 512.0  |
|=============|========|========|=======|=======|========|========|=======|=======|=========|========|


ℹ Request Latency Statistics (Completed Requests)
|=============|=========|========|=====|=====|=====|=====|=====|=====|
| Benchmark   | Request Latency || TTFT     || ITL      || TPOT     ||
| Strategy    | Sec             || ms       || ms       || ms       ||
|             | Mdn     | p95    | Mdn | p95 | Mdn | p95 | Mdn | p95 |
|-------------|---------|--------|-----|-----|-----|-----|-----|-----|
| synchronous | 2.7     | 2.7    | 0.0 | 0.0 | 0.0 | 0.0 | 5.3 | 5.3 |
|=============|=========|========|=====|=====|=====|=====|=====|=====|


ℹ Server Throughput Statistics (All Requests)
|=============|=======|======|=========|==============|===============|==============|
| Benchmark   | Requests             ||| Input Tokens | Output Tokens | Total Tokens |
| Strategy    | Concurrency || Per Sec | Per Sec      | Per Sec       | Per Sec      |
|             | Mdn   | Mean | Mean                                               ||||
|-------------|-------|------|---------|--------------|---------------|--------------|
| synchronous | 1.0   | 1.0  | 0.3     | 792.1        | 197.1         | 989.2        |
|=============|=======|======|=========|==============|===============|==============|


---

### 128 prompt + 1024 output tokens, 1 stream, 1 request per stream, 60s duration

ℹ Run Summary Info
|=============|==========|==========|======|======|======|========|=====|=====|=========|=====|=====|
| Benchmark   | Timings                              ||||| Input Tokens     ||| Output Tokens     |||
| Strategy    | Start    | End      | Dur  | Warm | Cool | Comp   | Inc | Err | Comp    | Inc | Err |
|             |          |          | Sec  | Sec  | Sec  | Tot    | Tot | Tot | Tot     | Tot | Tot |
|-------------|----------|----------|------|------|------|--------|-----|-----|---------|-----|-----|
| synchronous | 12:46:06 | 12:47:06 | 60.0 | 0.0  | 0.0  | 1656.0 | 0.0 | 0.0 | 12288.0 | 0.0 | 0.0 |
|=============|==========|==========|======|======|======|========|=====|=====|=========|=====|=====|


ℹ Text Metrics Statistics (Completed Requests)
|=============|========|========|=======|=======|=======|=======|======|======|=======|========|=======|=======|
| Benchmark   | Input Tokens                 |||| Input Words              |||| Input Characters            ||||
| Strategy    | Per Request    || Per Second   || Per Request  || Per Second || Per Request   || Per Second   ||
|             | Mdn    | p95    | Mdn   | Mean  | Mdn   | p95   | Mdn  | Mean | Mdn   | p95    | Mdn   | Mean  |
|-------------|--------|--------|-------|-------|-------|-------|------|------|-------|--------|-------|-------|
| synchronous | 138.0  | 138.0  | 26.9  | 29.4  | 100.0 | 102.0 | 19.7 | 21.4 | 685.0 | 714.0  | 133.7 | 143.3 |
|=============|========|========|=======|=======|=======|=======|======|======|=======|========|=======|=======|
| Benchmark   | Output Tokens                |||| Output Words             |||| Output Characters           ||||
| Strategy    | Per Request    || Per Second   || Per Request  || Per Second || Per Request   || Per Second   ||
|             | Mdn    | p95    | Mdn   | Mean  | Mdn   | p95   | Mdn  | Mean | Mdn   | p95    | Mdn   | Mean  |
|-------------|--------|--------|-------|-------|-------|-------|------|------|-------|--------|-------|-------|
| synchronous | 1024.0 | 1024.0 | 199.9 | 218.1 | 48.0  | 334.0 | 38.7 | 29.8 | 271.0 | 2083.0 | 241.9 | 180.7 |
|=============|========|========|=======|=======|=======|=======|======|======|=======|========|=======|=======|


ℹ Request Token Statistics (Completed Requests)
|=============|=======|=======|========|========|========|========|========|========|=========|========|
| Benchmark   | Input Tok    || Output Tok     || Total Tok      || Stream Iter    || Output Tok      ||
| Strategy    | Per Req      || Per Req        || Per Req        || Per Req        || Per Stream Iter ||
|             | Mdn   | p95   | Mdn    | p95    | Mdn    | p95    | Mdn    | p95    | Mdn     | p95    |
|-------------|-------|-------|--------|--------|--------|--------|--------|--------|---------|--------|
| synchronous | 138.0 | 138.0 | 1024.0 | 1024.0 | 1162.0 | 1162.0 | 1025.0 | 1027.0 | 2.8     | 3.6    |
|=============|=======|=======|========|========|========|========|========|========|=========|========|


ℹ Request Latency Statistics (Completed Requests)
|=============|=========|========|========|========|=====|=====|=====|=====|
| Benchmark   | Request Latency || TTFT           || ITL      || TPOT     ||
| Strategy    | Sec             || ms             || ms       || ms       ||
|             | Mdn     | p95    | Mdn    | p95    | Mdn | p95 | Mdn | p95 |
|-------------|---------|--------|--------|--------|-----|-----|-----|-----|
| synchronous | 5.1     | 5.1    | 2593.7 | 4814.2 | 0.3 | 2.5 | 5.0 | 5.0 |
|=============|=========|========|========|========|=====|=====|=====|=====|


ℹ Server Throughput Statistics (All Requests)
|=============|=======|======|=========|==============|===============|==============|
| Benchmark   | Requests             ||| Input Tokens | Output Tokens | Total Tokens |
| Strategy    | Concurrency || Per Sec | Per Sec      | Per Sec       | Per Sec      |
|             | Mdn   | Mean | Mean                                               ||||
|-------------|-------|------|---------|--------------|---------------|--------------|
| synchronous | 1.0   | 1.0  | 0.2     | 29.4         | 218.1         | 247.5        |
|=============|=======|======|=========|==============|===============|==============|
