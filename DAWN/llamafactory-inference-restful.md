# LLaMA-Factory inference on DAWN via the RESTful interface

- Author: Navdeep Kaur <nkaur@turing.ac.uk>
- Date: Sat 10 Aug 2024 17:58:00 BST

## Introduction

In this tutorial we will run a pre-trained model based on Llama using GPU acceleration. We will then send it queries and receive back responses from the server.
This tutorial assumes that you have already trained your model using **LLaMA-Factory** and now your aim is to perform inference using the same.
For more info about training the model see the walkthroughs covering finetuning using [srun](llamafactory-finetuning-srun-single.md), [single-XPU sbatch](llamafactory-finetuning-sbatch-single.md) and [multi-XPU sbatch](llamafactory-finetuning-sbatch-multi.md).

## Setting things up

Before you begin you should download the test scripts and data. In the first instance these can be downloaded to your `LLaMA-Factory` directory and moved to their appropriate locations from there.

```sh
cd LLaMA-Factory
curl -O https://raw.githubusercontent.com/alan-turing-institute/hpc-landscape/main/DAWN/scripts/api_curl.py
curl -O https://raw.githubusercontent.com/alan-turing-institute/hpc-landscape/main/DAWN/scripts/run_api.sh
curl -O https://raw.githubusercontent.com/alan-turing-institute/hpc-landscape/main/DAWN/scripts/openai_demo.py
curl -O https://raw.githubusercontent.com/alan-turing-institute/hpc-landscape/main/DAWN/scripts/valid_demo.json
curl -O https://raw.githubusercontent.com/alan-turing-institute/hpc-landscape/main/DAWN/config_node01xpu01.yaml
```

The trained model is stored inside the folder `./saves/llama3-8b-instruct/lora/sft/LLaMA-Factory`.
Now, to perform inference, move the file [`api_curl.py`](./scripts/api_curl.py) [1] inside the folder `src/`.

```sh
mv api_curl.py src/
```

The code in this file makes a call to the trained model, which is further hosted as a server at the address http://localhost:8000 or http://0.0.0.0:8000.
The user will act as a client and communicate with the server via either the `curl` command or using some Python to contact the server using OpenAI API calls.
We cover both approaches below.

To make the code run on GPU and not on CPU, open the file `LLaMA-Factory/src/llamafactory/model/loader.py` and consider line 178:

```python
param.data = param.data.to(model_args.compute_dtype)
```

Add the following two lines after this:

```python
model = model.eval().to("xpu") 
torch.autocast(device_type="xpu", enabled=True, dtype=param.data.dtype)
```

We'll also need to move the `valid_demo.json` data file to an appropriate location.

```sh
mkdir ./data/stepgame_valid
mv valid_demo.json ./data/stepgame_valid/
```

## Running the server

To run the model as the server, copy the batch file [`run_api.sh`](./scripts/run_api.sh) inside the `LLaMA-Factory` directory.
You'll need to edit the file to replace the `<ACCOUNT>` text with your account details and `<PATH_TO_LLAMA-FACTORY>` with the full path to your `LLaMA-Factory` directory.
Then run the batch command:

```sh
sbatch run_api.sh
```

This code will run the trained model and host it as a chat server at the address `http://0.0.0.0:8000`.

```sh
sbatch run_api.sh
squeue --me
```

You should see output similar to that shown in the following screenshot.

![Console output showing the results of the `sbatch` and `squeue` commands](./images/llama-finetune-01.png)

## Troubleshooting

If the server fails with an error there are a few things that may go wrong.
For example if the server is unable to find the model checkpoint you may need to ensure you've completed finetuning of the model first.
Alternatively you may see something like this in the console log:

```
ERROR:    [Errno 98] error while attempting to bind on address ('0.0.0.0', 8000): address already in use
```

In this case you should update the port used in the `src/api_curl.py` to a different number (any number between 1024 and 65535 should be viable).

## Accessing the correct compute node

To query the server using test data, the user (client) needs to be on the same compute node as the server.
This is ensured by first checking the compute node of the server code as follows:

```sh
sbatch run_api.sh
squeue --me
```

The server is only running locally so in order to send queries to it you'll need to be running the query commands from the same compute node.
To connect to the same compute node (in the screenshot above it's `pvc-s-37`) as the server use `ssh` as follows:

```sh
ssh pvc-s-37
```

The following screenshot shows the use of `ssh` to log in to the correct compute node like this:

![Console output showing the result of the `ssh` command](./images/llama-finetune-02.png)

Once on the same compute node as the server, the user can chat with the server by using either of the following two approaches:

## Sending queries using CURL

The user types the following `curl` command [2] in the console to get a response from the server:

```sh
curl http://0.0.0.0:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer none" \
  -d '{
     "model": "meta-llama/Meta-Llama-3-8B-Instruct",
     "messages": [{"role": "user", "content": "Say this is a test!"}],
     "temperature": 0.7
   }'
```

Please note that the `http://0.0.0.0:8000` address in the above command can be replaced by `http://localhost:8000` or the name of your compute node (e.g. `http://pvc-s-37:8000`).
The server responds to this command in a few seconds as follows:

![Console output showing the result of the `curl` command](./images/llama-finetune-03.png)

## Sending queries using the OpenAI API

Often, your aim is to test the trained Llama model with a large test data.
In such cases, manually calling the trained model with `curl` commands might not be feasible.
In such cases, you can call the `http://localhost:8000` by reading the test data from a file (here a json file) and calling the chat server by passing it one data point at a time using the OpenAI API as follows [3]:

```python
import sys
import openai

if __name__ == "__main__":

    openai.api_base = "http://localhost:8000/v1"
    openai.api_key = "none"
    for chunk in openai.ChatCompletion.create(
        model="main",
        messages=[
            {"role": "user", "content": sys.argv[1]}
        ],
        temperature=0.95,
        stream=True
    ):
        if hasattr(chunk.choices[0].delta, "content"):
            print(chunk.choices[0].delta.content, end="", flush=True)
```

For example, in the file [`openai_demo.py`](./scripts/openai_demo.py), we read the data from the json file [`valid_demo.py`](./scripts/valid_demo.py) and call the model for each data point.

```sh
python3 -m venv venv
. ./venv/bin/activate
pip install openai==0.28
python3 ./openai_demo.py
```

The model returns the output as follows:

![Console output showing the result of running `python openai_demo.py`](./images/llama-finetune-04.png)

## Tidying up

In order to stop the server you can use the following command. Note that this will cancel all of your running jobs and shut down all of your connections to them, so if you have more than one job running you should cancel the specific job rather than using this command.

```sh
scancel --me
```

## References

1. https://github.com/hiyouga/LLaMA-Factory/issues/160
2. https://github.com/hiyouga/LLaMA-Factory/issues/4666
3. https://github.com/hiyouga/LLaMA-Factory/issues/222
