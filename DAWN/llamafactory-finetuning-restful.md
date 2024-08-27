# LLaMA-Factory finetuning on DAWN via the RESTful interface

This tutorial assumes that you have already trained your model using **Llama-Factory** and now your aim is to perform inference using the same.
The trained model is stored inside the folder `./saves/llama3-8b-instruct/lora/sft/LLaMA-Factory`.
Now, to perform inference, copy the file [`api_curl.py`](./scripts/api_curl.py) [1] inside the folder `src/`.
The code in this file is making a call to the trained model, which is further hosted as a server at the address http://localhost:8000 or http://0.0.0.0:8000.
User will act as a client and communicate with the server via either the CURL command or the openai API call.

To make the code run on GPU and not on CPU, open the file `LLaMA-Factory/src/llamafactory/model/loader.py` and consider the line 178:

```python
param.data = param.data.to(model_args.compute_dtype)
```

and add the following two lines after them:

```python
model = model.eval().to("xpu") 
torch.autocast(device_type="xpu", enabled=True, dtype=param.data.dtype)
```

To run the model as the server, copy the batch file, [`run_api.sh`](./scripts/run_api.sh) inside the `LLaMA-Factory` folder and run the batch command:

```sh
sbatch run_api.sh
```

This code will run the trained model and host it as a chat server on the address `http://0.0.0.0:8000`.
To query the server about the test data, the user (client) needs to be on the same compute node as the server is.
This is ensured by first checking the compute node of the server code as follows:

```sh
$ sbatch run_api.sh
$ squeue --me
```

![Console output showing the results of the `sbatch` and `squeue` commands](./images/llama-finetune-01.png)

Then the user connects to the same compute node (here `pvc-s-37`) as server by using `ssh` command as follows:

```sh
ssh pvc-s-37
```

![Console output showing the result of the `ssh` command](./images/llama-finetune-02.png)

Once on the same compute node as server, the user can chat with the server by using either of the following two ways:

(i) CURL command:

The user types the following CURL command [2] on the prompt to get a response from the server:

```sh
curl http://pvc-s-37:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer none" \
  -d '{
     "model": "meta-llama/Meta-Llama-3-8B-Instruct",
     "messages": [{"role": "user", "content": "Say this is a test!"}],
     "temperature": 0.7
   }'
```

Please note that the `http://pvc-s-37:8000` in the above command can be replaced by `http://localhost:8000` or `http://0.0.0.0:8000`.
The server responds to this command in a few seconds as follows:

![Console output showing the result of the `curl` command](./images/llama-finetune-03.png)

(ii) Openai API call:

Often, you aim to test the trained Llama model with a large test data.
In such cases, manually calling the train model with CURL commands might not be feasible.
In such cases, you can call the `http://localhost:8000` by reading the test data from a file (here a json file) and calling the chat server by passing it one data point at a time with the Openai API as follows [3]:

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

For example, in the file [`openai_demo.py`](./scripts/openai_demo.py), we read the data from the json file [`valid_demo.py`](./scripts/valid_demo.py) and calls the model for each data point.
The model returns the output as follows:

![Console output showing the result of running `python openai_demo.py`](./images/llama-finetune-04.png)

Reference:

1. https://github.com/hiyouga/LLaMA-Factory/issues/160
2. https://github.com/hiyouga/LLaMA-Factory/issues/4666
3. https://github.com/hiyouga/LLaMA-Factory/issues/222
