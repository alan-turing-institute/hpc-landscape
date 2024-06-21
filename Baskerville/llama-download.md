# Llama download on Baskerville

We’ll be downloading the models from:

https://llama.meta.com/llama-downloads

This involves agreeing to the T&Cs in order to get keys, then following the instructions in the [REAMDE](https://github.com/meta-llama/llama3).

We can do all of this on a Baskerville login node, so make sure you’ve SSH-ed into Baskerville before continuing.

## 1. Clone the repositories:

```shell
$ git clone https://github.com/meta-llama/llama3.git
$ git clone https://github.com/meta-llama/llama
$ git clone https://github.com/meta-llama/codellama.git
```

## 2. Download Meta Llama 3

Run the download script and supply the relevant URL.
The prerequisites (`wget` and `md5sum`) are already installed on Baskerville.

```shell
$ cd llama3
$ ./download.sh
Enter the URL from email: https://download6.llamameta.net/*?...

Enter the list of models to download without spaces (8B,8B-instruct,70B,70B-instruct), or press Enter for all: <ENTER>
```

The total time taken to download all of the models:

```shell
real    36m55.852s
user    12m29.470s
sys     13m3.270s
```

The size of the files on disk:

```shell
$ du -hs Meta-Llama-3-*
132G    Meta-Llama-3-70B
132G    Meta-Llama-3-70B-Instruct
15G     Meta-Llama-3-8B
15G     Meta-Llama-3-8B-Instruct
```

## 3. Download Meta Llama

Run the download script and supply the relevant URL:

```shell
$ cd llama
$ ./download.sh
Enter the URL from email: https://download6.llamameta.net/*?...

Enter the list of models to download without spaces (8B,8B-instruct,70B,70B-instruct), or press Enter for all: <ENTER>
```

The total time taken to download all of the models:

```shell
real    42m49.237s
user    14m12.291s
sys     14m53.573s
```

The size of the files on disk:

```shell
$ du -hs llama-2-*
25G     llama-2-13b
25G     llama-2-13b-chat
129G    llama-2-70b
129G    llama-2-70b-chat
13G     llama-2-7b
13G     llama-2-7b-chat
```

## 4. Download Meta Code Llama

```shell
$ cd codellama
$ ./download.sh
Enter the URL from email: https://download6.llamameta.net/*?...

Enter the list of models to download without spaces (8B,8B-instruct,70B,70B-instruct), or press Enter for all: <ENTER>
```

The total time taken to download all of the models:

```shell
real    324m52.569s
user    29m56.629s
sys     35m47.038s
```

The size of the files on disk:

```shell
$ du -hs CodeLlama-*
25G     CodeLlama-13b
25G     CodeLlama-13b-Instruct
25G     CodeLlama-13b-Python
63G     CodeLlama-34b
63G     CodeLlama-34b-Instruct
63G     CodeLlama-34b-Python
129G    CodeLlama-70b
48G     CodeLlama-70b-Instruct
129G    CodeLlama-70b-Python
13G     CodeLlama-7b
13G     CodeLlama-7b-Instruct
13G     CodeLlama-7b-Python
```

## 5. Make use of the models.

Having downloaded the models you’ll want to make use of them.
Take a look at one of the following files to see how to execute them for inference on Baskerville:
1. `llama3-inference-srun.md`: execute from the command line using `srun`.
2. `llama3-inference-sbatch.md` execute using a batch script.


