# Print Nvidia GPU Stats

It is common to want to see a readout of GPU usage when working on an HPC to test that your scripts are operating as you expect them to.
One of the simplest ways to do this is to have a second terminal window (a possibility with TMUX), which regularly prints out the GPU statistics. 

A useful command for Nvidia GPUs is `nvidia-smi`, which displays information about the available GPUs and what processes are running.
The following command prints out some useful GPU statistics formatted into a line per GPU.

```bash
nvidia-smi -l 1 --query-gpu=index,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | awk -F ',' '{printf "GPU %d | Temp: %d°C | Util: %3d%% | Mem: %05dMB/%dMB\n", $1, $2, $3, $4, $5}'
```

This ends up returning outputs like:

```text
GPU 0 | Temp: 72°C | Util: 100% | Mem: 16384MB/24576MB
GPU 1 | Temp: 46°C | Util:   3% | Mem: 04576MB/24576MB
```

Here is a breakdown of what this command is doing:

* `nvidia-smi` - main command to view Nvidia stats
* `-l 1` - repeat command every second
* `--query-gpu=index,temperature.gpu,utilization.gpu,memory.used,memory.total` - get a choice of GPU statistics
* `--format=csv,noheader,nounits` output in a format that can be piped
* `awk -F ','` splits the given CSV into values
* ` '{printf "GPU %d | Temp: %d°C | Util: %3d%% | Mem: %05dMB/%dMB\n", $1, $2, $3, $4, $5}'` - formats the output so repeated calls are aligned

If you are using a GPU with more than 100 GB or less than 10GB, it may be worth changing the `5` in `Mem: %05dMB/%dMB` to show the correct number of digits. 