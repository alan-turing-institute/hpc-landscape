A DDP example based on Multi-GPU AI Training (Data-Parallel) with Intel® Extension for PyTorch [Intel's example](https://www.youtube.com/watch?v=3A8AVsNNHOg).

The adjustments were made by Kacper.

Please note that `dist.init_process_group` uses a file path to initialize the process group, which at the moment points to a file in Kacper's directory. You will need to change this to your own file path.

```python
        init_method = 'file:///rds/user/kk562/hpc-work/ddp-parallel/sync_file'
```
