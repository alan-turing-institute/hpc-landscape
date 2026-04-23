# Getting PyTorch using UV on Isambard-AI

Below is a minimal `pyproject.toml` example which will install torch on both a Mac and Isambard-AI for use with the GPU.

There are some issues with the availability of `aarch64` and `CUDA` builds of PyTorch on the standard Python Package Index (PyPI).
This tells UV where to look for the torch version when on `aarch64` and not to look there otherwise.
As standard, Isambard-AI has `CUDA 12.6`, hence that choice of version.
When setting the indices, the order matters, so if `pytorch-cu126` was listed first, the install would not happen correctly, either on Isambard or elsewhere.

```toml
[project]
name = "gettorch"
version = "0.1.0"
description = "Getting Torch on Isambard AI"
requires-python = ">=3.12.0"
dependencies = [
  "torch>=2.10",
]

## -------
## Here is what is special for torch
[tool.uv.sources]
torch = [
  { index = "pytorch-cu126", marker = "platform_machine == 'aarch64'"},
  { index = "pypi", marker = "platform_machine != 'aarch64'"},
]

[[tool.uv.index]]
name = "pypi"
url = "https://pypi.org/simple"

[[tool.uv.index]]
name = "pytorch-cu126"
url = "https://download.pytorch.org/whl/cu126"
## ------
```