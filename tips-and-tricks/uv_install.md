# UV Install on HPC

UV is a new Python package manager which offers a few helpful features.
[Website is here](https://docs.astral.sh/uv/)

This short guide doesn't explain how to use UV generally but just how to get UV to work out of a directory, which is what you would want if you wanted the Python version stored in the project space.

## Installation

To install UV, run the following commands.
It should be done in the project directory if you want all the users on the project to access it.

`cd /go/to/project/directory`

```bash
mkdir uv_install
mkdir uv_install/bin
mkdir uv_install/pythons
mkdir uv_install/python_bins
mkdir uv_install/cache
mkdir uv_install/tools

curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="${PWD}/uv_install/bin" INSTALLER_NO_MODIFY_PATH=1 sh
```

This only needs to be done once per project and doesn't require `conda` or any module.

## Setting Paths

Do this all the following times you want to use UV.
It needs to be done in the same directory in which you installed it.

```bash
export PATH="${PWD}/uv_install/bin:$PATH"
export UV_CACHE_DIR="${PWD}/uv_install/cache"
export UV_PYTHON_INSTALL_DIR="${PWD}/uv_install/pythons"
export UV_PYTHON_BIN_DIR="${PWD}/uv_install/python_bins"
export UV_TOOL_DIR="${PWD}/uv_install/tools"
```

## Using UV

After it is set up, it is possible to use UV.
One thing you should do to check is run `uv python dir` to check that it points to the `uv_install` folder.
This checks if the Python binaries are being saved to the correct place and `uv python list` can be used to see if the Python version you want is installed.

If the Python version you want isn't there, `uv python install ` can install the version number you want. 

Then, it is possible to use UV as usual.
The help files can be accessed with `uv help`.

To make an envrioment navidate to the python proejct and use `uv venv`.
The environment can be updated by activating the environment and using `uv sync`

## SBATCH and UV

Once UV has set up the environment, it can activated with
`source /path/to/python/project/.venv/bin/activate`

Once this has been called, it should be possible to run the required Python scripts.