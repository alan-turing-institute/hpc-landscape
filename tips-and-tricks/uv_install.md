# UV Install on HPC

UV is a new Python package manager which offers a few helpful features.
[Website is here](https://docs.astral.sh/uv/).

This short guide doesn't explain how to use UV generally, but just how to get UV to work out of a directory, which is what you would want if you wanted the Python version stored in the project space.

This approach is likely to be especially helpful on HPC systems, since no root access, nor an existing Python installation, is required to install it this way.

## Installation

To install UV, run the following commands.
The default form UV is to use the home directory; however, this can cause issues when other users on the project attempt to activate the environment.
Therefore, it should be installed in the project directory, so the first step is to navigate there.

```bash
cd /go/to/project/directory
```

When you are in the desired location, the next step is to make the necessary file structure and install the UV binary there.

```bash
mkdir uv_install
mkdir uv_install/bin
mkdir uv_install/pythons
mkdir uv_install/python_bins
mkdir uv_install/cache
mkdir uv_install/tools

curl -LsSf https://astral.sh/uv/install.sh | \
    env UV_INSTALL_DIR="${PWD}/uv_install/bin" INSTALLER_NO_MODIFY_PATH=1 sh
```

This only needs to be done once per project and doesn't require `conda` or any other module.

## Setting Paths

Every time you log in and want to use UV, you need to set these environment variables.
It needs to be done in the same directory in which you installed it.

```bash
export PATH="${PWD}/uv_install/bin:$PATH"
export UV_CACHE_DIR="${PWD}/uv_install/cache"
export UV_PYTHON_INSTALL_DIR="${PWD}/uv_install/pythons"
export UV_PYTHON_BIN_DIR="${PWD}/uv_install/python_bins"
export UV_TOOL_DIR="${PWD}/uv_install/tools"
```

It is a reasonable idea to save these in a shell script with the UV install, but it isn't required.

## Using UV

After it is set up, it is possible to use UV.
One thing you should do to check is run `uv python dir` to check that it points to the `uv_install` folder.
This checks if the Python binaries are being saved to the correct place and `uv python list` can be used to see if the Python version you want is installed.

If the Python version you want isn't there, `uv python install` can install the version number you want. 

Then, it is possible to use UV as usual.
The help files can be accessed with `uv help`.

To make an environment, navigate to the Python project and use `uv venv`.
The environment can be updated by activating the environment and using `uv sync`

## SBATCH and UV

Once UV has set up the environment, it can be activated with

```bash
source /path/to/python/project/.venv/bin/activate
```

Once this has been called, it should be possible to run the required Python scripts.
This is preferable over using `uv run`, as the environment variables do not need to be set to run Python scripts; they are only required to update the environment.
