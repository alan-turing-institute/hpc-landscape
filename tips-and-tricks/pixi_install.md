# Pixi Install on HPC

Pixi is a package manager which can produce reproducible lock files like `uv` but supports multiple languages like `conda`.
[Website is here.](https://pixi.sh/latest/)

This guide doesn't explain how to use Pixi, but rather how to set it up on an HPC.
This method shouldn't require root access and should be usable by multiple users at the same time.

## Installation

The installation of Pixi should be done in the project environment so that all the users of the project can access the files necessary to load the environment.

```bash
cd /go/to/project/directory
```

What the install will do is download and install the Pixi binary and set up the folders to store the cache. 

```bash
mkdir pixi_install
mkdir pixi_install/bin
mkdir pixi_install/cache
mkdir pixi_install/envs

curl -fsSL https://pixi.sh/install.sh | \
    PIXI_HOME="./pixi_install" \
    PIXI_NO_PATH_UPDATE=true \
    bash
```

This should only need to be done once per project, as even multiple Pixi environments can use the same install.

## Setting up Pixi

In order to use Pixi, a collection of environment variables needs to be set.
This can be done by using the following from the same directory as the previous code snippet.

```bash
export PATH="${PWD}/pixi_install/bin:${PATH}"
export PIXI_HOME="${PWD}/pixi_install"
export PIXI_CACHE_DIR="${PWD}/pixi_install/cache"
```

This needs to be done every time you log in and want to use Pixi; therefore, setting up a shell script with these commands might be helpful.

## Using the Environment

To use the environment, you first need to set up Pixi as described above and call `pixi install` to configer it originally.
Then the environment can be used with `pixi shell` and `pixi run`.
For example, `pixi run python my_code.py` would be a valid command (if Python is included in our Pixi environment).

Pixi is more powerful than Python virtual environments, as it can include things outside of PyPI, such as a CUDA compiler.
This does mean, though, that it is not as easy to activate, and the Pixi does need to be set up each time.
There is also a way around this by using `pixi shell-hook`, but instead, I would include a file to set up Pixi in your _pixi_install_ folder, which resembles the following.
Then, if this script is run in your sbatch, you can use Pixi as usual.

```sh
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export PATH="${SCRIPT_DIR}/bin:${PATH}"
export PIXI_HOME="${SCRIPT_DIR}"
export PIXI_CACHE_DIR="${SCRIPT_DIR}/cache"

echo "Set Pixi Home to ${SCRIPT_DIR}"
```
