# Claude Container

There might be a case where you would like to use a coding agent with GPU access, but only to the folder of interest.
This could be combined usefully with VS Code to either interact with the agent or just monitor the progress.
On shared systems, [podman](https://podman.io/) is often used instead of Docker for containers because it does not require root access to set up a container.
Podman takes a Dockerfile and can pull from `docker.io`, so the setup is the same as for Docker.

## The Dockerfile

There is an example [Dockerfile](example_workflows/claude_container/Dockerfile) which can be used.
It is based on an image from NVIDIA using Ubuntu.

```dockerfile
FROM docker.io/nvidia/cuda:13.3.0-base-ubuntu26.04
```

If CUDA 12 is needed for older GPUs, you can use the following image.

```dockerfile
FROM docker.io/nvidia/cuda:12.9.2-base-ubuntu24.04
```

The next section installs a collection of useful Linux packages.
This can be adjusted to fit your use case.
One important package to flag is `cuda-toolkit-13-3`, which provides some useful tools for developing with an NVIDIA GPU.
It is very large, so if it is not needed, it might be better to be skipped, and it needs to match the CUDA version, so for the alternative example above, it would require `cuda-toolkit-12-9`.

```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
     curl ca-certificates git zsh sudo less unzip tmux jq build-essential cuda-toolkit-13-3 \
  && rm -rf /var/lib/apt/lists/*
```

The section installs Claude by using Node; this should be changed if a different coding agent is needed.

```dockerfile
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
  && apt-get install -y nodejs \
  && rm -rf /var/lib/apt/lists/*

RUN npm install -g @anthropic-ai/claude-code
```

The next step installs the VS Code Server, which can be used to set up a tunnel or when ssh in with VS Code externally.
An important point here is the `os=cli-alpine-arm64` flag, which is the binary for the ARM CPUs, but if the system runs on Intel or AMD, then this needs to be changed to `os=cli-alpine-x64`

```dockerfile
RUN curl -fsSL "https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-arm64" \
      -o /tmp/vscode-cli.tar.gz \
  && tar -xzf /tmp/vscode-cli.tar.gz -C /usr/local/bin \
  && rm /tmp/vscode-cli.tar.gz \
  && chmod +x /usr/local/bin/code
```

The next step installs `uv` and some Python versions.
As the cache isn't stored externally from the container, `uv sync` will need to be called each time you want to start using the virtual environment. 

```dockerfile
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh \
  && UV_PYTHON_INSTALL_DIR=/opt/uv-python uv python install 3.12 3.13 3.14
```

The final step is setting the Linux user to dev, adding the environment variables, and pointing at the working directory.

```dockerfile
RUN usermod -l dev -d /home/dev -m ubuntu \
  && groupmod -n dev ubuntu \
  && chsh -s /bin/zsh dev \
  && echo "dev ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python
USER dev
WORKDIR /home/dev/workspace
```

## Building

To set up the container for the first time, you use the following command.

```bash
podman build -t claude-dev .
```

It is also a good idea to create a folder for the VS Code server files; this will speed up gaining access with the VS Code Tunnel when the container is used repeatedly.

```bash
mkdir claude-vscode-server
```

## Running

To run the container, you can use the following command.
Remember that this assumes the Dockerfile is in `~/claude-container`, and you will need to set `/path/to/project/folder` to the codebase you want to work on.

```bash
podman run -it --rm \
  --userns=keep-id \
  --device=nvidia.com/gpu=all \
  --security-opt=label=disable \
  -v ~/claude-container/claude-vscode-server:/home/dev/.vscode-server \
  -v /path/to/project/folder:/home/dev/workspace \
  claude-dev
```

This command sets up an interactive session because of the `-i` flag.
Another option would be to omit that flag and call `sleep infinity`, which has the container persist in the background.
To do this, it might be worth running it in detached (`-d`) mode.

The container can be accessed by calling `podman exec -it claude-dev`.
You can check that it is running by using `podman ps`.
This container was set up with the intent of calling `podman exec -it claude-dev code tunnel`, which can then be used to access the code base internally.

## Persisting Information

This container, at the moment, only persists the changes to the codebase and some useful VS Code installs.
It is possible to expand this list by adding other mounts.
The other option would be to not use the `--rm` flag, and then the container can be restarted, as it is not deleted after use.
These can be set up by creating a directory and mounting it with the `-v` flag.
When working on a shared system, remember to `chmod -R 700` folders with keys you do not want other people to access.

* `~/.vscode` - This would save your tunnel authentication token.
* `~/.claude` - This would save your Claude authentication and also the conversation history.
* `~/.cache/uv` - This would save your UV cache, speeding up the first `uv sync` on the code base after the container is opened.

It is also possible to mount other directories, such as `~/.ssh`, which might have useful features for GitHub.

## Isambard

On Isambard, you need to interact with containers slightly differently.
This mostly involves changing the calls from `podman` to `podman-hpc`.
There is some subtlety with the need to either migrate the containers or build them on the compute node in question.
I also found some benefit in using the `--userns=keep-id:uid=1000,gid=1000` option so that the permissions are consistent when editing inside and outside the container.
I also updated the user section of the Dockerfile as follows.

```dockerfile
# Set User IDs to match Isambard
RUN userdel -r ubuntu 2>/dev/null || true \
  && groupadd -g 1000 dev \
  && useradd -m -d /home/dev -s /bin/zsh -u 1000 -g 1000 dev \
  && echo "dev ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
```

More information on using containers on Isambard is available here.
<https://docs.isambard.ac.uk/user-documentation/guides/containers/>
