# DGX Spark Admin Setup

This doc covers initial set up of the sparks and guidance for admins setting up new users.

See also [getting started docs](https://github.com/alan-turing-institute/hpc-landscape/blob/main/DGX-Spark/getting_started.md) for general user advice.

## System wide packages (install once)

### Podman

Podman is an alternative to Docker which doesn't require root access when running containers. It should be installed systemwide for all users to be able to use.

Install using:

```bash
sudo apt update && sudo apt install -y podman
```

Then verify by running:
```bash
podman --version
```

### Nvitop (GPU monitor)

Since we have no user workload management system set up on the DGX Sparks, we are using [nvitop](https://nvitop.readthedocs.io/en/latest/) to help with this. It is a CLI tool that shows you the current GPU usage on the Spark, what processes are running and which user launched those processes. It should be installed systemwide for all users.

First install `pipx` using: 

```bash
sudo apt install -y pipx
```

Then install nvitop:
```bash
sudo PIPX_HOME=/opt/pipx PIPX_BIN_DIR=/usr/local/bin pipx install nvitop
```

Verify your installation by running:
```bash
which nvitop
nvitop --version
```

## Adding new users

New users can be created with `useradd`:

```bash
sudo useradd -m -G users -s /bin/bash <username>
```

- `-m` — creates home directory at `/home/<username>`
- `-G users` — adds to the `users` group
- `-s /bin/bash` — sets bash as the login shell


You should then set the password for the user using:

```bash
sudo passwd <username>
```

To check its worked, run:

```bash
id <username>
grep <username> /etc/passwd
ls /home/<username>
```

