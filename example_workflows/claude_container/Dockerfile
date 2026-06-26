FROM docker.io/nvidia/cuda:13.3.0-base-ubuntu26.04

# Install useful linux packages
RUN apt-get update && apt-get install -y --no-install-recommends \
     curl ca-certificates git zsh sudo less unzip tmux jq build-essential cuda-toolkit-13-3 \
  && rm -rf /var/lib/apt/lists/*

# Node 20
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
  && apt-get install -y nodejs \
  && rm -rf /var/lib/apt/lists/*

# Claude Code
RUN npm install -g @anthropic-ai/claude-code

# VS Code CLI (headless, provides `code tunnel`)
RUN curl -fsSL "https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-arm64" \
      -o /tmp/vscode-cli.tar.gz \
  && tar -xzf /tmp/vscode-cli.tar.gz -C /usr/local/bin \
  && rm /tmp/vscode-cli.tar.gz \
  && chmod +x /usr/local/bin/code

# uv (Python package/project manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh \
  && UV_PYTHON_INSTALL_DIR=/opt/uv-python uv python install 3.12 3.13 3.14

# Rename the pre-existing ubuntu user (UID 1000) to dev
RUN usermod -l dev -d /home/dev -m ubuntu \
  && groupmod -n dev ubuntu \
  && chsh -s /bin/zsh dev \
  && echo "dev ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python
USER dev
WORKDIR /home/dev/workspace

