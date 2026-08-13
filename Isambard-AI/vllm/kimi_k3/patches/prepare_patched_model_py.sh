#!/bin/bash
# vim: et:ts=4:sts=4:sw=4
#
# Applies patch to current container state.
# Fails if container is already fixed.
#
# Exports PATCHED_MODEL_PY as the path to the patched file.
# Usage: source this file from within the batch script, PRIMARY_HOST must already be set.
set -e

mkdir -p patches/_generated
export PATCHED_MODEL_PY="$PWD/patches/_generated/model.py.patched"

srun --nodes=1 --ntasks=1 -w "$PRIMARY_HOST" \
    apptainer exec --nv container/kimi-k3.sif python3 \
    "$PWD/patches/fix_kimi_k3_ep_padding.py" \
    /usr/local/lib/python3.12/dist-packages/vllm/models/kimi_k3/nvidia/model.py \
    "$PATCHED_MODEL_PY"

echo "Using patched model.py: $PATCHED_MODEL_PY"
