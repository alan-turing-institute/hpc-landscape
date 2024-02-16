"""Let accelerate do the XPU placement.

Remember to set CCL_WORKER_COUNT=1 before running."""

import os
import time
import warnings
import re

with warnings.catch_warnings():
    # Silence the torchvision warning.
    warnings.simplefilter("ignore")
    import torch
    from accelerate import Accelerator

rank = os.environ["PMI_RANK"]
os.environ["RANK"] = rank

size = os.environ["PMI_SIZE"]
os.environ["WORLD_SIZE"] = size

# Let MPI set the local rank.
# Otherwise all models end up on xpu:0.
# See accelerate/state.py, line 177ish for more.
os.environ["ACCELERATE_TORCH_DEVICE"] = "xpu:" + os.environ["MPI_LOCALRANKID"]

nodelist_env = os.getenv("SLURM_JOB_NODELIST")
if "[" in nodelist_env:
    # either a single hostname or e.g. "pvc-s-[24-25]"
    numbers = re.compile(r"\d+")
    prefix = nodelist_env[0 : nodelist_env.index("[")]

    nodelist = tuple(prefix + x for x in numbers.findall(nodelist_env))
else:
    nodelist = (nodelist_env,)
master_addr = nodelist[0]

os.environ["MASTER_ADDR"] = master_addr
os.environ["MASTER_PORT"] = "29878"

accelerator = Accelerator()

model = torch.nn.Sequential(
    torch.nn.Linear(in_features=1, out_features=1_000),
    torch.nn.Linear(in_features=1_000, out_features=1_000),
    torch.nn.Linear(in_features=1_000, out_features=1),
)
optimizer = torch.optim.Adam(model.parameters())

weight = 0.7
bias = 0.3
start = 0
stop = 1
step = 0.001

X = torch.arange(start, stop, step).unsqueeze(dim=1)
y = X * weight + bias

dataset = []
for i in range(len(X)):
    dataset.append([X[i], y[i]])

data = torch.utils.data.DataLoader(dataset)

print("preparing", flush=True)
model, optimizer, data = accelerator.prepare(model, optimizer, data)

loss_func = torch.nn.L1Loss()
model.train()
print("training", flush=True)
start_counter = time.perf_counter()
start_time = time.time()
for epoch in range(50):
    for source, targets in data:
        optimizer.zero_grad()

        output = model(source)

        loss = loss_func(targets, output)

        accelerator.backward(loss)

        optimizer.step()

    if epoch % 10 == 0:
        print(f"{epoch=}", flush=True)


print("success. loss was", loss, flush=True)
print("counter was", time.perf_counter() - start_counter, flush=True)
print("time taken was", time.time() - start_time, flush=True)
