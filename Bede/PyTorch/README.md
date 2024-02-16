# PyTorch on GraceHopper

The GraceHoper superchip uses an ARM CPU combined with an H100 GPU on a single chip.
This makes it unusual in being both ARM and CUDA based.
Official [PyTorch binaries](https://pytorch.org/get-started/locally/) aren’t yet available for this configuration and so must be built from source.
PyTorch offer [decent instructions](https://github.com/pytorch/pytorch#from-source) for how to build wheels for use across multiple configurations and the Bede docs also contains some relevant instructions.
However, following these directly didn’t give successful results.

The steps below are those used to build PyTorch on the GraceHopper login node.
Following these steps may save some time in working around problems when building PyTorch for GraceHopper devices.

Before starting it’s worth noting that on the Bede login nodes building with multiple processes (e.g. using `-j 16`) caused build failures due to memory exhaustion.
On the flipside, building with just a single process (`-j 1`) was very slow.
Since the build is incremental, the best solution I found was to build with 16 processes until the build failed (which happened towards the end), then set the build off again using just a single process.
In this way the bulk of the build completed quickly and only the troublesome part at the end was left to run slowly.
It means you can’t run the build fully non-interactively though.
If you really need a non-interactive build, you may need to run the whole thing using just one process.

## Prebuilt wheels

You can find prebuilt wheels created using the steps described here [on SharePoint](https://thealanturininstitute.sharepoint.com/:f:/s/ResearchComputing/ElyYboALQwhKiki7jnk0DPMBC6548bnDCEOx6U5CIAypoA?e=4qKvC3) (requires an Alan Turing Institute account).

If you’d like to build your own version, then read on.

## Log in

First we log in to Bede, then to the GraceHopper login node.
Finally we also do a quick check to ensure we’re on an aarch64 device.

```
$ ssh <username>@bede.dur.ac.uk
$ ghlogin -A <account>
$ lscpu | head
Architecture:                       aarch64
CPU op-mode(s):                     64-bit
Byte Order:                         Little Endian
CPU(s):                             72
On-line CPU(s) list:                0-71
Vendor ID:                          ARM
Model name:                         Neoverse-V2
Model:                              0
Thread(s) per core:                 1
Core(s) per socket:                 72
```

## Get the PyTorch source code

Clone the PyTorch repository to a suitable folder (ideally on Bede’s Lustre filesystem).

```
$ git clone --recursive https://github.com/pytorch/pytorch
```

## Set up the environment

This step is specific to the Bede environment.
It’s used to make available the CUDA drivers, suitable build tools and development libraries.

```
$ module load gcc/12.2 openmpi/4.1.6 cuda/12.3.2 nsight-systems/2023.4.1 openblas/0.3.26
```

## Install Miniconda

```
$ mkdir miniconda3
$ curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh
$ sh Miniconda3-latest-Linux-aarch64.sh -b -p ${PWD}/miniconda3
$ rm Miniconda3-latest-Linux-aarch64.sh
```

## Install the prerequisites

Now we have things set up we can start broadly following these instructions from the [PyTorch docs](https://github.com/pytorch/pytorch#from-source).
However we’ve tailored the steps to include specific changes needed for the CUDA/ARM environment provided by the GraceHopper superchip.

```
$ conda activate
$ cd pytorch
$ conda install cmake ninja
$ pip install -r requirements.txt
$ export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
$ export USE_PRIORITIZED_TEXT_FOR_LD=1
```

## Configure build script

We now need to configure the cmake script to take into account features available in the compilers on the Bede system.
To do this we run just the initial build steps needed to set up cmake.
We’ll then make our changes and continue the build from there.

```
$ python setup.py build --cmake-only
```

We need to update the `build/CMakeCache.txt` file so that `CMAKE_CXX_FLAGS` is set to `-ffunction-sections -fdata-sections -flax-vector-conversions -Wno-nonnull`.
We can edit the file to make the change manually, but an easier approach is just to run the following command.

```
$ cmake -DCMAKE_CXX_FLAGS:STRING="-ffunction-sections -fdata-sections -flax-vector-conversions -Wno-nonnull" build
```

## Start the build

We’ll now start the build.

```
$ python setup.py develop
```

This will run in 16 processes by default but will still take quite some time, so now might be a good time to make yourself a coffee.
Running in 16 processes is great for decreasing build time, but not so great for the amount of memory needed for the build.
When testing this I found that the build wouldn’t go through with 16 processes.
To get the build through I had to reduce it to use just a single process.
We don’t want to run with just a single process from the start though &mdash; it’ll take way too long &mdash; so we run with multiple processes, then restart the build from where it left off using just a single process once it’s failed.

If you’re luck the build may go through; but if it does fail due to memory exhaustion you should set it off again.
This time we’re building with just a single process, like this:

```
$ CMAKE_BUILD_PARALLEL_LEVEL=1 python setup.py develop
```

Hopefully, after some time, the build will succeed. We’re not quite there yet, one more step to go.

## Build the Wheel

It’s convenient to have PyTorch built as a wheel so you can use it with multiple projects installed in different virtual environments.
The following command will build the wheel.

```
$ python setup.py bdist_wheel
$ cp dist/torch-2.4.0a0+gitbad8d25-cp312-cp312-linux_aarch64.whl ..
```

## Tidy up

If you’re done here you can deactivate the conda environment.

```
$ deactivate
```

## The Wheel

You should be left with a wheel called something like `torch-2.4.0a0+gitbad8d25-cp312-cp312-linux_aarch64.whl`.
Note the different parts of this name:
1. `torch`: the name of the package.
2. `2.4.0a0+gitbad8d25`: the package version, here we’ve built from a specific git commit that lives under the `2.4.0a0` tag.
3. `cp312`: the Python distribution tag; CPython 3.12, matching the Python we built the package with.
4. `cp312`: the Python ABI tag: CPython 3.12 for the same reason.
5. `linux_aarch64`: the platform tag, in this case Linux running on an aarch64 (ARM) processor.

If we try to install the package on a platform where these aren’t compatible, the install will fail.
There’s no indication here that we installed for CUDA, but PyTorch should be smart enough to pick it up and use it.

## Troubleshooting

There are a variety of errors that can arise when building the package.
These are all errors I hit while trying to build PyTorch and I’ve included details of how I worked around them.

### Cannot convert ‘int16x4_t’ to ‘uint16x4_t’

```
pytorch/aten/src/ATen/native/cpu/int4mm_kernel.cpp:378:86: error: cannot
    convert ‘int16x4_t’ to ‘uint16x4_t’
/usr/lib/gcc/aarch64-redhat-linux/11/include/arm_neon.h:1556:38: note:
    initializing argument 2 of ‘uint16x4_t vand_u16(uint16x4_t, uint16x4_t)’
 1556 | vand_u16 (uint16x4_t __a, uint16x4_t __b)
      |                           ~~~~~~~~~~~^~~
[5453/6512] Building CXX object caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/
    ATen/native/quantized/cpu/kernels/QuantizedOpKernels.cpp.DEFAULT.cpp.o
ninja: build stopped: subcommand failed.
```

This is caused by `-flax-vector-conversions` not being activated for `gcc`.
To fix this open `build/CMakeCache.txt` and search for `CMAKE_CXX_FLAGS`.
Ensure the line is as follows:

```
CMAKE_CXX_FLAGS:STRING=-ffunction-sections -fdata-sections -flax-vector-conversions -Wno-nonnull
```

If not amend it as necessary, then kick off the build again.

### Died due to signal 9

```
nvcc error   : 'cicc' died due to signal 9 (Kill signal)
[6443/7088] Building CUDA object caffe2/CMakeFiles/torch_cuda.dir/__/aten/src/
    ATen/native/transformers/cuda/flash_attn/kernels/flash_bwd_hdim64_fp16_sm80.cu.o
ninja: build stopped: subcommand failed.
```

You ran out of memory.
Try kicking off the build again but using fewer processes, all the way down to 1 if necessary:

```
$ CMAKE_BUILD_PARALLEL_LEVEL=1 python setup.py develop
```

### Not a supported wheel on this platform

```
ERROR: torch-2.4.0a0+gitbad8d25-cp312-cp312-linux_aarch64.whl is not a supported wheel on this platform.
```

The version of Python you used to build the package doesn’t match with the version of Python you’re trying to use it with.

