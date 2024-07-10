# DAWN

**Last updated: 2024-07-10**

Dawn is a new supercomputing resource hosted by the Cambridge Open Zettascale Lab and run by the University of Cambridge Research Computing Service (RCS).
It’s been developed as part of the Artificial Intelligence Research Resource (AIRR) project and is billed as the UK’s [fastest artificial intelligence supercomputer](https://www.hpc.cam.ac.uk/d-w-n).

## DAWN Hardware

DAWN is made up of 256 Dell PowerEdge XE9640 servers, each consisting of:

1. 2 &times; Intel&reg; Xeon&reg; Platinum 8468 (96 cores in total).
2. 1024 GiB RAM.
3. 4 &times; Intel&reg; Data Center GPU Max 1550 GPUs (each with 128 GiB GPU RAM, two stacks, 1024 compute units).
4. Xe-Link 4-way GPU interconnect within the node.
5. Quad-rail NVIDIA (Mellanox) HDR200 InfiniBand interconnect.

## Glossary

RCS: the “Research Computing Service” at the University of Cambridge is the equivalent of Turing’s Research Computing Platforms team.

CSD3: the “Cambridge Service for Data Driven Discovery”.
This is the service through which DAWN is accessed, run by RCS at Cambridge.

PVC: the Intel Data Center GPU Max 1550 GPU was originally codenamed “Ponte Vecchio” and is sometimes referred to as a “PVC GPU”.
In development libraries such as PyTorch it’s likely to be referred to as an “XPU”.

## Accessing compute

In order to use DAWN you will first need an account on the system.
The process for getting a project allocation and agreement to use the system falls outside the scope of this walkthrough.
However, having gained this agreement, we will detail the steps we went through in order to set the our account up.

Having liaised with RCS at Cambridge and assuming that they agree for a project to make use of DAWN, the next steps would be:

1. Receive an email from RCS with further details.
2. Complete the CSD3 external [applications form](https://www.hpc.cam.ac.uk/external-application), with “Account type” set to “Other/Dawn Early Access”.
   Please don’t complete this form before you’ve received confirmation from the RCP Team at the Turing.
3. Await confirmation that your account has been created.
4. Configure your SSH client for access.
5. Request space on the “Research Data Store” Lustre filesystem.

## Accessing storage

In order to make practical use of DAWN you’ll need more storage than your home folder provides.
Having created your account on the system, the next step should therefore be to then request project storage for your work on the [Research Data Store](https://www.hpc.cam.ac.uk/research-data-store), a networked Lustre filesystem run by RCS.

To do this, you’ll need to request storage by emailing [support@hpc.cam.ac.uk](mailto:support@hpc.cam.ac.uk) and explaining that you need storage on the [Research Data Store](https://docs.hpc.cam.ac.uk/storage/rds/).
The minimum storage space that can be allocated is 1 TiB.

If your request is accepted you’ll then need to accept the [terms and conditions](https://docs.hpc.cam.ac.uk/storage/terms.html).

## Examples

In this directory you’ll find the following examples:

1. [DDP Demo](./examples/ddp_multi_node): Intel’s example code, a DDP example based on Multi-GPU AI Training (Data-Parallel) with Intel&reg; Extension for PyTorch.
2. [Multi Node Accelerate](./examples/multi_node_accelerate): An example using PyTorch accelerate to distribute training across multiple nodes.
3. [Distributed lit-GPT](./examples/gpt_lightning): Running the min-GPT code over multiple nodes using PyTorch Lightning.

## Hints and tips

The [Hints and Tips](./hints-and-tips) directory contains a collection of useful information we've gathered while working with DAWN.

## Documentation

For more information about using DAWN once you have access, see the following docs.

DAWN documentation: https://docs.hpc.cam.ac.uk/hpc/user-guide/pvc.html

Storage docs: https://docs.hpc.cam.ac.uk/storage/rds/

You can email Cambridge HPC at: support@hpc.cam.ac.uk

