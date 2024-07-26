# DAWN

Dawn is a new supercomputing resource hosted by the Cambridge Open Zettascale Lab and run by the University of Cambridge Research Computing Service (RCS).
It’s been developed as part of the Artificial Intelligence Research Resource (AIRR) project and is billed as the UK’s [fastest artificial intelligence supercomputer](https://www.hpc.cam.ac.uk/d-w-n).

With JADE2 ramping down during the Autumn of 2024 and Baskerville due to do the same the Spring after, research at the Turing will be gradually transitioning to the new AIRR systems: DAWN, Isambard-AI and the upcoming AIRR+ services.

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

# Getting access to DAWN

Currently DAWN is still in its early access period.
Eventually it’s expected that access will be delegated to UKRI, but for now, Turing researchers can request access on an individual basis and with the understanding that access may have to be re negotiated (meaning that it could be withdrawn) in the future.

In the first instance we request Turing researchers wanting access to DAWN to get in contact with the RCP Team either by submitting the [Request Allocation](https://turingcomplete.topdesk.net/tas/public/ssp/content/serviceflow?unid=ac51b39d8bfc46f9bf41132ef8601b5e) form on [Turing Complete](https://turingcomplete.topdesk.net/) (Research Services > Research Computing Platforms > Request Allocation) or by getting in touch by [email](mailto:ResearchComputePlatforms@turing.ac.uk) to discuss your requrements.
While we do currently have researchers using DAWN, we don’t have any general access agreement, so the RCP team will need to approach the RCS team individually and with a clear justification and request for access.
We’ll therefore need to go through this with you in some detail to understand your needs and to assess whether DAWN would be the most appropriate service for your needs.

Having said all that, we are keen to get researchers using DAWN, so if you’re interested, please do get in touch.

## Accessing compute

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
3. [Distributed lit-GPT](/examples/lit-gpt): Running the min-GPT code over multiple nodes using PyTorch Lightning.

## Documentation

For more information about using DAWN once you have access, see the following docs.

DAWN info on Mathison: https://mathison.turing.ac.uk/page/3297

DAWN documentation: https://docs.hpc.cam.ac.uk/hpc/user-guide/pvc.html

Storage docs: https://docs.hpc.cam.ac.uk/storage/rds/

Support: support@hpc.cam.ac.uk

