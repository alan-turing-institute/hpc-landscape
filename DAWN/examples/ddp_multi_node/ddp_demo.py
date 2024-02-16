import argparse
import os
import sys
import time
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim
from torch.optim.lr_scheduler import StepLR

import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import intel_extension_for_pytorch as ipex
import oneccl_bindings_for_pytorch


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='127.0.0.1', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-port', default='29500', type=str,
                    help='url port used to set up distributed training')
parser.add_argument('--xpu', default=None, type=int,
                    help='XPU id to use.')

parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

global_num_iter = 0
os.environ["CCL_ZE_IPC_EXCHANGE"]="sockets"

def main():
    args = parser.parse_args()
    args.epochs = 1
    args.batch_size = 128
    args.workers = 4

    if args.world_size == -1:
        mpi_world_size = int(os.environ.get('PMI_SIZE', -1))

        if mpi_world_size > 0:
            os.environ['MASTER_ADDR'] = args.dist_url #'127.0.0.1'
            os.environ['MASTER_PORT'] = args.dist_port #'29500'
            os.environ['RANK'] = os.environ.get('PMI_RANK', -1)
            os.environ['WORLD_SIZE'] = os.environ.get('PMI_SIZE', -1)
            args.rank = int(os.environ.get('PMI_RANK', -1))
        args.world_size = int(os.environ.get("WORLD_SIZE", -1))
    else: # mpich set
        if 'PMIX_RANK' in os.environ.keys(): # mpich set
            os.environ['MASTER_ADDR'] = args.dist_url #'127.0.0.1'
            os.environ['MASTER_PORT'] = args.dist_port #'29500'
            os.environ['RANK'] = os.environ.get('PMIX_RANK')
            os.environ['WORLD_SIZE'] = str(args.world_size)
            args.rank = int(os.environ.get('PMIX_RANK', -1))

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    # 1 XPU card has 2 tile, and both are regarded as isolated devices/nodes
    ngpus_per_node = 1
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(ngpus_per_node, args)


def main_worker(ngpus_per_node, args):
    # rank, local_rank setup
    if args.distributed:
        if args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + args.xpu
        init_method = 'file:///rds/user/kk562/hpc-work/ddp-parallel/sync_file'
        dist.init_process_group(backend='ccl', init_method=init_method,
                                world_size=args.world_size, rank=args.rank)
    
    local_rank = args.xpu
    print('world_size:{}, rank:{}, local_rank:{}'.format(args.world_size, args.rank, local_rank))
    print("Using XPU: {}".format(args.xpu))
    args.xpu = "xpu:{}".format(args.xpu)
    
    # model, optimizer, schuduler, criterion initialization
    model = models.resnet18()
    torch.xpu.set_device(args.xpu)
    model = model.to(args.xpu)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    model.train()
    print('doing torch xpu optimize for training')
    model, optimizer = torch.xpu.optimize(model=model, optimizer=optimizer, level="O1")
    criterion = nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer=optimizer, step_size=30, gamma=0.1)

    if args.distributed:
        print("Generating DDP model for {}".format(args.xpu))
        print('---> Calling xpu.set_device', args.xpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.xpu])
        print('---> Done')

    # Using dummy data
    print("Dummy data is used!")
    train_dataset = datasets.FakeData(1281167, (3, 224, 224), 1000, transforms.ToTensor())

    train_sampler = None
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, pin_memory_device="xpu", sampler=train_sampler)
  
    # only 1 epoch for demo
    for epoch in range(args.epochs):
        epoch_start_time = time.time()

        if not args.distributed or (args.distributed and args.rank == 0):
            print('[info] Epoch[', epoch, '] start time = ', time.asctime(time.localtime(epoch_start_time)), flush=True)

        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, mode='training')

        # show time cost info
        if not args.distributed or (args.distributed and args.rank == 0):
            epoch_end_time = time.time()
            print('[info] Epoch[', epoch, '] end time = ', time.asctime(time.localtime(epoch_end_time)))
            print('[info] Epoch[', epoch, '] consume time = ', ((epoch_end_time - epoch_start_time) / 3600.0), ' hours')

    print("DDP demo for rank {} completed!".format(args.rank))

def train(train_loader, model, criterion, optimizer, epoch, args, mode='training'):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))
    global global_num_iter

    # switch to train mode
    model.train()

    # record time
    duration_total = 0.0

    data_start = time.time()
    for i, (images, target) in enumerate(train_loader):
        global_num_iter +=1
        # measure data loading time
        data_time.update(time.time() - data_start)

        start_time = time.time()
        images = images.to(args.xpu, non_blocking=True)
        target = target.to(args.xpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        torch.xpu.synchronize()
        # measure elapsed time
        duration_train = time.time() - start_time
        batch_time.update(duration_train)

        # record loss
        losses.update(loss.item(), images.size(0))
        
        if i % 10 == 0:
            progress.display(i + 1)

        # exclude first iteration for calculating througput
        if i >= 3:
            duration_total += duration_train
        data_start = time.time()


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        total = torch.FloatTensor([self.sum, self.count])
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


if __name__ == '__main__':
    main()
