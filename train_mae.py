# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import os
import sys
import math
import time
import argparse
import datetime
import json
from pathlib import Path

import torch
print(torch.__version__)

import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, DistributedSampler

import timm
assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import models_mae

from typing import Iterable
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size_per_gpu', default=256, type=int, help='Batch size per GPU (effective batch size is batch_size_per_gpu * accum_iter * # gpus')
    parser.add_argument('--epochs', default=999, type=int)
    parser.add_argument('--accum_iter', default=1, type=int, help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument("--save_prefix", default="", type=str, help="""prefix for saving checkpoint and log files""")

    # Model parameters
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL', help='Name of model to train')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--input_size', default=224, type=int, help='images input size')
    parser.add_argument('--mask_ratio', default=0.8, type=float, help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--norm_pix_loss', action='store_true', help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='learning rate (absolute lr)')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR', help='lower lr bound for cyclic schedulers that hit 0')

    # Dataset parameters
    parser.add_argument('--data_path', default='/scratch/eo41/data/saycam/SAY_5fps_300s_{000000..000009}.tar', type=str, help='dataset path')
    parser.add_argument('--output_dir', default='./output_dir', help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', help='device to use for training/testing')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--pin_mem', action='store_true', help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    return parser


def main(args):
    misc.init_distributed_mode(args)
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)
    cudnn.benchmark = True

    # simple augmentation
    transform = transforms.Compose([
        transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3), 
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    dataset = ImageFolder(args.data_path, transform=transform)
    sampler = DistributedSampler(dataset, num_replicas=misc.get_world_size(), rank=misc.get_rank(), shuffle=True)
    data_loader = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size_per_gpu, num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True)

    # define the model
    model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)
    model.to(device)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
    model_without_ddp = model.module
    
    n_parameters = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)
    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
    optimizer.lr = args.lr  # override loaded lr
    
    print("Starting MAE training!")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        train_one_epoch(model, data_loader, optimizer, device, loss_scaler, epoch, args=args)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer, device: torch.device, loss_scaler, epoch, args=None):
    
    data_loader.sampler.set_epoch(epoch)
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    accum_iter = args.accum_iter

    optimizer.zero_grad()

    for it, (samples, _) in enumerate(data_loader):

        samples = samples.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            loss, _, _ = model(samples, mask_ratio=args.mask_ratio)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss = loss / accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(), update_grad=(it + 1) % accum_iter == 0)
        if (it + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

    # ============ writing logs + saving checkpoint ============
    save_dict = {
        'model': model.module.state_dict(),
        'optimizer': optimizer.state_dict(),
        'args': args,
        'epoch': epoch,
        'scaler': loss_scaler.state_dict(),
    }

    misc.save_on_master(save_dict, os.path.join(args.output_dir, args.save_prefix + '_checkpoint.pth'))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}

    if misc.is_main_process():
        with (Path(args.output_dir) / (args.save_prefix + "_log.txt")).open("a") as f:
            f.write(json.dumps(log_stats) + "\n")

    # start a fresh logger to wipe off old stats
    metric_logger = misc.MetricLogger(delimiter="  ")


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)