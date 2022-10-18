# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import math
import os
import shutil
import time
from logging import getLogger

import apex
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torchvision.transforms as transforms
from apex.parallel.LARC import LARC

import src.pseudo_transforms as custom_transforms
import src.resnet as resnet_models
from options import getOption
from src.singlecropdataset import PseudoLabelDataset
from src.utils import (AverageMeter, accuracy, fix_random_seeds,
                       init_distributed_mode, initialize_exp,
                       restart_from_checkpoint)

logger = getLogger()
parser = getOption()


def main():
    global args
    args = parser.parse_args()
    init_distributed_mode(args)
    fix_random_seeds(args.seed)
    logger, training_stats = initialize_exp(args, 'epoch', 'loss')

    # build data
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dataset = PseudoLabelDataset(
        args.data_path,
        custom_transforms.Compose([
            custom_transforms.RandomResizedCropSemantic(224),
            custom_transforms.RandomHorizontalFlipSemantic(),
            custom_transforms.ToTensorSemantic(),
            normalize,
        ]),
        pseudo_path=args.pseudo_path,
    )
    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               sampler=sampler,
                                               batch_size=args.batch_size,
                                               num_workers=args.workers,
                                               pin_memory=True,
                                               drop_last=True)
    logger.info('Building data done with {} images loaded.'.format(
        len(train_dataset)))

    # build model
    model = resnet_models.__dict__[args.arch](hidden_mlp=0,
                                              output_dim=0,
                                              nmb_prototypes=0,
                                              num_classes=args.num_classes,
                                              train_mode='finetune')
    # loading pretrained weights
    checkpoint = torch.load(args.pretrained,
                            map_location='cpu')['state_dict']
    for k in list(checkpoint.keys()):
        if k.startswith('module.'):
            checkpoint[k[len('module.'):]] = checkpoint[k]
        del checkpoint[k]
    msg = model.load_state_dict(checkpoint, strict=False)
    logger.info("Loaded pretrained weights '{}' with missing {}".format(
        args.pretrained, msg.missing_keys))

    # synchronize batch norm layers
    if args.sync_bn == 'pytorch':
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    elif args.sync_bn == 'apex':
        # with apex syncbn we sync bn per group
        # because it speeds up computation
        # compared to global syncbn
        process_group = apex.parallel.create_syncbn_process_group(
            args.syncbn_process_group_size)
        model = apex.parallel.convert_syncbn_model(model,
                                                   process_group=process_group)
    # copy model to GPU
    model = model.cuda()
    if args.rank == 0:
        logger.info(model)
    logger.info('Building model done.')

    # build optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.base_lr,
        momentum=0.9,
        weight_decay=args.wd,
    )
    optimizer = LARC(optimizer=optimizer, trust_coefficient=0.001, clip=False)
    warmup_lr_schedule = np.linspace(args.start_warmup, args.base_lr,
                                     len(train_loader) * args.warmup_epochs)
    iters = np.arange(len(train_loader) * (args.epochs - args.warmup_epochs))
    cosine_lr_schedule = \
        np.array([args.final_lr + 0.5 * (args.base_lr - args.final_lr) * (
            1 +
            math.cos(
                math.pi * t /
                (len(train_loader) * (args.epochs - args.warmup_epochs))
            )
        )
                for t in iters])
    lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
    logger.info('Building optimizer done.')

    # init mixed precision
    if args.use_fp16:
        model, optimizer = apex.amp.initialize(model,
                                               optimizer,
                                               opt_level='O1')
        logger.info('Initializing mixed precision done.')

    # wrap model
    model = nn.parallel.DistributedDataParallel(
        model, device_ids=[args.gpu_to_work_on])

    # optionally resume from a checkpoint
    to_restore = {'epoch': 0}
    restart_from_checkpoint(
        os.path.join(args.dump_path, 'checkpoint.pth.tar'),
        run_variables=to_restore,
        state_dict=model,
        optimizer=optimizer,
        amp=apex.amp,
    )
    start_epoch = to_restore['epoch']

    # loss function
    criterion = nn.CrossEntropyLoss()

    cudnn.benchmark = True

    for epoch in range(start_epoch, args.epochs):

        # train the network for one epoch
        logger.info('============ Starting epoch %i ... ============' % epoch)

        # set sampler
        train_loader.sampler.set_epoch(epoch)

        # train the network
        scores = train(train_loader, model, optimizer, criterion, epoch,
                       lr_schedule)
        training_stats.update(scores)

        # save checkpoints
        if args.rank == 0:
            save_dict = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            if args.use_fp16:
                save_dict['amp'] = apex.amp.state_dict()
            torch.save(
                save_dict,
                os.path.join(args.dump_path, 'checkpoint.pth.tar'),
            )
            if epoch % args.checkpoint_freq == 0 or epoch == args.epochs - 1:
                shutil.copyfile(
                    os.path.join(args.dump_path, 'checkpoint.pth.tar'),
                    os.path.join(args.dump_checkpoints,
                                 'ckp-' + str(epoch) + '.pth'),
                )


def train(train_loader, model, optimizer, criterion, epoch, lr_schedule):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    model.train()

    end = time.time()
    for it, (inputs, labels) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # update learning rate
        iteration = epoch * len(train_loader) + it
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_schedule[iteration]

        # ============ forward step ... ============
        inputs = inputs.cuda()
        labels = labels.cuda()
        labels = labels[:, 1, :, :] * 256 + labels[:, 0, :, :]
        labels = labels.long()

        output = model(inputs)
        labels = F.interpolate(labels.float().unsqueeze(1),
                               scale_factor=0.5,
                               mode='nearest').long().squeeze(1)
        output = F.interpolate(output,
                               size=(labels.shape[1], labels.shape[2]),
                               mode='bilinear')
        c = output.shape[1]
        loss = criterion(output, labels)
        (acc1, ) = accuracy(
            output.permute(0, 2, 3, 1).contiguous().view(-1, c),
            labels.view(-1))

        # ============ backward and optim step ... ============
        optimizer.zero_grad()
        if args.use_fp16:
            with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        # ============ misc ... ============
        losses.update(loss.item(), inputs[0].size(0))
        acc.update(acc1.item(), inputs[0].size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if args.rank == 0 and it % 50 == 0:
            logger.info('Epoch: [{0}][{1}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Acc@1 {acc1.val:.2f} ({acc1.avg:.2f})\t'
                        'Lr: {lr:.4f}'.format(
                            epoch,
                            it,
                            batch_time=batch_time,
                            data_time=data_time,
                            loss=losses,
                            lr=optimizer.optim.param_groups[0]['lr'],
                            acc1=acc,
                        ))
    return (epoch, losses.avg)


if __name__ == '__main__':
    main()
