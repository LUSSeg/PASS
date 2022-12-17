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

import numpy as np
import jittor as jt
jt.flags.use_cuda = 1

from src.utils import (
    initialize_exp,
    restart_from_checkpoint,
    fix_random_seeds,
    AverageMeter,
    distributed_sinkhorn
)
from src.multicropdataset import MultiCropDataset
import src.resnet as resnet_models
from options import getOption

logger = getLogger()
parser = getOption()


def main():
    global args
    args = parser.parse_args()
    fix_random_seeds(args.seed)
    logger, training_stats = initialize_exp(args, "epoch", "loss")

    # build data
    train_dataset = MultiCropDataset(
        args.data_path,
        args.size_crops,
        args.nmb_crops,
        args.min_scale_crops,
        args.max_scale_crops,
    )
    train_loader = train_dataset.set_attrs(
        batch_size=args.batch_size,
        num_workers=args.workers,
        drop_last=True,
        shuffle=True
    )
    logger.info("Building data done with {} images loaded.".format(len(train_dataset)))

    # build model
    model = resnet_models.__dict__[args.arch](
        normalize=True,
        hidden_mlp=args.hidden_mlp,
        output_dim=args.feat_dim,
        nmb_prototypes=args.nmb_prototypes,
        train_mode='pixelattn'
    )
    if jt.in_mpi:
        for n, p in model.named_parameters():
            p.assign(p.mpi_broadcast())

    # for pixel attention, only finetuning the attention head and prototypes
    for name, param in model.named_parameters():
        if "fbg" not in name and "prototypes" not in name:
            param.requires_grad = False
    
    # loading pretrained weights
    checkpoint = jt.load(args.pretrained)["state_dict"]
    for k in list(checkpoint.keys()):
        if k not in model.state_dict().keys():
            del checkpoint[k]
    model.load_state_dict(checkpoint)
    logger.info("Loaded pretrained weights '{0}'".format(args.pretrained))

    # copy model to GPU
    if jt.rank == 0:
        logger.info(model)
    logger.info("Building model done.")

    # build optimizer
    optimizer = jt.optim.SGD(
        model.parameters(),
        lr=args.base_lr,
        momentum=0.9,
        weight_decay=args.wd,
    )
    warmup_lr_schedule = np.linspace(args.start_warmup, args.base_lr, len(train_loader) * args.warmup_epochs)
    iters = np.arange(len(train_loader) * (args.epochs - args.warmup_epochs))
    cosine_lr_schedule = np.array([args.final_lr + 0.5 * (args.base_lr - args.final_lr) * (1 + \
                         math.cos(math.pi * t / (len(train_loader) * (args.epochs - args.warmup_epochs)))) for t in iters])
    lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
    logger.info("Building optimizer done.")

    # optionally resume from a checkpoint
    to_restore = {"epoch": 0}
    restart_from_checkpoint(
        os.path.join(args.dump_path, "checkpoint.pth.tar"),
        run_variables=to_restore,
        state_dict=model,
        optimizer=optimizer,
    )
    start_epoch = to_restore["epoch"]

    # build the queue
    queue = None
    queue_path = os.path.join(args.dump_path, "queue" + str(jt.rank) + ".pth.tar")
    if os.path.isfile(queue_path):
        queue = jt.load(queue_path)["queue"]
    # the queue needs to be divisible by the batch size
    args.queue_length -= args.queue_length % (args.batch_size * jt.world_size)

    for epoch in range(start_epoch, args.epochs):

        # train the network for one epoch
        logger.info("============ Starting epoch %i ... ============" % epoch)

        # optionally starts a queue
        if args.queue_length > 0 and epoch >= args.epoch_queue_starts and queue is None:
            queue = jt.zeros(
                (
                    len(args.crops_for_assign),
                    args.queue_length // jt.world_size,
                    args.feat_dim
                )
            )

        # train the network
        scores, queue = train(train_loader, model, optimizer, epoch, lr_schedule, queue)
        training_stats.update(scores)

        # save checkpoints
        if jt.rank == 0:
            save_dict = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            jt.save(
                save_dict,
                os.path.join(args.dump_path, "checkpoint.pth.tar"),
            )
            if epoch % args.checkpoint_freq == 0 or epoch == args.epochs - 1:
                shutil.copyfile(
                    os.path.join(args.dump_path, "checkpoint.pth.tar"),
                    os.path.join(args.dump_checkpoints, "ckp-" + str(epoch) + ".pth.tar"),
                )
        if queue is not None:
            jt.save({"queue": queue}, queue_path)
        jt.sync_all()


def train(train_loader, model, optimizer, epoch, lr_schedule, queue):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()
    use_the_queue = False

    end = time.time()
    for it, inputs in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # update learning rate
        iteration = epoch * len(train_loader) + it
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_schedule[iteration]

        # normalize the prototypes
        with jt.no_grad():
            model.prototypes.weight.assign(
                model.prototypes.weight.normalize(dim=1, p=2))

        # ============ multi-res forward passes ... ============
        embedding, output = model(inputs)
        embedding = embedding.detach()
        bs = inputs[0].size(0)

        # ============ swav loss ... ============
        loss = 0
        for i, crop_id in enumerate(args.crops_for_assign):
            with jt.no_grad():
                out = output[bs * crop_id: bs * (crop_id + 1)].detach()

                # time to use the queue
                if queue is not None:
                    if use_the_queue or not jt.all(queue[i, -1, :] == 0):
                        use_the_queue = True
                        out = jt.concat((jt.matmul(
                            queue[i],
                            model.prototypes.weight.t()
                        ), out))
                    # fill the queue
                    queue[i, bs:] = queue[i, :-bs].clone()
                    queue[i, :bs] = embedding[crop_id * bs: (crop_id + 1) * bs]

                # get assignments
                q = distributed_sinkhorn(args, out)[-bs:]

            # cluster assignment prediction
            subloss = 0
            for v in np.delete(np.arange(np.sum(args.nmb_crops)), crop_id):
                x = output[bs * v: bs * (v + 1)] / args.temperature
                subloss -= jt.mean(jt.sum(q * jt.nn.log_softmax(x, dim=1), dim=1))
            loss += subloss / (np.sum(args.nmb_crops) - 1)
        loss /= len(args.crops_for_assign)

        # ============ backward and optim step ... ============
        for name, param in model.named_parameters():
            if "prototypes" in name:
                if iteration >= args.freeze_prototypes_niters:
                    param.start_grad()
                    assert not param.is_stop_grad()
                else:
                    param.stop_grad()
                    assert param.is_stop_grad()
        optimizer.step(loss)

        # ============ misc ... ============
        losses.update(loss.item(), inputs[0].size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if jt.rank == 0 and it % 50 == 0:
            logger.info(
                "Epoch: [{0}][{1}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Lr: {lr:.4f}".format(
                    epoch,
                    it,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    lr=optimizer.param_groups[0]["lr"],
                )
            )
    return (epoch, losses.avg), queue


if __name__ == "__main__":
    main()
