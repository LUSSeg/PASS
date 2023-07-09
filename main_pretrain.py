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
from src.multicropdataset import MultiCropDatasetGrid
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
    train_dataset = MultiCropDatasetGrid(
        args.data_path,
        args.size_crops,
        args.nmb_crops,
        args.min_scale_crops,
        args.max_scale_crops,
        grid_size=7
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
        train_mode='pretrain',
        shallow=args.shallow
    )
    if jt.in_mpi:
        for n, p in model.named_parameters():
            p.assign(p.mpi_broadcast())

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
        optimizer=optimizer
    )
    start_epoch = to_restore["epoch"]

    # build the queue
    queue = None
    queue_path = os.path.join(args.dump_path, "queue" + str(jt.rank) + ".pth.tar")
    if os.path.isfile(queue_path):
        queue = jt.load(queue_path)["queue"]
    # the queue needs to be divisible by the batch size
    args.queue_length -= args.queue_length % (args.batch_size * jt.world_size)
    if queue is not None and not isinstance(queue, jt.Var):
        queue = jt.Var(queue)
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
    losses_p2p = AverageMeter()
    losses_d2s = AverageMeter()

    model.train()
    use_the_queue = False

    end = time.time()
    for it, (inputs, gridq, gridk) in enumerate(train_loader):
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
        (
            embedding,
            output,
            embedding_deep_pixel,
            output_deep_pixels,
        ) = model(inputs, gridq=gridq, gridk=gridk)
        embedding = embedding.detach()
        bs = inputs[0].size(0)

        # ============ swav loss ... ============
        loss_i2i, labels, queue, use_the_queue = swav_loss(
            args, model, embedding, output, queue, use_the_queue, bs
        )
        loss_d2s = d2s_loss(
            args, 
            output[bs * np.sum(args.nmb_crops):], 
            labels, bs, shallow=args.shallow)
        loss_p2p = p2p_loss(
            args,
            output_deep_pixels,
            embedding_deep_pixel,
        )
        loss = jt.sum(
            jt.stack(
                [loss_i2i * args.weights[0]] + \
                [x * args.weights[1 + i] for i, x in enumerate(loss_d2s)], dim=0)) / sum(args.weights) + loss_p2p

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
        losses.update(loss_i2i.item(), inputs[0].size(0))
        losses_p2p.update(loss_p2p.item(), inputs[0].size(0))
        losses_d2s.update(jt.mean(jt.stack(loss_d2s, dim=0)).item(), inputs[0].size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if jt.rank == 0 and it % 50 == 0:
            logger.info(
                "Epoch: [{0}][{1}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "P2P {loss_p2p.val:.4f} ({loss_p2p.avg:.4f})\t"
                "D2S {loss_d2s.val:.4f} ({loss_d2s.avg:.4f})\t"
                "Lr: {lr:.4f}".format(
                    epoch,
                    it,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    loss_p2p=losses_p2p,
                    loss_d2s=losses_d2s,
                    lr=optimizer.param_groups[0]["lr"],
                )
            )
    return (epoch, losses.avg), queue


def swav_loss(args, model, embedding, output, queue, use_the_queue, bs):
    embedding = embedding.detach()
    labels = []
    # ============ swav loss ... ============
    loss = 0
    for i, crop_id in enumerate(args.crops_for_assign):
        with jt.no_grad():
            out = output[bs * crop_id : bs * (crop_id + 1)].detach()

            # time to use the queue
            if queue is not None:
                if use_the_queue or not jt.all(queue[i, -1, :] == 0):
                    use_the_queue = True
                    out = jt.concat(
                        (
                            jt.matmul(queue[i], model.prototypes.weight.t()),
                            out,
                        )
                    )
                # fill the queue
                queue[i, bs:] = queue[i, :-bs].clone()
                queue[i, :bs] = embedding[crop_id * bs : (crop_id + 1) * bs]

            # get assignments
            q = distributed_sinkhorn(args, out)[-bs:]
            labels.append(q)

        # cluster assignment prediction
        subloss = 0
        for v in np.delete(np.arange(np.sum(args.nmb_crops)), crop_id):
            x = output[bs * v : bs * (v + 1)] / args.temperature
            subloss -= jt.mean(jt.sum(q * jt.nn.log_softmax(x, dim=1), dim=1))
        loss += subloss / (np.sum(args.nmb_crops) - 1)
    loss /= len(args.crops_for_assign)
    return loss, labels, queue, use_the_queue


def d2s_loss(args, output, labels, bs, shallow=None):
    if shallow is None:
        return jt.float32([0])

    # alignment from deep to shallow
    losses_shallow = []
    if shallow is not None:
        for stage in shallow:
            loss_shallow = 0
            assert stage < 4, 'A shallow stage should be 1, 2 or 3.'
            for i, crop_id in enumerate(args.crops_for_assign):
                q = labels[i]

                # cluster assignment prediction
                subloss = 0
                for v in np.delete(
                    np.arange(2 * (3 - stage), 2 * (3 - stage) + 2),
                    crop_id,
                ):
                    x = output[bs * v : bs * (v + 1)] / args.temperature
                    subloss -= jt.mean(jt.sum(q * jt.nn.log_softmax(x, dim=1), dim=1))
                loss_shallow += subloss
            loss_shallow /= len(args.crops_for_assign)
            losses_shallow.append(loss_shallow)

    return losses_shallow


def p2p_loss(args, output_pixel, embedding_pixel):
    n, c, h, w = embedding_pixel.shape
    embedding_pixel = embedding_pixel.view(n, c, h * w).permute(0, 2, 1)
    output_pixel = output_pixel.view(n, c, h * w).permute(0, 2, 1)
    z1, z2 = embedding_pixel.split(n // 2, dim=0)
    p1, p2 = output_pixel.split(n // 2, dim=0)

    z1 = jt.normalize(z1, p=2, dim=-1) # [B HW C]
    z2 = jt.normalize(z2, p=2, dim=-1)
    p1 = jt.normalize(p1, p=2, dim=-1)
    p2 = jt.normalize(p2, p=2, dim=-1)
    loss = -((p1 * z2.detach()).sum(dim=-1).mean() + (p2 * z1.detach()).sum(dim=-1).mean()) * 0.5
    return loss


if __name__ == "__main__":
    main()
