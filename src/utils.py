# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import os
import pickle
from logging import getLogger
import cv2
import numpy as np
import math
import jittor as jt
import jittor.nn as nn
from munkres import Munkres
import warnings
import json
from .logger import PD_Stats, create_logger

FALSY_STRINGS = {'off', 'false', '0'}
TRUTHY_STRINGS = {'on', 'true', '1'}

logger = getLogger()


def bool_flag(s):
    """Parse boolean arguments from the command line."""
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError('invalid value for a boolean flag')


def initialize_exp(params, *args, dump_params=True):
    """Initialize the experience:

    - dump parameters
    - create checkpoint repo
    - create a logger
    - create a panda object to keep track of the training statistics
    """

    # dump parameters
    if dump_params:
        pickle.dump(params,
                    open(os.path.join(params.dump_path, 'params.pkl'), 'wb'))

    # create repo to store checkpoints
    params.dump_checkpoints = os.path.join(params.dump_path, 'checkpoints')
    if not params.rank and not os.path.isdir(params.dump_checkpoints):
        os.mkdir(params.dump_checkpoints)

    # create a panda object to log loss and acc
    training_stats = PD_Stats(
        os.path.join(params.dump_path, 'stats' + str(params.rank) + '.pkl'),
        args)

    # create a logger
    logger = create_logger(os.path.join(params.dump_path, 'train.log'),
                           rank=params.rank)
    logger.info('============ Initialized logger ============')
    logger.info('\n'.join(f'{k}: {str(v)}'
                          for k, v in sorted(dict(vars(params)).items())))
    logger.info('The experiment will be stored in %s\n' % params.dump_path)
    logger.info('')
    return logger, training_stats


def restart_from_checkpoint(ckp_paths, run_variables=None, **kwargs):
    """Re-start from checkpoint."""
    # look for a checkpoint in exp repository
    if isinstance(ckp_paths, list):
        for ckp_path in ckp_paths:
            if os.path.isfile(ckp_path):
                break
    else:
        ckp_path = ckp_paths

    if not os.path.isfile(ckp_path):
        return

    logger.info(f'Found checkpoint at {ckp_path}')

    # open checkpoint file
    checkpoint = jt.load(ckp_path)

    # key is what to look for in the checkpoint file
    # value is the object to load
    # example: {'state_dict': model}
    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                msg = value.load_state_dict(checkpoint[key], strict=False)
                print(msg)
            except TypeError:
                msg = value.load_state_dict(checkpoint[key])
            logger.info("=> loaded {} from checkpoint '{}'".format(
                key, ckp_path))
        else:
            logger.warning("=> failed to load {} from checkpoint '{}'".format(
                key, ckp_path))

    # re load variable important for the run
    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]


def fix_random_seeds(seed=31):
    """Fix random seeds."""
    jt.set_global_seed(seed)


class AverageMeter:
    """computes and stores the average and current value."""
    def __init__(self):
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


def accuracy(output:jt.Var, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified
    values of k."""
    with jt.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = jt.misc.topk(output, maxk, 1, True, True)
        pred = pred.t()
        correct = jt.equal(pred, target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(
                0, keepdims=True)
            res.append(correct_k * (100.0 / batch_size))
        return res


def distributed_sinkhorn(args, out):
    with jt.no_grad():
        Q = jt.exp(out / args.epsilon).t(
        )  # Q is K-by-B for consistency with notations from our paper
        B = Q.shape[1] * jt.world_size  # number of samples to assign
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = jt.sum(Q)
        if jt.in_mpi:
            sum_Q = sum_Q.mpi_all_reduce('add')
        Q /= sum_Q

        for it in range(args.sinkhorn_iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = jt.sum(Q, dim=1, keepdims=True)
            if jt.in_mpi:
                sum_of_rows = sum_of_rows.mpi_all_reduce('add')
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= jt.sum(Q, dim=0, keepdims=True)
            Q /= B

        Q *= B  # the columns must sum to 1 so that Q is an assignment
        return Q.t()


def hungarian(target, prediction, num_classes=50):
    total = len(target)
    matrix = np.zeros(shape=(num_classes, num_classes), dtype=np.float32)
    preds = {}
    gts = {}
    for i in range(num_classes):
        preds[i] = np.array([i in item for item in prediction])
        gts[i] = np.array([i in item for item in target])

    for i in range(num_classes):
        for j in range(num_classes):
            coi = np.logical_and(preds[i], gts[j])
            matrix[i][j] = - np.sum(coi)

    np.set_printoptions(threshold=np.inf)
    matrix = matrix.tolist()
    m = Munkres()
    indexes = m.compute(matrix)
    total = 0
    matched = {}
    for row, column in indexes:
        total += matrix[row][column]
        matched[row] = column

    return num_classes - total, matched


def get_mask_of_class(mask, v):
    """
    Get binary mask of v-th class.
    :param mask (numpy array, uint8): semantic segmentation mask
    :param v (int): the index of given class
    :return: binary mask of v-th class
    """
    mask_v = (mask == v) * 255
    return mask_v.astype(np.uint8)


def mask_to_boundary(mask, dilation_ratio=0.02):
    """
    https://github.com/bowenc0221/boundary-iou-api/blob/master/boundary_iou/utils/boundary_utils.py
    Convert binary mask to boundary mask.
    :param mask (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
    # Pad image so mask truncated by the image border is also considered as boundary.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
    # G_d intersects G in the paper.
    return mask - mask_erode


def _no_grad_trunc_normal_(tensor: jt.Var, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    requires_grad = tensor.requires_grad
    with jt.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        jt.init.uniform_(tensor, 2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.assign(jt.erfinv(tensor))

        # Transform to proper mean, std
        tensor.assign(tensor * std * math.sqrt(2.) + mean)

        # Clamp to ensure it's in the proper range
        tensor.assign(jt.clamp(tensor, min_v=a, max_v=b))
    tensor.requires_grad = requires_grad
    return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def has_batchnorms(model):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
    for name, module in model.named_modules():
        if isinstance(module, bn_types):
            return True
    return False


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def clip_gradients(model: jt.nn.Module, optimizer: jt.nn.Optimizer, clip):
    norms = []
    for pg in optimizer.param_groups:
        for p, g in zip(pg["params"], pg["grads"]):
            if p.is_stop_grad(): 
                continue
            param_norm = jt.norm(g.flatten(), p=2)
            norms.append(param_norm.item())
            clip_coef = clip / (param_norm + 1e-6)
            if clip_coef < 1:
                g.update(g * clip_coef)
