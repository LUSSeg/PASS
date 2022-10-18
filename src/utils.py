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
import torch
import torch.distributed as dist
from munkres import Munkres

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


def init_distributed_mode(args):
    """Initialize the following variables:

    - world_size
    - rank
    """

    args.is_slurm_job = 'SLURM_JOB_ID' in os.environ

    if args.is_slurm_job:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.world_size = int(os.environ['SLURM_NNODES']) * int(
            os.environ['SLURM_TASKS_PER_NODE'][0])
    else:
        # multi-GPU job (local or multi-node)
        # jobs started with torch.distributed.launch
        # read environment variables
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])

    # prepare distributed
    dist.init_process_group(
        backend='nccl',
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

    # set cuda device
    args.gpu_to_work_on = args.rank % torch.cuda.device_count()
    torch.cuda.set_device(args.gpu_to_work_on)
    return


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
    checkpoint = torch.load(
        ckp_path,
        map_location='cuda:' +
        str(torch.distributed.get_rank() % torch.cuda.device_count()))

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
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


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


def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified
    values of k."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(
                0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


@torch.no_grad()
def distributed_sinkhorn(args, out):
    Q = torch.exp(out / args.epsilon).t(
    )  # Q is K-by-B for consistency with notations from our paper
    B = Q.shape[1] * args.world_size  # number of samples to assign
    K = Q.shape[0]  # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    dist.all_reduce(sum_Q)
    Q /= sum_Q

    for it in range(args.sinkhorn_iterations):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        dist.all_reduce(sum_of_rows)
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B  # the columns must sum to 1 so that Q is an assignment
    return Q.t()


def hungarian(target, prediction, num_classes=50):
    total = len(target)
    matrix = np.zeros(shape=(num_classes, num_classes), dtype=np.float)
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