import math
import numpy as np


def intersectionAndUnionGPU(output, target, K):
    # 'K' classes,
    # output and target sizes are N or N * L or N * H * W,
    # each value in range 0 to K - 1.
    assert output.ndim in [1, 2, 3]
    assert output.shape == target.shape
    output = output.reshape(-1)
    target = target.reshape(-1)
    intersection = output[output == target]
    area_intersection = np.histogram(intersection, bins=K, range=(0, K - 1))[0]
    area_output = np.histogram(output, bins=K, range=(0, K - 1))[0]
    area_target = np.histogram(target, bins=K, range=(0, K - 1))[0]
    return area_intersection, area_output, area_target


def fscore(predict, target):
    target[target > 0] = 1
    predict[predict > 0] = 1

    t = np.sum(target)
    p = np.sum(predict)
    tp = np.sum(target * predict).astype(np.float32)
    recall = tp / (t + 1e-20)
    precision = tp / (p + 1e-20)
    f_score = (1 + 0.3) * precision * recall / (0.3 * precision + recall +
                                                1e-20)

    return f_score


def IoUDifferentSizeGPUWithBoundary(predict, gt, boundary_predict, boundary_gt, K, Ts, Ps, TPs, BTs, BPs, BTPs):
    """
    Not all categories have objects from all sizes.
    When a category has no object with a certain size, the IoU of this category under the object size is set to zero.
    We give the upper bound of mIoU under different object sizes as follows:
    ----------------------------------------------------------------------------------------
    |                |           validation             |test                              |
    |                |   S.   |  M.S.  |  M.L.  |   L   |   S.   |  M.S.  |  M.L.  |   L   |
    | ImageNet-S-50  |  58.8  |  96.1  |  94.1  |  82.4 |  78.4  |  98.0  |  100.0 | 86.3  |
    | ImageNet-S-300 |  54.5  |  96.7  |  93.4  |  70.1 |  71.1  |  99.3  |  97.7  | 78.4  |
    | ImageNet-S-919 |  57.9  |  96.5  |  92.9  |  71.3 |  75.0  |  99.5  |  97.6  | 80.4  |
    ----------------------------------------------------------------------------------------
    """
    s5, s25, s50, s100 = [], [], [], []
    gts = {}
    predicts = {}
    boundary_gts = {}
    boundary_predicts = {}
    bg_gt = gt[gt == 0]
    bg_predict = predict[gt == 0]
    bg_boundary_gt = boundary_gt[gt == 0]
    bg_boundary_predict = boundary_predict[gt == 0]

    counts = np.bincount(gt.reshape(-1).astype(np.int32))
    size = {}
    for c in np.unique(gt.reshape(-1)):
        if c == 0:
            continue
        size[int(c)] = math.ceil(100.0 * counts[int(c)] / sum(counts))

    for k, v in size.items():
        if v <= 5:
            s5.append(k)
        elif v <= 25:
            s25.append(k)
        elif v <= 50:
            s50.append(k)
        else:
            s100.append(k)

        gts[k] = gt[gt == k]
        predicts[k] = predict[gt == k]
        boundary_gts[k] = boundary_gt[gt == k]
        boundary_predicts[k] = boundary_predict[gt == k]

    # different_size
    if len(s5) > 0:
        gt5 = np.concatenate([bg_gt] + [gts[k] for k in s5], axis=0)
        predict5 = np.concatenate([bg_predict] + [predicts[k] for k in s5], axis=0)
        area_intersection, area_output, area_target = intersectionAndUnionGPU(predict5, gt5, K)

        Ts[0] += area_output
        Ps[0] += area_target
        TPs[0] += area_intersection

        # boundary
        gt5 = np.concatenate([bg_boundary_gt] + [boundary_gts[k] for k in s5], axis=0)
        predict5 = np.concatenate([bg_boundary_predict] + [boundary_predicts[k] for k in s5], axis=0)
        area_intersection, area_output, area_target = intersectionAndUnionGPU(predict5, gt5, K + 1)

        BTs[0] += area_output[1:]
        BPs[0] += area_target[1:]
        BTPs[0] += area_intersection[1:]

    if len(s25) > 0:
        gt25 = np.concatenate([bg_gt] + [gts[k] for k in s25], axis=0)
        predict25 = np.concatenate([bg_predict] + [predicts[k] for k in s25], axis=0)
        area_intersection, area_output, area_target = intersectionAndUnionGPU(predict25, gt25, K)

        Ts[1] += area_output
        Ps[1] += area_target
        TPs[1] += area_intersection

        # boundary
        gt25 = np.concatenate([bg_boundary_gt] + [boundary_gts[k] for k in s25], axis=0)
        predict25 = np.concatenate([bg_boundary_predict] + [boundary_predicts[k] for k in s25], axis=0)
        area_intersection, area_output, area_target = intersectionAndUnionGPU(predict25, gt25, K + 1)

        BTs[1] += area_output[1:]
        BPs[1] += area_target[1:]
        BTPs[1] += area_intersection[1:]

    if len(s50) > 0:
        gt50 = np.concatenate([bg_gt] + [gts[k] for k in s50], axis=0)
        predict50 = np.concatenate([bg_predict] + [predicts[k] for k in s50], axis=0)
        area_intersection, area_output, area_target = intersectionAndUnionGPU(predict50, gt50, K)

        Ts[2] += area_output
        Ps[2] += area_target
        TPs[2] += area_intersection

        # boundary
        gt50 = np.concatenate([bg_boundary_gt] + [boundary_gts[k] for k in s50], axis=0)
        predict50 = np.concatenate([bg_boundary_predict] + [boundary_predicts[k] for k in s50], axis=0)
        area_intersection, area_output, area_target = intersectionAndUnionGPU(predict50, gt50, K + 1)

        BTs[2] += area_output[1:]
        BPs[2] += area_target[1:]
        BTPs[2] += area_intersection[1:]

    if len(s100) > 0:
        gt100 = np.concatenate([bg_gt] + [gts[k] for k in s100], axis=0)
        predict100 = np.concatenate([bg_predict] + [predicts[k] for k in s100], axis=0)
        area_intersection, area_output, area_target = intersectionAndUnionGPU(predict100, gt100, K)

        Ts[3] += area_output
        Ps[3] += area_target
        TPs[3] += area_intersection

        # boundary
        gt100 = np.concatenate([bg_boundary_gt] + [boundary_gts[k] for k in s100], axis=0)
        predict100 = np.concatenate([bg_boundary_predict] + [boundary_predicts[k] for k in s100], axis=0)
        area_intersection, area_output, area_target = intersectionAndUnionGPU(predict100, gt100, K + 1)

        BTs[3] += area_output[1:]
        BPs[3] += area_target[1:]
        BTPs[3] += area_intersection[1:]
