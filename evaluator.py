import argparse
import os
from multiprocessing import Manager, Process

import numpy as np
import torch
from tqdm import tqdm

from src.singlecropdataset import EvalDataset
from src.utils import hungarian

NUM_THREADS = 24
manager = Manager()


def func_hungarian(i, targets, predictions, thresholds, matches, num_classes):
    for j in range(i, len(thresholds), NUM_THREADS):
        _, match = hungarian(targets,
                             predictions[thresholds[j]],
                             num_classes=num_classes)
        match[1000] = 1000
        matches[thresholds[j]] = match


def match(loader, num_classes, threshold):
    if isinstance(threshold, float):
        threshold = [threshold]

    predictions = {t: [] for t in threshold}
    targets = []
    for target, _, predict, _, logit in tqdm(loader):

        target = target.cuda()
        predict = predict.cuda()

        target = torch.unique(target.view(-1))
        target = target - 1
        target = target.tolist()
        if -1 in target:
            target.remove(-1)
        targets.append(target)
        logit = logit.cuda()
        for t in threshold:
            predict_ = predict.clone()
            predict_[logit < t] = 0

            predict_ = torch.unique(predict_.view(-1))
            predict_ = predict_ - 1
            predict_ = predict_.tolist()
            if -1 in predict_:
                predict_.remove(-1)
            predictions[t].append(predict_)

    # multi-thread matching
    match = manager.dict()
    p_list = []
    for i in range(min(NUM_THREADS, len(threshold))):
        t = threshold[i]
        p = Process(target=func_hungarian,
                    args=(i, targets, predictions, threshold, match,
                          num_classes))
        p.start()
        p_list.append(p)
    for p in p_list:
        p.join()

    return match


def intersectionAndUnionGPU(output, target, K):
    # 'K' classes,
    # output and target sizes are N or N * L or N * H * W,
    # each value in range 0 to K - 1.
    assert output.dim() in [1, 2, 3]
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K - 1)
    area_output = torch.histc(output, bins=K, min=0, max=K - 1)
    area_target = torch.histc(target, bins=K, min=0, max=K - 1)
    return area_intersection, area_output, area_target


def fscore(predict, target):
    target[target > 0] = 1
    predict[predict > 0] = 1

    t = torch.sum(target)
    p = torch.sum(predict)
    tp = torch.sum(target * predict).float()
    recall = tp / (t + 1e-20)
    precision = tp / (p + 1e-20)
    f_score = (1 + 0.3) * precision * recall / (0.3 * precision + recall +
                                                1e-20)

    return f_score


def evaluator(loader, num_classes, thresholds, matches):
    assert thresholds is not None
    if isinstance(thresholds, float):
        thresholds = [thresholds]

    logs = []
    for t in thresholds:
        T = torch.zeros(size=(num_classes + 1, )).cuda()
        P = torch.zeros(size=(num_classes + 1, )).cuda()
        TP = torch.zeros(size=(num_classes + 1, )).cuda()
        BT = torch.zeros(size=(num_classes + 1,)).cuda()
        BP = torch.zeros(size=(num_classes + 1,)).cuda()
        BTP = torch.zeros(size=(num_classes + 1,)).cuda()
        IoU = torch.zeros(size=(num_classes + 1, )).cuda()
        FMeasure = 0.0
        ACC = 0.0

        loader.dataset.threshold = t
        loader.dataset.match = matches[t]
        for target, boundary_target, predict, boundary_predict, _ in tqdm(loader):
            target = target.cuda()
            predict = predict.cuda()
            boundary_target = boundary_target.cuda()
            boundary_predict = boundary_predict.cuda()

            area_intersection, area_output, area_target = \
                intersectionAndUnionGPU(
                    predict.view(-1), target.view(-1), num_classes + 1)
            
            area_intersection_boundary, area_output_boundary, area_target_boundary = \
                intersectionAndUnionGPU(
                    boundary_predict.view(-1), boundary_target.view(-1), num_classes + 2)

            f_score = fscore(predict, target)

            T += area_output
            P += area_target
            TP += area_intersection
            BT += area_output_boundary[1:]
            BP += area_target_boundary[1:]
            BTP += area_intersection_boundary[1:]
            FMeasure += f_score

            img_label = torch.argmax(area_output[1:]) + 1
            ACC += (area_target[img_label] > 0) * (area_output[img_label] > 0)

        IoU = TP / (T + P - TP + 1e-10)
        BIoU = BTP / (BT + BP - BTP + 1e-10)
        mIoU = torch.mean(IoU).item() * 100
        mBIoU = torch.mean(BIoU).item() * 100
        FMeasure = FMeasure.item() / len(loader.dataset)
        ACC = ACC.item() * 100 / len(loader.dataset)

        print('Threshold: {:.2f}\tAcc: {:.4f}\tmIoU: {:.4f}\tmBIoU: {:.4f}\tFMeasure: {:.4f}'.
              format(t, ACC, mIoU, mBIoU, FMeasure))

        log = dict(th=t,
                   match=matches[t],
                   Acc=ACC,
                   mIoU=mIoU,
                   mBIoU=mBIoU,
                   IoUs=IoU * 100,
                   FMeasure=FMeasure)
        logs.append(log)

    mious = [log['mIoU'] for log in logs]
    best = np.argmax(mious)
    log = logs[best]
    return log


def evaludation(args, mode):

    # dataloader
    gt_path = os.path.join(args.data_path, f'{mode}-segmentation')
    predict_path = os.path.join(args.predict_path, mode)
    dataset = EvalDataset(predict_path, gt_path)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
        drop_last=False,
    )

    if args.curve:
        thresholds = [
            threshold / 100.0 for threshold in range(args.min, args.max + 1)
        ]
    else:
        thresholds = [args.t / 100.0]
    matches = match(loader, args.num_classes, threshold=thresholds)
    log = evaluator(
        loader,
        num_classes=args.num_classes,
        thresholds=thresholds,
        matches=matches,
    )
    return log


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--predict_path',
                        default=None,
                        type=str,
                        help='The path to the predictions.')
    parser.add_argument('--data_path',
                        default=None,
                        type=str,
                        help='The path to ImagenetS dataset')
    parser.add_argument('--mode',
                        type=str,
                        default='validation',
                        choices=['validation', 'test'],
                        help='Evaluating on the validation or test set.')
    parser.add_argument('--workers', default=32, type=int)
    parser.add_argument('--t',
                        default=0,
                        type=float,
                        help='The used threshold when curve is disabled.')
    parser.add_argument('--min',
                        default=0,
                        type=int,
                        help='The minimum threshold when curve is enabled.')
    parser.add_argument('--max',
                        default=60,
                        type=int,
                        help='The maximum threshold when curve is enabled.')
    parser.add_argument('-c',
                        '--num_classes',
                        type=int,
                        default=50,
                        help='The number of classes.')
    parser.add_argument('--curve',
                        action='store_true',
                        help='Whether to try different thresholds.')
    args = parser.parse_args()

    log = evaludation(args, args.mode)
