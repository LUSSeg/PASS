import argparse
import os
from multiprocessing import Manager, Process

import numpy as np
from tqdm import tqdm

from src.singlecropdataset import EvalDataset
from src.utils import hungarian
from src.metric import intersectionAndUnionGPU, fscore, IoUDifferentSizeGPUWithBoundary

NUM_THREADS = 24
manager = Manager()


def get_dataset(args, mode, threshold=None, match=None):
    gt_path = os.path.join(args.data_path, f'{mode}-segmentation')
    predict_path = os.path.join(args.predict_path, mode)
    dataset = EvalDataset(predict_path, gt_path, threshold=threshold, match=match)
    dataset.set_attrs(
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        drop_last=False,
    )
    return dataset


def func_hungarian(i, targets, predictions, thresholds, matches, num_classes):
    for j in range(i, len(thresholds), NUM_THREADS):
        _, match = hungarian(targets,
                             predictions[thresholds[j]],
                             num_classes=num_classes)
        match[1000] = 1000
        matches[thresholds[j]] = match


def match(args, mode, num_classes, threshold):
    loader = get_dataset(args, mode)
    if isinstance(threshold, float):
        threshold = [threshold]

    predictions = {t: [] for t in threshold}
    targets = []
    for target, _, predict, _, logit in tqdm(loader):

        target = target.numpy()
        predict = predict.numpy()

        target = np.unique(target.reshape(-1))
        target = target - 1
        target = target.tolist()
        if -1 in target:
            target.remove(-1)
        targets.append(target)
        logit = logit.numpy()
        for t in threshold:
            predict_ = predict.copy()
            predict_[logit < t] = 0

            predict_ = np.unique(predict_.reshape(-1))
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


def evaluator(args, mode, num_classes, thresholds, matches):
    assert thresholds is not None
    if isinstance(thresholds, float):
        thresholds = [thresholds]

    logs = []
    for t in thresholds:
        T = np.zeros((num_classes + 1, ))
        P = np.zeros((num_classes + 1, ))
        TP = np.zeros((num_classes + 1, ))
        BT = np.zeros((num_classes + 1,))
        BP = np.zeros((num_classes + 1,))
        BTP = np.zeros((num_classes + 1,))
        IoU = np.zeros((num_classes + 1, ))
        FMeasure = 0.0
        ACC = 0.0

        # mIoUs under different object sizes
        Ts = [np.zeros((num_classes + 1,)) for _ in range(4)]
        Ps = [np.zeros((num_classes + 1,)) for _ in range(4)]
        TPs = [np.zeros((num_classes + 1,)) for _ in range(4)]
        mIoUs = [np.zeros((num_classes + 1,)) for _ in range(4)]
        # bIoUs under different object sizes
        BTs = [np.zeros((num_classes + 1,)) for _ in range(4)]
        BPs = [np.zeros((num_classes + 1,)) for _ in range(4)]
        BTPs = [np.zeros((num_classes + 1,)) for _ in range(4)]
        mBIoUs = [np.zeros((num_classes + 1,)) for _ in range(4)]

        loader = get_dataset(args, mode, threshold=t, match=matches[t])
        for target, boundary_target, predict, boundary_predict, _ in tqdm(loader):
            target = target.numpy()
            predict = predict.numpy()
            boundary_target = boundary_target.numpy()
            boundary_predict = boundary_predict.numpy()

            area_intersection, area_output, area_target = \
                intersectionAndUnionGPU(
                    predict.reshape(-1), target.reshape(-1), num_classes + 1)
            
            area_intersection_boundary, area_output_boundary, area_target_boundary = \
                intersectionAndUnionGPU(
                    boundary_predict.reshape(-1), boundary_target.reshape(-1), num_classes + 2)
            
            IoUDifferentSizeGPUWithBoundary(predict.reshape(-1), target.reshape(-1), boundary_predict.reshape(-1), boundary_target.reshape(-1), num_classes + 1, Ts, Ps, TPs, BTs, BPs, BTPs)

            f_score = fscore(predict, target)

            T += area_output
            P += area_target
            TP += area_intersection
            BT += area_output_boundary[1:]
            BP += area_target_boundary[1:]
            BTP += area_intersection_boundary[1:]
            FMeasure += f_score

            img_label = np.argmax(area_output[1:]) + 1
            ACC += (area_target[img_label] > 0) * (area_output[img_label] > 0)

        IoU = TP / (T + P - TP + 1e-10)
        BIoU = BTP / (BT + BP - BTP + 1e-10)
        mIoU = np.mean(IoU).item() * 100
        mBIoU = np.mean(BIoU).item() * 100
        FMeasure = FMeasure.item() / len(loader.dataset) * 100
        ACC = ACC.item() * 100 / len(loader.dataset)

        for i in range(4): mIoUs[i] = np.mean((TPs[i] / (Ts[i] + Ps[i] - TPs[i] + 1e-10))[Ps[i] > 0]).item() * 100
        for i in range(4): mBIoUs[i] = np.mean((BTPs[i] / (BTs[i] + BPs[i] - BTPs[i] + 1e-10))[BPs[i] > 0]).item() * 100

        print(
            'Threshold: {:.2f}\tAcc: {:.2f}\tmIoU: {:.2f}\tmBIoU: {:.2f}\tFMeasure: {:.2f}\t'\
            'S: {:.2f}\tMS: {:.2f}\tML: {:.2f}\tL: {:.2f}\tBS: {:.2f}\tBMS: {:.2f}\tBML: {:.2f}\tBL: {:.2f}'.
            format(t, ACC, mIoU, mBIoU, FMeasure, mIoUs[0], mIoUs[1], mIoUs[2], mIoUs[3], mBIoUs[0], mBIoUs[1], mBIoUs[2], mBIoUs[3])
        )

        log = dict(th=t,
                   match=matches[t],
                   Acc=ACC,
                   mIoU=mIoU,
                   mBIoU=mBIoU,
                   IoUs=IoU * 100,
                   FMeasure=FMeasure,
                   S=mIoUs[0],
                   MS=mIoUs[1],
                   ML=mIoUs[2],
                   L=mIoUs[3],
                   BS=mBIoUs[0],
                   BMS=mBIoUs[1],
                   BML=mBIoUs[2],
                   BL=mBIoUs[3])

        logs.append(log)

    mious = [log['mIoU'] for log in logs]
    best = np.argmax(mious)
    log = logs[best]
    return log


def evaludation(args, mode):

    if args.curve:
        thresholds = [
            threshold / 100.0 for threshold in range(args.min, args.max + 1)
        ]
    else:
        thresholds = [args.t / 100.0]
    matches = match(args, mode, args.num_classes, threshold=thresholds)
    log = evaluator(
        args, mode,
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
