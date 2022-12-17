from munkres import Munkres
import numpy as np
import jittor as jt


def reAssignSingle(target1, target2, num_classes):
    """
    Matching with single label for each image.

    Args:
    target1 (list[int]): Groud truth label for each image.
    target2 (list[int]): Generated label for each image.
    num_classes (int): The number of classes.
    """
    matrix = np.zeros(shape=(num_classes, num_classes), dtype=np.float32).tolist()
    for i in range(num_classes):
        for j in range(num_classes):
            oldi = np.where(target1 == i)
            newj = np.where(target2 == j)
            co = np.intersect1d(oldi, newj)
            matrix[i][j] = - 1.0 * len(co)

    m = Munkres()
    indexes = m.compute(matrix)
    total = 0
    match = {}
    for row, column in indexes:
        total += matrix[row][column]
        match[column] = row

    return - total / len(target1), match


def reAssignMultiply(target1, target2, num_classes):
    """
    Matching with multiply labels for each image.

    Args:
    target1 (list[list[int]]): Groud truth label for each image.
    target2 (list[list[int]]): Generated label for each image.
    num_classes (int): The number of classes.
    """
    matrix = np.zeros(shape=(num_classes, num_classes), dtype=np.float32).tolist()
    olds = {}
    news = {}
    for i in range(num_classes):
        olds[i] = np.array([i in item for item in target1])
        news[i] = np.array([i in item for item in target2])
    for i in range(num_classes):
        for j in range(num_classes):
            coi = np.logical_and(olds[i], news[j])
            matrix[i][j] = - 1.0 * np.sum(coi)

    m = Munkres()
    indexes = m.compute(matrix)
    total = 0
    match = {}
    for row, column in indexes:
        total += matrix[row][column]
        match[column] = row

    return - total / len(target1), match


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with jt.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdims=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
