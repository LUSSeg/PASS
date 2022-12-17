import os
import jittor as jt
import jittor.nn as nn
jt.flags.use_cuda = 1
import argparse
from tqdm import tqdm
import numpy as np
from cluster.kmeans import Kmeans
from sklearn.metrics.cluster import normalized_mutual_info_score
from cluster.hungarian import reAssignSingle
import json
import src.resnet as resnet_model
from src.utils import bool_flag
import jittor.transform as transforms
from src.singlecropdataset import ClusterImageFolder

parser = argparse.ArgumentParser(description="Argument For Eval")
parser.add_argument("--num_workers", type=int, default=32, help="num of workers to use")
parser.add_argument("-a", "--arch", default="resnet50", type=str, help="convnet architecture")
parser.add_argument("-b", "--batch_size", default=256, type=int, metavar="N", help="batch size")
parser.add_argument("-c", "--num_classes", default=50, type=int, help="the number of classes")
parser.add_argument("-s", "--seed", default=None, type=int, help="the seed for clustering")
parser.add_argument("--pretrained", type=str, default=None, help="the model checkpoint")
parser.add_argument("--data_path", type=str, default=None, help="path to data")
parser.add_argument("--dump_path", type=str, default=None, help="path to save clustering results")
parser.add_argument("--checkpoint_key", type=str, default='state_dict', help="key of model in checkpoint")
args = parser.parse_args()


def main():

    if 'resnet' in args.arch:
        model = resnet_model.__dict__[args.arch](hidden_mlp=0, output_dim=0, nmb_prototypes=0, train_mode='pixelattn')
    else:
        raise NotImplementedError()

    # loading pretrained weights
    checkpoint = jt.load(args.pretrained)[args.checkpoint_key]
    for k in list(checkpoint.keys()):
        if k.startswith('module.'):
            checkpoint[k[len('module.'):]] = checkpoint[k]
            del checkpoint[k]
            k = k[len('module.'):]
        if k not in model.state_dict().keys():
            del checkpoint[k]

    model.load_state_dict(checkpoint)
    print("=> loaded model '{}'".format(args.pretrained))
    model.eval()

    # build datasets
    train_folder = os.path.join(args.data_path, "train")
    val_folder = os.path.join(args.data_path, "validation")
    dump_path = os.path.join(args.dump_path, "cluster")
    if not os.path.exists(dump_path):
        os.makedirs(dump_path)

    normalize = transforms.ImageNormalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    train_dataset = ClusterImageFolder(
        train_folder,
        transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )
    train_loader = train_dataset.set_attrs(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    val_dataset = ClusterImageFolder(
        val_folder,
        transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        )
    )
    val_loader = val_dataset.set_attrs(
        batch_size=1, 
        num_workers=args.num_workers
    )

    # extracting features
    print("Extracting features ...")
    _, train_targets, train_embeddings, train_paths = getEmb(train_loader, model, len(train_dataset.imgs))
    _, _, val_embeddings, val_paths = getEmb(val_loader, model, len(val_dataset.imgs))
    train_targets = train_targets.tolist()

    # clustering
    print("Clustering features ...")
    deepcluster = Kmeans(args.num_classes, nredo=30)
    deepcluster.cluster(train_embeddings.copy(), npdata2=val_embeddings.copy(), save_centroids=True)
    train_labels = deepcluster.labels[:len(train_dataset.imgs)]
    val_labels = [[x] for x in deepcluster.labels[len(train_dataset.imgs):]]

    # clustering metric
    nmi_train = normalized_mutual_info_score(train_targets, train_labels)
    acc_train, _ = reAssignSingle(np.array(train_targets), np.array(train_labels), num_classes=args.num_classes,)
    print("train nmi {:.4f}".format(nmi_train))
    print("train acc {:.4f}".format(acc_train))

    result = dict(
        nmi_train=nmi_train,
        acc_train=acc_train,
        train_labels=train_labels,
        val_labels=val_labels,
        centroids=deepcluster.centroids,
    )

    save(train_paths, val_paths, dump_path, result)


def save(train_paths, val_paths, dump_path, result):

    # save centroids o clustering
    centroids = result["centroids"].reshape(args.num_classes, -1)
    np.save(os.path.join(dump_path, "centroids.npy"), centroids)

    # save generated labels
    train_labeled = []
    val_labeled = []
    for img, label in zip(train_paths, result['train_labels']):
        train_labeled.append("{0}/{1} {2}".format(img.split('/')[-2], img.split('/')[-1], label))
    for img, label in zip(val_paths, result['val_labels']):
        val_labeled.append("{0}/{1} {2}".format(img.split('/')[-2], img.split('/')[-1], label[0]))

    with open(os.path.join(dump_path, "train_labeled.txt"), "w") as f:
        f.write("\n".join(train_labeled))
    with open(os.path.join(dump_path, "val_labeled.txt"), "w") as f:
        f.write("\n".join(val_labeled))


def getEmb(dataloader, model, size):
    targets = jt.zeros(size).long()
    embeddings = None
    indexes = jt.zeros(size).long()
    paths = []
    start_idx = 0
    with jt.no_grad():
        for idx, path, inputs, target in tqdm(dataloader):
            nmb_unique_idx = inputs.size(0)

            # get embeddings
            emb = model(inputs, mode="cluster")

            if start_idx == 0:
                embeddings = jt.zeros(size, emb.shape[1])

            # fill the memory bank
            targets[start_idx : start_idx + nmb_unique_idx] = target.copy()
            indexes[start_idx : start_idx + nmb_unique_idx] = idx.copy()
            embeddings[start_idx : start_idx + nmb_unique_idx] = emb.copy()
            paths += path
            start_idx += nmb_unique_idx

            jt.clean_graph()
            jt.sync_all()
            jt.gc()
    return indexes.cpu().numpy(), targets.cpu().numpy(), embeddings.cpu().numpy(), paths


def fix_random_seeds():
    """
    Fix random seeds.
    """
    if args.seed is None:
        return
    jt.set_global_seed(args.seed)


if __name__ == "__main__":
    fix_random_seeds()
    main()
