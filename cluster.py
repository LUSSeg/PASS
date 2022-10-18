import os
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
from torchvision import transforms
import numpy as np
from cluster.kmeans import Kmeans
from sklearn.metrics.cluster import normalized_mutual_info_score
from cluster.hungarian import reAssignSingle, reAssignMultiply
import json
import src.resnet as resnet_model
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
args = parser.parse_args()


def main():

    model = resnet_model.__dict__[args.arch](hidden_mlp=0, output_dim=0, nmb_prototypes=0, train_mode='pixelattn')

    # loading pretrained weights
    checkpoint = torch.load(args.pretrained, map_location="cpu")["state_dict"]
    state_dict = {}
    for k in checkpoint.keys():
        if k.startswith("module") and not k.startswith("module.prototypes") and not k.startswith("module.projection"):
            state_dict[k[len("module.") :]] = checkpoint[k]

    msg = model.load_state_dict(state_dict, strict=False)
    print("=> loaded model '{}'".format(args.pretrained))
    assert len(msg.missing_keys) == 0, msg.missing_keys
    model = nn.DataParallel(model)
    model.cuda()
    model.eval()

    # build datasets
    train_folder = os.path.join(args.data_path, "train")
    val_folder = os.path.join(args.data_path, "validation")
    mask_folder = os.path.join(args.data_path, "validation-segmentation")
    dump_path = os.path.join(args.dump_path, "cluster")
    if not os.path.exists(dump_path):
        os.makedirs(dump_path)

    normalize = transforms.Normalize(
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
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
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
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, num_workers=args.num_workers, pin_memory=True
    )

    # extracting features
    print("Extracting features ...")
    _, train_targets, train_embeddings, train_paths = getEmb(train_loader, model, len(train_dataset))
    _, _, val_embeddings, val_paths = getEmb(val_loader, model.module, len(val_dataset))
    train_targets = train_targets.tolist()

    # clustering
    print("Clustering features ...")
    deepcluster = Kmeans(args.num_classes, nredo=30)
    deepcluster.cluster(train_embeddings.copy(), npdata2=val_embeddings.copy(), save_centroids=True)
    train_labels = deepcluster.labels[:len(train_dataset)]
    val_labels = [[x] for x in deepcluster.labels[len(train_dataset):]]

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
    targets = torch.zeros(size).long().cuda()
    embeddings = None
    indexes = torch.zeros(size).long().cuda()
    paths = []
    start_idx = 0
    with torch.no_grad():
        for idx, path, inputs, target in tqdm(dataloader):
            nmb_unique_idx = inputs.size(0)

            # get embeddings
            inputs = inputs.cuda(non_blocking=True)
            emb = model(inputs, mode="cluster")

            if start_idx == 0:
                embeddings = torch.zeros(size, emb.shape[1]).cuda()

            # fill the memory bank
            targets[start_idx : start_idx + nmb_unique_idx] = target
            indexes[start_idx : start_idx + nmb_unique_idx] = idx
            embeddings[start_idx : start_idx + nmb_unique_idx] = emb
            paths += path
            start_idx += nmb_unique_idx
    return indexes.cpu().numpy(), targets.cpu().numpy(), embeddings.cpu().numpy(), paths


def fix_random_seeds():
    """
    Fix random seeds.
    """
    if args.seed is None:
        return
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)


if __name__ == "__main__":
    fix_random_seeds()
    main()
