import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from PIL import Image
import data
import shutil
from torchvision.models import resnet


parser = argparse.ArgumentParser(description="Argument For Eval")
parser.add_argument("--data_path", default='imagenet50', type=str)
parser.add_argument("--dump_path", default='imagenet50', type=str)
parser.add_argument('--workers', default=32, type=int)
parser.add_argument('--arch', default='resnet50', type=str)
parser.add_argument('--pretrained', default=None, type=str, required=True)
parser.add_argument('--num_classes', type=int, default=50, choices=[50, 300, 919])
parser.add_argument('--mode', type=str, default='validation', choices=['validation', 'test'])
parser.add_argument("-k", type=int, default=10)
parser.add_argument("-T", type=float, default=0.07)

parser.add_argument('--method',
                    default='example submission',
                    help='Method name in method description file(.txt).')
parser.add_argument('--train_data',
                    default='null',
                    help='Training data in method description file(.txt).')
parser.add_argument(
    '--train_scheme',
    default='null',
    help='Training scheme in method description file(.txt), \
        e.g., SSL, Sup, SSL+Sup.')
parser.add_argument(
    '--link',
    default='null',
    help='Paper/project link in method description file(.txt).')
parser.add_argument(
    '--description',
    default='null',
    help='Method description in method description file(.txt).')
args = parser.parse_args()
args = parser.parse_args()


@torch.no_grad()
def knn_classifier(train_features, train_labels, test_features, k, T, num_classes=1000):
    # https://github.com/facebookresearch/dino/blob/main/eval_knn.py
    train_features = train_features.t()
    num_test_images, num_chunks = test_features.shape[0], 1
    imgs_per_chunk = num_test_images // num_chunks
    retrieval_one_hot = torch.zeros(k, num_classes).cuda()
    for idx in range(0, num_test_images, imgs_per_chunk):
        # get the features for test images
        features = test_features[idx : min((idx + imgs_per_chunk), num_test_images), :]
        batch_size = features.shape[0]

        # calculate the dot product and compute top-k neighbors
        similarity = torch.mm(features, train_features)
        distances, indices = similarity.topk(k, largest=True, sorted=True)
        candidates = train_labels.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)

        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        distances_transform = distances.clone().div_(T).exp_()
        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, num_classes),
                distances_transform.view(batch_size, -1, 1),
            ),
            1,
        )
        _, predictions = probs.sort(1, True)

    return predictions


def main():

    centroids = torch.load(os.path.join(args.dump_path, "embeddings_train.pth")).cuda()
    labels = torch.load(os.path.join(args.dump_path, "labels_train.pth")).cuda()
    centroids = nn.functional.normalize(centroids, dim=1, p=2)

    # dataloader
    args.gt_dir = None
    loader = data.get_loader(args, False)

    # model
    resnet_model = getattr(resnet, args.arch)
    model = resnet_model()
    state_dict = torch.load(args.pretrained, map_location='cpu')['state_dict']
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"=> loaded model '{args.pretrained}' with msg: {msg}")
    model = nn.Sequential(*list(model.children())[:-2])
    model.cuda()
    model.eval()

    save_path = os.path.join(args.dump_path, args.mode)
    for images, path, height, width in tqdm(loader):
        path = path[0]
        cate = path.split("/")[-2]
        name = path.split("/")[-1].split(".")[0]
        if not os.path.exists(os.path.join(save_path, cate)):
            os.makedirs(os.path.join(save_path, cate))

        with torch.no_grad():

            H = height.item()
            W = width.item()

            out = model.forward(images.cuda())
            out = F.interpolate(out, size=(out.shape[2] * 4, out.shape[3] * 4), mode="bilinear")

            n, c, h, w = out.shape
            assert n == 1
            out = nn.functional.normalize(out, dim=1, p=2)
            pred = knn_classifier(
                centroids,
                labels,
                out.view(c, -1).transpose(1, 0),
                k=args.k,
                T=args.T,
                num_classes=args.num_classes + 1,
            )
            pred = pred[:, 0]
            pred = pred.view(h, w)

            pred = F.interpolate(pred.unsqueeze(0).unsqueeze(0).float(), (H, W), mode="nearest").squeeze()
            res = torch.zeros(size=(pred.shape[0], pred.shape[1], 3))
            res[:, :, 0] = pred % 256
            res[:, :, 1] = pred // 256
            res = res.cpu().numpy()

            res = Image.fromarray(res.astype(np.uint8))
            res.save(os.path.join(save_path, cate, name + ".png"))


    if args.mode == 'test':

        method = 'Method name: {}\n'.format(args.method) + \
            'Training data: {}\nTraining scheme: {}\n'.format(
                args.train_data, args.train_scheme) + \
            'Networks: {}\nPaper/Project link: {}\n'.format(
                args.arch, args.link) + \
            'Method description: {}'.format(args.description)
        with open(os.path.join(save_path, 'method.txt'), 'w') as f:
            f.write(method)

        # zip for submission
        shutil.make_archive(os.path.join(args.dump_path, args.mode), 'zip', root_dir=save_path)

if __name__ == "__main__":
    main()