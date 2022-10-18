import os
import torch
import argparse
from tqdm import tqdm
import torch.nn.functional as F
import data
import torch.nn as nn
from torchvision.models import resnet

parser = argparse.ArgumentParser(description="Argument For Eval")
parser.add_argument("--data_path", default='imagenet50', type=str)
parser.add_argument("--dump_path", default='imagenet50', type=str)
parser.add_argument('--workers', default=32, type=int)
parser.add_argument('--arch', default='resnet50', type=str)
parser.add_argument('--pretrained', default=None, type=str, required=True)
parser.add_argument('--num_classes', type=int, default=50, choices=[50, 300, 919])
parser.add_argument('--mode', type=str, default='train-semi', choices=['train-semi'])
args = parser.parse_args()


def main():

    # dataloader
    loader = data.get_loader(args, True)

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

    embeddings = []
    labels = []
    with torch.no_grad():
        for inputs, segmentation, _, _, _ in tqdm(loader):
            inputs = inputs.cuda(non_blocking=True)
            segmentation = segmentation.cuda()
            emb = model(inputs)
            emb = F.interpolate(emb, size=(emb.shape[2] * 4, emb.shape[3] * 4), mode="bilinear")
            n, c, h, w = emb.shape
            segmentation = F.interpolate(
                segmentation.unsqueeze(1).float(), size=(h, w), mode="nearest"
            ).long()
            emb = emb.view(n, c, -1)
            segmentation = segmentation.view(n, 1, -1)
            for i in range(n):
                sample = emb[i]
                seg = segmentation[i]
                vs = torch.unique(seg).tolist()
                for v in vs:
                    if v == 1000:
                        continue
                    mask = 1.0 * (seg == v)
                    instance = sample * mask
                    instance = torch.mean(instance, dim=1)
                    embeddings.append(instance.unsqueeze(0))
                    labels.append(v)
    embeddings = torch.cat(embeddings, dim=0)
    labels = torch.LongTensor(labels)

    if not os.path.exists(args.dump_path):
        os.makedirs(args.dump_path)
    torch.save(embeddings, os.path.join(args.dump_path, "embeddings_train.pth"))
    torch.save(labels, os.path.join(args.dump_path, "labels_train.pth"))


if __name__ == "__main__":
    main()