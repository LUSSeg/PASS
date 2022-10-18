import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import src.resnet as resnet_model
from src.singlecropdataset import InferImageFolder
import torch.multiprocessing as mp
import torch.distributed as dist


def parse_args():
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument("--mode", type=str, required=True)
    parser.add_argument("--dump_path", type=str, default=None, help="The path to save results.")
    parser.add_argument("--data_path", type=str, default=None, help="The path to ImagenetS dataset.")
    parser.add_argument("--pretrained", type=str, default=None, help="The model checkpoint file.")
    parser.add_argument("-a", "--arch", metavar="ARCH", help="The model architecture.")
    parser.add_argument("-c", "--num-classes", default=50, type=int, help="The number of classes.")
    parser.add_argument("-t", "--threshold", default=0, type=float, help="The threshold to filter the 'others' categroies.")
    parser.add_argument("--test", action='store_true', help="whether to save the logit. Enabled when finding the best threshold.")
    parser.add_argument("--centroid", type=str, default=None, help="The centroids of clustering.")
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--rank", type=int, default=0)
    args = parser.parse_args()

    return args


def main_worker(rank, args):

    args.rank = rank 
    dist.init_process_group(backend='nccl', init_method=args.dist_url, world_size=args.num_gpus,
                                    rank=args.rank)
    torch.cuda.set_device(rank)
    device = torch.device("cuda:{}".format(rank))

    centroids = np.load(args.centroid)
    centroids = torch.from_numpy(centroids).cuda()
    centroids = nn.functional.normalize(centroids, dim=1, p=2)

    # build model
    model = resnet_model.__dict__[args.arch](hidden_mlp=0, output_dim=0, nmb_prototypes=0, train_mode='pixelattn')

    checkpoint = torch.load(args.pretrained, map_location="cpu")["state_dict"]
    state_dict = {}
    for k in checkpoint.keys():
        if k.startswith("module") and not k.startswith("module.prototypes") and not k.startswith("module.projection"):
            state_dict[k[len("module.") :]] = checkpoint[k]

    msg = model.load_state_dict(state_dict, strict=False)
    print("=> loaded model '{}'".format(args.pretrained))
    assert len(msg.missing_keys) == 0, msg.missing_keys
    model.cuda()
    model.to(device)
    model.eval()

    # build dataset
    data_path = os.path.join(args.data_path, args.mode)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    dataset = InferImageFolder(
        root=data_path,
        transform=transforms.Compose(
            [
                transforms.Resize(256),
                transforms.ToTensor(),
                normalize,
            ]
        ),
        rank=rank,
        num_gpus=args.num_gpus
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, num_workers=16, pin_memory=True
    )

    dump_path = os.path.join(args.dump_path, args.mode)

    for images, path, height, width in tqdm(dataloader):
        path = path[0]
        cate = path.split("/")[-2]
        name = path.split("/")[-1].split(".")[0]
        if not os.path.exists(os.path.join(dump_path, cate)):
            os.makedirs(os.path.join(dump_path, cate))

        with torch.no_grad():
            h = height.item()
            w = width.item()

            out, mask = model(images.cuda(device), mode='inference_pixel_attention')

            mask = F.upsample(mask, (h, w), mode="bilinear", align_corners=False).squeeze()

            out = nn.functional.normalize(out, dim=1, p=2)
            B, C, H, W = out.shape
            out = out.view(B, C, -1).permute(0, 2, 1).contiguous().view(-1, C)

            cosine = torch.mm(out, centroids.t())
            cosine = cosine.view(1, H, W, args.num_classes).permute(0, 3, 1, 2)

            logit = mask
            prediction = torch.argmax(cosine, dim=1, keepdim=True) + 1
            prediction = F.interpolate(prediction.float(), (h, w), mode="nearest").squeeze()
            
            prediction[logit.squeeze() < args.threshold] = 0

            res = torch.zeros(size=(prediction.shape[0], prediction.shape[1], 3))
            res[:, :, 0] = prediction % 256
            res[:, :, 1] = prediction // 256

            res = res.cpu().numpy()
            logit = logit.cpu().numpy()

            res = Image.fromarray(res.astype(np.uint8))
            res.save(os.path.join(dump_path, cate, name + ".png"))
            if args.test:
                np.save(os.path.join(dump_path, cate, name + ".npy"), logit)


if __name__ == "__main__":
    args = parse_args()
    args.num_gpus = torch.cuda.device_count()
    if args.num_gpus == 1:
        main_worker(rank=0, args=args)
    else:
        torch.multiprocessing.set_start_method('spawn')
        mp.spawn(main_worker, nprocs=args.num_gpus, args=(args,))