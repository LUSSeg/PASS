from torchvision.datasets import ImageFolder
from PIL import Image
from torchvision import transforms
import os
import torch


def get_loader(args, return_gt=True):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    dataset = InferenceDataset(
        os.path.join(args.data_path, args.mode), os.path.join(args.data_path, f'{args.mode}-segmentation'), 
        transform=transforms.Compose(
            [
                transforms.Resize(256),
                transforms.ToTensor(),
                normalize,
            ]
        ),
        target_transform=transforms.Compose(
            [
                transforms.ToTensor(),
                lambda x: (x * 255.0).long(),
                ]
            ),
        return_gt=return_gt)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
        drop_last=False)
    return loader

class InferenceDataset(ImageFolder):
    def __init__(self, img_root, label_root=None, transform=None, target_transform=None, return_gt=False):
        super(InferenceDataset, self).__init__(img_root, transform)

        self.label_root = label_root
        self.return_gt = return_gt
        self.segmentation_transform = target_transform

    def __getitem__(self, item):

        path = self.imgs[item][0]
        sample = self.loader(path)
        height, width = sample.size[1], sample.size[0]
        if self.transform is not None:
            sample = self.transform(sample)

        if self.return_gt:
            gt = self.load_semantic(path)
            gt = self.segmentation_transform(gt)
            gt = gt[1, :, :] * 256 + gt[0, :, :]
            return sample, gt, path, height, width

        return sample, path, height, width

    def __len__(self):
        return len(self.imgs)

    def load_semantic(self, path):
        cate, name = path.split('/')[-2:]
        name = name.replace("JPEG", "png")
        path = os.path.join(self.label_root, cate, name)
        semantic = Image.open(path).convert('RGB')

        return semantic