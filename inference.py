import argparse
import json
import os
import shutil

import numpy as np
import jittor as jt
import jittor.nn as nn
jt.flags.use_cuda = 1
from PIL import Image
import jittor.transform as transforms
from tqdm import tqdm

import src.resnet as resnet_model
from src.singlecropdataset import InferImageFolder
from src.utils import hungarian


def parse_args():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('--mode', type=str, required=True)
    parser.add_argument('--dump_path',
                        type=str,
                        default=None,
                        help='The path to save results.')
    parser.add_argument('--match_file',
                        type=str,
                        default=None,
                        help='The matching file for test set.')
    parser.add_argument('--data_path',
                        type=str,
                        default=None,
                        help='The path to ImagenetS dataset.')
    parser.add_argument('--pretrained',
                        type=str,
                        default=None,
                        help='The model checkpoint file.')
    parser.add_argument('-a',
                        '--arch',
                        metavar='ARCH',
                        help='The model architecture.')
    parser.add_argument('-c',
                        '--num-classes',
                        default=50,
                        type=int,
                        help='The number of classes.')
    parser.add_argument('--max_res', default=1000, type=int, help="Maximum resolution for evaluation. 0 for disable.")
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

    return args


def main_worker(args):
    # build model
    if 'resnet' in args.arch:
        model = resnet_model.__dict__[args.arch](
            hidden_mlp=0, output_dim=0, nmb_prototypes=0, train_mode='finetune', num_classes=args.num_classes)
    else:
        raise NotImplementedError()

    checkpoint = jt.load(args.pretrained)["state_dict"]
    for k in list(checkpoint.keys()):
        if k not in model.state_dict().keys():
            del checkpoint[k]
    model.load_state_dict(checkpoint)
    print("=> loaded model '{}'".format(args.pretrained))
    model.eval()

    # build dataset
    assert args.mode in ['validation', 'test']
    data_path = os.path.join(args.data_path, args.mode)
    validation_segmentation = os.path.join(args.data_path,
                                           'validation-segmentation')
    normalize = transforms.ImageNormalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    dataset = InferImageFolder(root=data_path,
                               transform=transforms.Compose([
                                   transforms.Resize(256),
                                   transforms.ToTensor(),
                                   normalize,
                               ]))
    dataloader = dataset.set_attrs(
        batch_size=1, 
        num_workers=16)

    dump_path = os.path.join(args.dump_path, args.mode)

    targets = []
    predictions = []
    for images, path, height, width in tqdm(dataloader):
        path = path[0]
        cate = path.split('/')[-2]
        name = path.split('/')[-1].split('.')[0]
        if not os.path.exists(os.path.join(dump_path, cate)):
            os.makedirs(os.path.join(dump_path, cate))

        with jt.no_grad():
            H = height.item()
            W = width.item()

            output = model(images)

            if H * W > args.max_res * args.max_res and args.max_res > 0:
                output = nn.interpolate(output, (args.max_res, int(args.max_res * W / H)), mode="bilinear", align_corners=False)
                output = jt.argmax(output, dim=1, keepdims=True)[0]
                prediction = nn.interpolate(output.float(), (H, W), mode="nearest").long()
            else:
                output = nn.interpolate(output, (H, W), mode="bilinear", align_corners=False)
                prediction = jt.argmax(output, dim=1, keepdims=True)[0]

            prediction = prediction.squeeze(0).squeeze(0)
            res = jt.zeros((prediction.shape[0], prediction.shape[1], 3))
            res[:, :, 0] = prediction % 256
            res[:, :, 1] = prediction // 256
            res = res.cpu().numpy()

            res = Image.fromarray(res.astype(np.uint8))
            res.save(os.path.join(dump_path, cate, name + '.png'))

            if args.mode == 'validation':
                target = Image.open(os.path.join(validation_segmentation, cate, name + '.png'))
                target = np.array(target).astype(np.int32)
                target = target[:, :, 1] * 256 + target[:, :, 0]

                # Prepare for matching (target)
                target_unique = np.unique(target.reshape(-1))
                target_unique = target_unique - 1
                target_unique = target_unique.tolist()
                if -1 in target_unique:
                    target_unique.remove(-1)
                targets.append(target_unique)

                # Prepare for matching (prediction)
                prediction_unique = np.unique(prediction.cpu().numpy().reshape(-1))
                prediction_unique = prediction_unique - 1
                prediction_unique = prediction_unique.tolist()
                if -1 in prediction_unique:
                    prediction_unique.remove(-1)
                predictions.append(prediction_unique)
            
            jt.clean_graph()
            jt.sync_all()
            jt.gc()
                
    if args.mode == 'validation':    
        _, match = hungarian(targets, predictions, num_classes=args.num_classes)
        match = {k + 1: v + 1 for k, v in match.items()}
        match[0] = 0

        with open(os.path.join(dump_path, 'match.json'), 'w') as f:
            f.write(json.dumps(match))

    elif args.mode == 'test':
        assert os.path.exists(args.match_file)
        shutil.copyfile(args.match_file, os.path.join(dump_path, 'match.json'))

        method = 'Method name: {}\n'.format(args.method) + \
            'Training data: {}\nTraining scheme: {}\n'.format(
                args.train_data, args.train_scheme) + \
            'Networks: {}\nPaper/Project link: {}\n'.format(
                args.arch, args.link) + \
            'Method description: {}'.format(args.description)
        with open(os.path.join(dump_path, 'method.txt'), 'w') as f:
            f.write(method)

        # zip for submission
        shutil.make_archive(os.path.join(args.dump_path, args.mode), 'zip', root_dir=dump_path)


if __name__ == '__main__':
    args = parse_args()
    main_worker(args=args)

