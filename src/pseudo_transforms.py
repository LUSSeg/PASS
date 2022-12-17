import math
import random
from PIL import Image
import warnings
import jittor as jt
from jittor.transform import to_tensor, to_pil_image, _get_image_size
import jittor.transform.function_pil as F_pil
import numpy as np


class Compose(object):
    '''
    Base class for combining various transformations.

    Args::

    [in] transforms(list): a list of transform.

    Example::

        transform = transform.Compose([
            transform.Resize(opt.img_size),
            transform.Gray(),
            transform.ImageNormalize(mean=[0.5], std=[0.5]),
        ])
        img_ = transform(img)
    '''
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *data):
        assert len(data) == 2
        img, semantic = data

        for t in self.transforms:
            if 'RandomResizedCropSemantic' in t.__class__.__name__:
                img, semantic = t(img, semantic)
            elif 'FlipSemantic' in t.__class__.__name__:
                img, semantic = t(img, semantic)
            elif 'ToTensorSemantic' in t.__class__.__name__:
                img, semantic = t(img, semantic)
            else:
                img = t(img)
        return img, semantic


class RandomHorizontalFlipSemantic(object):
    """
    Random flip the image horizontally.

    Args::

        [in] p(float): The probability of image flip, default: 0.5.

    Example::

        transform = transform.RandomHorizontalFlip(0.6)
        img_ = transform(img)
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img:Image.Image, semantic:Image.Image):
        if not isinstance(img, Image.Image):
            img = to_pil_image(img)
        if not isinstance(semantic, Image.Image):
            semantic = to_pil_image(semantic)
        if random.random() < self.p:
            return img.transpose(Image.FLIP_LEFT_RIGHT), semantic.transpose(Image.FLIP_LEFT_RIGHT)
        return img, semantic


class RandomResizedCropSemantic(object):
    """Crop the given PIL Image to random size and aspect ratio.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR):
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        width, height = _get_image_size(img)
        area = height * width

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if (in_ratio < min(ratio)):
            w = width
            h = int(round(w / min(ratio)))
        elif (in_ratio > max(ratio)):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def __call__(self, img, semantic):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        if not isinstance(img, Image.Image):
            img = to_pil_image(img)
        if not isinstance(semantic, Image.Image):
            semantic = to_pil_image(semantic)
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return F_pil.resized_crop(img, i, j, h, w, self.size, self.interpolation), \
               F_pil.resized_crop(semantic, i, j, h, w, self.size, Image.NEAREST)

    def __repr__(self):
        interpolate_str = str(self.interpolation)
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string


class ToTensorSemantic:
    def __call__(self, pic, semantic):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        semantic = jt.array(np.asarray(semantic).astype(np.float32)).transpose(2, 0, 1)
        return to_tensor(pic), semantic

    def __repr__(self):
        return self.__class__.__name__ + '()'