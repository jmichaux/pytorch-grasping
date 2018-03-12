import numbers
import numpy as np

import cv2
from PIL import Image
import scipy.ndimage as ndimage
from skimage.transform import resize
from scipy.misc import bytescale

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms, utils

class Compose(object):
    """ Composes several co_transforms together.
    """

    def __init__(self, co_transforms):
        self.co_transforms = co_transforms

    def __call__(self, img, target):
        for t in self.co_transforms:
            img, target = t(img,target)
        return img, target

class CenterCrop(object):
    '''Crops image at center'''
    def __init__(self, size=320):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, target):
        h1, w1, _ = img.shape
        th, tw = self.size

        # note that we are indexing images as numpy arrays
        x1 = int(round((h1 - th) / 2.))
        y1 = int(round((w1 - tw) / 2.))

        img = img[x1: x1 + th, y1: y1 + tw, :]

        target = target - np.array([[x1, y1]])
        return img, target

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, img, target):

        h, w = img.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        # img = resize(img, (new_h, new_w))
        img = cv2.resize(img, (new_h, new_w))
        ratio = h / new_h
        target /= ratio
        return img, target

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self):
        pass

    def __call__(self, img, target):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # image = image.transpose((2, 0, 1))
        return torch.from_numpy(img), torch.from_numpy(target)
