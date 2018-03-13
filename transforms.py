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
from torchvision.transforms import ToTensor

class Compose(object):
    """ Composes several co_transforms together.
    """

    def __init__(self, co_transforms):
        self.co_transforms = co_transforms

    def __call__(self, img, target):
        for t in self.co_transforms:
            img, target = t(img,target)
        return img, target

    
class RandomTranslate(object):
    def __init__(self, translation):
        if isinstance(translation, numbers.Number):
            self.translation = (int(translation), int(translation))
        else:
            self.translation = translation

    def __call__(self, img, target):
        h, w, c = img.shape
        th, tw = self.translation
        tw = random.randint(-tw, tw)
        th = random.randint(-th, th)
        if tw == 0 and th == 0:
            return img, target

        x1, x2, x3, x4 = max(0, -th), min(h - th, h), max(0, th), min(h + th, h)
        y1, y2, y3, y4 = max(0, -tw), min(w - tw, w), max(0, tw), min(w + tw, w)
        img[x3:x4, y3:y4, :] = img[x1:x2, y1:y2, :]
        
        target = target + np.array([[th, tw]])
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

    
class RandomVerticalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """

    def __call__(self, img, target):
        h, _, _ = im.shape
        if random.random() < 0.5:
            img = np.copy(np.flipud(img))
            target[:, 0] = h - np.copy(target[:, 0])
        return img, target
    
    
class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """

    def __call__(self, img, target):
        _, w, _ = im.shape
        if random.random() < 0.5:
            img = np.copy(np.fliplr(img))
            target[:, 1] = w - np.copy(target[:, 1])
        return img, target

    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self):
        self.to_tensor = ToTensor()

    def __call__(self, img, target):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # image = image.transpose((2, 0, 1))
        return to_tensor(img), torch.from_numpy(target)
