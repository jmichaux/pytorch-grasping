import random
import numbers
import numpy as np

from PIL import Image
from scipy.misc import bytescale

from torchvision.transforms import functional as f


class Compose(object):
    """ Composes several co_transforms together.
    """

    def __init__(self, co_transforms):
        self.co_transforms = co_transforms

    def __call__(self, img, bbox, pcd=None):
        for t in self.co_transforms:
            img, bbox, pcd = t(img, bbox, pcd)
        return img, bbox, pcd


class RandomTranslation(object):
    """
    https://stackoverflow.com/questions/37584977/translate-image-using-pil
    """
    def __init__(self, translation=None):
        if isinstance(translation, numbers.Number):
            self.translation = (int(translation), int(translation))
        else:
            if len(degrees) != 2:
                raise ValueError("If translation is a sequence, it must be of len 2.")
            self.translation = translation

    @staticmethod
    def get_params(translation, img_size, bbox):
        if translation is not None:
            min_x, max_x, min_y, max_y = min(bbox[:,0]), max(bbox[:,0]), min(bbox[:,1]), max(bbox[:,1])

            # added 20-30 pixel buffer to allowable translations
            max_dx = min(450 - max_x, translation[0])
            min_dx = max(-(min_x - 180), -translation[0])

            max_dy = min(370 - max_y, translation[1])
            min_dy = max(-(min_y - 100), -translation[1])

            translations = (np.round(random.uniform(min_dx, max_dx)),
                            np.round(random.uniform(min_dy, max_dy)))
#             max_dx = translation[0] #* img_size[0]
#             max_dy = translation[1] #* img_size[1]
#             translations = (np.round(random.uniform(-max_dx, max_dx)),
#                             np.round(random.uniform(-max_dy, max_dy)))
        else:
            translations = (0, 0)
        return translations

    @staticmethod
    def translate(img, translation):
        return img.transform(img.size, Image.AFFINE, (1, 0, -translation[0], 0,  1, -translation[1]))

    def __call__(self, img, bbox, pcd=None):
        translation = self.get_params(self.translation, img.size, bbox)
        img = self.translate(img, translation)
        bbox = bbox + np.array([[translation[0], translation[1]]])

        if pcd is not None:
            pcd = self.translate(pcd, translation)
        return img, bbox, pcd


class RandomRotation(object):
    def __init__(self, degrees, resample=False, expand=False, center=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.resample = resample
        self.expand = expand
        self.center = center

    @staticmethod
    def get_params(degrees):
        angle = random.uniform(degrees[0], degrees[1])
        return angle

    def __call__(self, img, bbox, pcd=None):
        # images have non-standard axes, so a positive angle should rotate
        # the image clockwise. This is why we use negative angle
        angle = self.get_params(self.degrees)
        img = f.rotate(img, -angle, self.resample, self.expand, self.center)

        theta = np.radians(angle)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s), (s, c)))

        # Translate center
        center = np.array([[320, 240]])
        bbox = bbox - center

        bbox = np.transpose(np.matmul(R, np.transpose(bbox.copy())))
        bbox = bbox + center

        if pcd is not None:
            pcd = f.rotate(pcd, -angle, self.resample, self.expand, self.center)
        return img, bbox, pcd


class CenterCrop(object):
    '''Crops image at center'''
    def __init__(self, size=320):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, bbox, pcd=None):
        w, h, = img.size
        th, tw = self.size

        img = f.center_crop(img, self.size)

        # compute image center
        x = int(round((w - tw) / 2.))
        y = int(round((h - th) / 2.))

        bbox = bbox - np.array([[x, y]])
        if pcd is not None:
            pcd = f.center_crop(pcd, self.size)
        return img, bbox, pcd


class Resize(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size, interpolation=Image.BILINEAR):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
        self.interpolation = interpolation

    def __call__(self, img, bbox, pcd=None):

        w, h = img.size
        if isinstance(self.output_size, int):
            if w > h:
                new_w, new_h = self.output_size * w / h, self.output_size
            else:
                new_w, new_h = self.output_size, self.output_size * h / w
        else:
            new_w, new_h = self.output_size

        img = f.resize(img, self.output_size, self.interpolation)
        new_w, new_h = int(new_w), int(new_h)

        ratio = w / new_w
        bbox = bbox / ratio
        if pcd is not None:
            pcd = f.resize(pcd, self.output_size, self.interpolation)
        return img, bbox, pcd


class RandomVerticalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """

    def __call__(self, img, bbox, pcd=None):
        w, h = img.size
        if random.random() < 0.5:
            img = f.vflip(img)
            bbox[:, 1] = h - np.copy(bbox[:, 1])
            if pcd is not None:
                pcd = f.vflip(pcd)
        return img, bbox, pcd


class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """

    def __call__(self, img, bbox, pcd=None):
        w, h = img.size
        if random.random() < 0.5:
            img = f.hflip(img)
            bbox[:, 0] = w - np.copy(bbox[:, 0])
            if pcd is not None:
                pcd = f.hflip(pcd)
        return img, bbox, pcd

class PCDtoRGB(object):
    """
    Convert single-channel PCD PIL image to three-channel RGB-like image
    """
    def __call__(self, img, bbox, pcd):
        _pcd = pcd
        pcd = Image.new("RGB", _pcd.size)
        pcd.paste(_pcd)
        return img, bbox, pcd

class ToTensor(object):
    """
    Convert PIL image and bounding box to Tensor objects.
    """
    def __call__(self, img, bbox, pcd=None):
        img = f.to_tensor(img)
        bbox = torch.from_numpy(bbox.copy())
        if pcd is not None:
            pcd = f.to_tensor(pcd)
        return img, bbox, pcd
