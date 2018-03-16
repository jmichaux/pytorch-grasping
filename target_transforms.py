import torch

class TargetTensor(object):
    """
    Convert targets to Tensor objects.
    """
    def __call__(self, target):
        target = torch.from_numpy(target)
        return target