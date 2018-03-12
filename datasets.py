import os
import glob
import random

import numpy as np
import pandas as pd

from PIL import Image
from torch.utils.data.dataset import Dataset
from scipy.misc import bytescale



class CornellGraspingDataset(Dataset):
    def __init__(self, csv_file, root_dir, use_depth=True, shuffle_channels=True, transform=None):
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.shuffle_channels = shuffle_channels
        self.use_depth = use_depth
        # image dimensions
        self.height = 480
        self.width = 640
        # number of bounding box points
        self.nbox_pts = 4

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Get filenames
        data_subdir = join(self.root_dir,
                                   str(self.df.iloc[index, 0]))

        img_name = join(self.root_dir,
                                   data_subdir,
                                   self.df.iloc[index, 1])

        pcd_name = join(self.root_dir,
                                   data_subdir,
                                   self.df.iloc[index, 2])

        pos_name = join(self.root_dir,
                                   data_subdir,
                                   self.df.iloc[index, 3])

        # open image
        img = np.array(Image.open(img_name))

        if self.shuffle_channels:
            img = img[:, :, np.random.permutation(3)]

        # open target
        _targets = np.loadtxt(pos_name)
        num_grasps = int(len(_targets) / 4)
        idx = random.randint(0, (num_grasps - 1))
        target = _targets[self.nbox_pts * idx : self.nbox_pts * (idx + 1)]
        target = target[: , [1, 0]]

        # open depth map
        if self.use_depth:
            _pcd = np.loadtxt(pcd_name, skiprows=10, usecols=(4, 2))

            # normalize pcd_data
            # TODO: replace bytescale
            _pcd[:, 1] = bytescale(_pcd[:, 1])

            # convert pcd to array
            # row = np.floor(index / 640)
            # col = np.mod(index, 640)
            pcd = np.zeros((self.height * self.width))
            pcd[_pcd[:, 0].astype(int)] = _pcd[:, 1]
            pcd = np.reshape(pcd, (-1, self.width))

            # replace last channel with pcd
            img = np.array(img)
            img[:, :, 2] = pcd

        # transform
        if self.transform:
            img, target = self.transform(img, target)

        return img, target
