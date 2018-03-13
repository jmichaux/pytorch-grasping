import os
import glob
import random

import numpy as np
import pandas as pd

from PIL import Image
from torch.utils.data.dataset import Dataset
from scipy.misc import bytescale



class CornellGraspingDataset(Dataset):
    def __init__(self, csv_file, root_dir, use_depth=True, shuffle_channels=True, sixd_grasp=True, transform=None):
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
        self.sixd_grasp = sixd_grasp
        
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
            img[:, :, 2] = pcd
            
        # transform
        if self.transform:
            img, target = self.transform(img, target)
            img = self.transform(img)
        
        # original target has [col, row] cartesion coordinates
        # transformed target has [row, col] numpy array coordinates
        y1, x1 = target[0]
        y2, x2 = target[1]
        y3, x3 = target[2]
        y4, x4 = target[3]
        
        # calculate x, y, h, w, theta of bounding box
        x = (x1 + x3) / 2
        y = (y1 + y3) / 2
        h = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        w = np.sqrt((x1 - x4)**2 + (y1 - y4)**2)
        theta = np.arctan((y1 - y2) / (x1 - x2))
        
        if self.sixd_grasp:
            target = np.array([x, y, h, w, np.cos(2*theta), np.sin(2*theta)])
        else:
            target = np.array([x, y, h, w, theta])

        return img, target