import random
from os.path import basename, dirname, join
import glob
import numbers
import numpy as np
import pandas as pd

from PIL import Image
import matplotlib.pyplot as plt
from scipy.misc import bytescale

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torchvision
from torchvision import models, utils
import torchvision.transforms as t
from torchvision.transforms import functional as f



class CornellGraspingDataset(Dataset):
    def __init__(self, csv_file, root_dir, 
                 use_depth=False,
                 concat_depth=False,
                 sixd_grasp=False, 
                 transform=None, 
                 co_transform=None):
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.co_transform = co_transform
        self.use_depth = use_depth
        self.concat_depth = concat_depth
        self.im_height = 480
        self.im_width = 640
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
        img = Image.open(img_name)
        
        # transform
        if self.transform:
            img = self.transform(img)
        
        # open depth map
        # TODO: speed this part up
        if self.use_depth:
            _pcd = np.loadtxt(pcd_name, skiprows=10, usecols=(4, 2))
            
            # normalize pcd_data (TODO: replace bytescale)
            _pcd[:, 1] = bytescale(_pcd[:, 1])
            
            # convert pcd to array
            # row = np.floor(index / 640) 
            # col = np.mod(index, 640) 
            pcd = np.zeros((self.im_height * self.im_width))
            pcd[_pcd[:, 0].astype(int)] = _pcd[:, 1]
            pcd = np.reshape(pcd, (-1, self.width))
            pcd = Image.fromarray(pcd.astype("uint8"))
        else:
            pcd = None
            
            
        # load random bounding box
        pos = np.loadtxt(pos_name)
        num_grasps = int(len(pos) / 4)
        idx = random.randint(0, (num_grasps - 1))
        bbox = pos[self.nbox_pts * idx : self.nbox_pts * (idx + 1)]
        
            
        # co_transforms
        if self.co_transform:
            img, bbox, pcd = self.co_transform(img, bbox, pcd)
        
        
        # img and pcd
        if self.use_depth:
            if self.concat_depth:
                img = torch.cat((img, pcd), 0)
            else:
                # replace last channel with pcd
                img = torch.cat((img[:2, :, :], pcd), 0)
   

        # corners of bbox
        x1, y1 = bbox[0]
        x2, y2 = bbox[1]
        x3, y3 = bbox[2]
        x4, y4 = bbox[3]
        
        # center of bbox
        x = (x1 + x3) / 2
        y = (y1 + y3) / 2
        
        # bbox height and width
        box_h = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        box_w = np.sqrt((x1 - x4)**2 + (y1 - y4)**2)
        
        # orientation of bbox
        theta = np.arctan((y2 - y1) / (x2 - x1))
        
        if self.sixd_grasp:
            target = np.array([x, y, h_box, w_box, np.cos(2*theta), np.sin(2*theta)])
        else:
            target = np.array([x, y, h_box, w_box, theta])

        return img, target, bbox