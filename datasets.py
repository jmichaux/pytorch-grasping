import random
from os.path import basename, dirname, join
import glob
import numbers
import numpy as np
import pandas as pd

from PIL import Image
from scipy.misc import bytescale

import torch
from torch.utils.data.dataset import Dataset
from torchvision.transforms import functional as f



class CornellGraspingDataset(Dataset):
    def __init__(self, csv_file, data_dir,
                 im_height = 480,
                 im_width = 640,
                 num_channels = 2,
                 nbox_pts = 4,
                 use_pcd=False,
                 concat_pcd=False,
                 fold=0,
                 split=None,
                 split_type=None,
                 pre_img_transform=None,
                 pre_pcd_transform=None,
                 co_transform=None,
                 post_img_transform=None,
                 post_pcd_transform=None,
                 target_transform=None,
                 grasp_config=5):
        
        self.df = pd.read_csv(csv_file)
        self.data_dir = data_dir
        splits = {'image': 'Image-wise', 'object': 'Object-wise'}
        if split == 'train':
            self.df = self.df[self.df[splits[split_type]] != fold]
        if split == 'val':
            self.df = self.df[self.df[splits[split_type]] == fold]
                    
        self.im_height = im_height
        self.im_width = im_width
        self.num_channels = num_channels
        self.use_pcd = use_pcd
        self.concat_pcd = concat_pcd
            
        self.pre_img_transform = pre_img_transform
        self.pre_pcd_transform = pre_pcd_transform
        self.co_transform = co_transform
        self.post_img_transform = post_img_transform
        self.post_pcd_transform = post_pcd_transform
        self.target_transform=target_transform
        
        self.nbox_pts = nbox_pts
        self.grasp_config = grasp_config

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Get filenames
        data_subdir = join(self.data_dir,
                                   str(self.df.iloc[index, 0]))

        img_name = join(self.data_dir,
                                   data_subdir,
                                   self.df.iloc[index, 1])

        pcd_name = join(self.data_dir,
                                   data_subdir,
                                   self.df.iloc[index, 2])

        pos_name = join(self.data_dir,
                                   data_subdir,
                                   self.df.iloc[index, 3])

        # open image
        img = Image.open(img_name)

        # open point cloud data map
        if self.use_pcd:
            _pcd = np.array(pd.read_csv(pcd_name, sep=" ", skiprows=10, usecols=[4,2], header=None))
            
            # normalize pcd (TODO: replace bytescale)
            _pcd[:, 0] = bytescale(_pcd[:, 0])

            # convert pcd to array
            # row = np.floor(index / 640)
            # col = np.mod(index, 640)
            pcd = np.zeros((self.im_height * self.im_width))
            pcd[_pcd[:, 1].astype(int)] = _pcd[:, 0]
            pcd = np.reshape(pcd, (-1, self.im_width))
            pcd = Image.fromarray(pcd.astype("uint8"))
        else:
            pcd = None

        # load random bounding box
        pos = np.loadtxt(pos_name)
        num_grasps = int(len(pos) / self.nbox_pts)
        nan_grasp = True
        while nan_grasp:
            grasp_idx = random.randint(0, (num_grasps - 1))
            bbox = pos[self.nbox_pts * grasp_idx : self.nbox_pts * (grasp_idx + 1)]
            nan_grasp = np.isnan(bbox).any()

        # pre-transform img
        if self.pre_img_transform:
            img = self.pre_img_transform(img)

        # pre-transform pcd
        if pcd is not None:
            if self.pre_pcd_transform:
                pcd = self.pre_pcd_transform(pcd)

        # co-transform img and pcd
        if self.co_transform:
            img, bbox, pcd = self.co_transform(img, bbox, pcd)

        # post-transform img
        if self.post_img_transform:
            img = self.post_img_transform(img)
        
        # post-transform pcd
        if pcd is not None:
            if self.post_pcd_transform:
                pcd = self.post_pcd_transform(pcd)
                pcd = pcd[:1, :, :]
            
        # concatenate img and pcd
        if self.use_pcd:
            if self.concat_pcd:
                if self.num_channels == 3:
                    img = torch.cat((img, pcd), 0)
                else:
                    img = torch.cat((img[:2, :, :], pcd), 0)
                pcd = 0

        if pcd is None:
            pcd = 0
                
        # calculate target
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
        if x2 == x1:
            theta = np.pi / 2
        else:
            theta = np.arctan((y2 - y1) / (x2 - x1))
#             theta = np.arctan2(y2 - y1, x2 - x1)

        if self.grasp_config == 3:
            target = np.array([x, y, theta])
        if self.grasp_config == 4:
            target = np.array([x/224, y/224, box_w/224, box_h/224])
        elif self.grasp_config == 5:
            target = np.array([x/224, y/224, box_w/224, box_h/224, theta])
        else:
            target = np.array([x, y, box_h, box_w, np.cos(2*theta), np.sin(2*theta)])

        if self.target_transform:
            target = self.target_transform(target)
            
        im_idx = index
        return img, target, bbox, pcd #, im_idx, grasp_idx, img_name
