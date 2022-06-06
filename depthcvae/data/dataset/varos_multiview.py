import glob
import torch
import torch.utils.data
import numpy as np
import pandas as pd
from PIL import Image

class VAROSMultiviewDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, split="train", transform=None):
        """Dataset for VAROS. Assumes RGB images as input (transform) and depth as target (target_transform)
        Args:
            data_dir: the root of the VAROS dataset camera data, the directory contains the following sub-directories:
                A, B, C, D, camM0_poses.
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform

        self.rgb_list = sorted(glob.glob(data_dir+'A/*.png'))
        self.rgb_nowater_list = sorted(glob.glob(data_dir+'B/*.png'))
        self.surface_normal_list = sorted(glob.glob(data_dir+'C/*.png'))
        self.depth_list = sorted(glob.glob(data_dir+'D/*.png'))

        self.timestamps_df = pd.read_csv(data_dir+'camM0_timestamps.csv', sep=',',header=None, comment="#")
        self.transformation_matrices_df = pd.read_csv(data_dir+'camM0_poses/camM0_poses_transformation_matrix.csv', sep=',',header=None, comment="#")
        self.euler_df = pd.read_csv(data_dir+'camM0_poses/camM0_poses_euler.csv', sep=',',header=None, comment="#")
        
        self.len = len(self.rgb_list)
        self.N = 3      # Number of images to return
        self.step = 4   # Gap between subsequent returned images, to ensure enough motion
        
        assert len(self.rgb_list) == len(self.depth_list),\
            f'Number of images in sub-directories \"A\" and \"D\" do not match'
        
        assert len(self.rgb_list) == len(self.timestamps_df),\
            f'Number of images in \"A\" do not match number of timestamps in camM0_timestamps.csv'
        
    def __getitem__(self, index):
        while index < self.step * (self.N - 1):
            # Can't use first images as target
            index = index + np.random.randint(1, self.step * (self.N - 1))
        
        while index >= 901 and index < (901 + self.step * (self.N - 1)):
            # Unfortunate hardcoding, but can't use first images after validation set split (large gap between 900 and 901)
            index = index + np.random.randint(1, self.step * (self.N - 1))

        # Target image
        rgb_images = self._read_rgb_image(index)
        depth_images = self._read_depth_image(index)
        Ts = self._read_pose(index)

        if self.transform:
            rgb_images, depth_images = self.transform(rgb_images, depth_images)
        
        # Source images (previous images)
        start = index - self.step * (self.N - 1)
        for i in reversed(range(start, index, self.step)):
            rgb_image = self._read_rgb_image(i)
            depth_image = self._read_depth_image(i)
            T = self._read_pose(i)

            if self.transform:
                rgb_image, depth_image = self.transform(rgb_image, depth_image)

            rgb_images = torch.cat([rgb_images, rgb_image], dim=0)
            depth_images = torch.cat([depth_images, depth_image], dim=0)
            Ts = torch.cat([Ts, T], dim=0)

        return rgb_images, depth_images, Ts, index

    def __len__(self):
        return self.len

    def _read_rgb_image(self, image_id):
        image_file = self.rgb_list[image_id]
        image = Image.open(image_file).convert("RGB")
        image = np.array(image)
        return image
    
    def _read_rgb_nowater_image(self, image_id):
        image_file = self.rgb_nowater_list[image_id]
        image = Image.open(image_file).convert("RGB")
        image = np.array(image)
        return image
    
    def _read_surface_normal_image(self, image_id):
        image_file = self.surface_normal_list[image_id]
        image = Image.open(image_file)
        image = np.array(image)
        return image

    def _read_depth_image(self, image_id):
        image_file = self.depth_list[image_id]
        image = Image.open(image_file)
        image = np.array(image)
        return image
    
    def _read_pose(self, index):
        timestamp = self.timestamps_df.at[index, 0]
        pose = self.transformation_matrices_df.loc[self.transformation_matrices_df[0] == timestamp].values[0][1:]
        T_wc = torch.tensor([
            [pose[0], pose[1],  pose[2],  pose[3]],
            [pose[4], pose[5],  pose[6],  pose[7]],
            [pose[8], pose[9], pose[10], pose[11]],
            [      0,       0,        0,        1]
        ])

        return T_wc.unsqueeze(0)