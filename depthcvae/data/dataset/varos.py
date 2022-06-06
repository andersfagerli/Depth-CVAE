import glob
import torch
import torch.utils.data
import numpy as np
from PIL import Image

class VAROSDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, split="train", transform=None):
        """Dataset for VAROS. Assumes RGB images as input (transform) and depth as target (target_transform)
        Args:
            data_dir: the root of the VAROS dataset camera data, the directory contains the following sub-directories:
                A, B, C, D.
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform

        self.rgb_list = sorted(glob.glob(data_dir+'A/*.png'))
        self.rgb_nowater_list = sorted(glob.glob(data_dir+'B/*.png'))
        self.surface_normal_list = sorted(glob.glob(data_dir+'C/*.png'))
        self.depth_list = sorted(glob.glob(data_dir+'D/*.png'))
        
        assert len(self.rgb_list) == len(self.depth_list),\
            f'Number of images in sub-directories \"A\" and \"D\" do not match'
        
    def __getitem__(self, index):
        
        rgb_image = self._read_rgb_image(index)
        depth_image = self._read_depth_image(index)

        if self.transform:
            rgb_image, depth_image = self.transform(rgb_image, depth_image)
        
        return rgb_image, depth_image, index

    def __len__(self):
        return len(self.rgb_list)

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
    


if __name__ == "__main__":
    dataset = VAROSDataset(data_dir='datasets/VAROS2021Dataset/2021-08-17_SEQ1/train/vehicle0/cam0/')
    
    sample_rgb, sample_depth = dataset[0]

    sample_rgb, sample_depth = dataset[0]

    num_pixels_rgb = sample_rgb.shape[0] * sample_rgb.shape[1] * len(dataset)
    num_pixels_depth = sample_depth.shape[0] * sample_depth.shape[1] * len(dataset)

    sum_rgb = np.array([0.0, 0.0, 0.0])
    sum_depth = 0.0
    for i in range(len(dataset)):
        image, depth = dataset[i]

        sum_rgb += image.sum((0,1))
        sum_depth += depth.sum()
     
    rgb_means = sum_rgb / num_pixels_rgb
    depth_mean = sum_depth / num_pixels_depth

    sum_rgb_squared_error = np.array([0.0, 0.0, 0.0])
    sum_depth_squared_error = 0.0
    for i in range(len(dataset)):
        image, depth = dataset[i]

        sum_rgb_squared_error += ((image - rgb_means)**2).sum((0,1))
        sum_depth_squared_error += ((depth - depth_mean)**2).sum()
    
    rgb_stds = np.sqrt(sum_rgb_squared_error / num_pixels_rgb)
    depth_std = np.sqrt(sum_depth_squared_error / num_pixels_depth)

    print(f'RGB means:\t{rgb_means}')
    print(f'RGB stds:\t{rgb_stds}')
    print(f'Depth mean:\t{depth_mean}')
    print(f'Depth std:\t{depth_std}')
    
    
    