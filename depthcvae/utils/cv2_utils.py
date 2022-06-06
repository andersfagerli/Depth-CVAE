import cv2
import torch
import numpy as np

def dense_to_sparse(cfg, image: torch.Tensor, dense_depth_image: torch.Tensor, npoints=1000):
    """
    Takes a dense depth map (possibly multiple) and returns a sparse map
    based on detected keypoints from a feature detector (e.g ORB)
    NB! Assumes images are on CPU
    Args:
        image: 3D tensor (C=1, H, W) or 4D tensor (N, C=1, H, W) of normalized grayscale image(s)
        dense_depth_image: 3D tensor (C=1, H, W) or 4D tensor (N, C=1, H, W) of depth image
        npoints: number of sparse points to compute
    Return:
        sparse_depth_image: 3D (C, H, W) or 4D tensor (N, C, H, W)
    """
    rgb_means = np.array(cfg.INPUT.PIXEL_MEAN, dtype=np.float32)
    rgb_stds = np.array(cfg.INPUT.PIXEL_STD, dtype=np.float32)
    rgb2gray = np.array([0.299, 0.587, 0.114])

    gray_mean = rgb_means.dot(rgb2gray)
    gray_std = rgb_stds.dot(rgb2gray)

    sparse_depth_image = torch.zeros_like(dense_depth_image) # Masks all other points to zero, which in proximity parameterization is inf
    orb = cv2.ORB_create(nfeatures=npoints)
    if len(image.size()) == 3: # 3D tensor
        image_cv = image.permute(1, 2, 0).numpy()
        image_cv = np.uint8(image_cv*gray_std + gray_mean)

        kps, _ = orb.detectAndCompute(image_cv, None)
        for kp in kps:
            (x, y) = kp.pt
            (x, y) = (int(x), int(y))
            sparse_depth_image[0, y, x] = dense_depth_image[0, y, x]
    
    else: # 4D tensor
        images_cv = image.permute(0, 2, 3, 1).numpy()
        images_cv = np.uint8(images_cv*gray_std + gray_mean)

        for i in range(images_cv.shape[0]):
            kps, _ = orb.detectAndCompute(images_cv[i], None)
            for kp in kps:
                (x, y) = kp.pt
                (x, y) = (int(x), int(y))
                sparse_depth_image[i, 0, y, x] = dense_depth_image[i, 0, y, x]
    
    return sparse_depth_image


    
if __name__ == "__main__":
    
    image = torch.randint(0, 255, (16, 1, 288, 512))
    images_cv = np.uint8(image.permute(0, 2, 3, 1).numpy())

    orb = cv2.ORB_create(nfeatures=100)

    kps, _ = orb.detectAndCompute(images_cv[0], None)

    
