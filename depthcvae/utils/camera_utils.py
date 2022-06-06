import numpy as np
import torch
import random

from depthcvae.utils.torch_utils import to_cuda

class PerspectiveCamera:
    def __init__(self, cfg):
        fx = cfg.INPUT.CAMERA_INTRINSICS.FX
        fy = cfg.INPUT.CAMERA_INTRINSICS.FY
        cx = cfg.INPUT.CAMERA_INTRINSICS.CX
        cy = cfg.INPUT.CAMERA_INTRINSICS.CY
        sx = cfg.INPUT.IMAGE_SIZE[1]/cfg.INPUT.ORIGINAL_IMAGE_SIZE[1]
        sy = cfg.INPUT.IMAGE_SIZE[0]/cfg.INPUT.ORIGINAL_IMAGE_SIZE[0]
        
        self._scale = cfg.OUTPUT.RESOLUTION
        S = torch.tensor([
            [sx,  0, 0],
            [ 0, sy, 0],
            [ 0,  0, 1]
        ])
        self._K = S @ torch.tensor([[fx,  0, cx],
                                    [ 0, fy, cy],
                                    [ 0,  0,  1]]).float()
        self._K_inv = torch.linalg.inv(self._K)

        self._K = to_cuda(self._K)
        self._K_inv = to_cuda(self._K_inv)

        h, w = cfg.INPUT.IMAGE_SIZE[0], cfg.INPUT.IMAGE_SIZE[1]
        x = torch.linspace(0, w - 1, w).int()
        y = torch.linspace(0, h - 1, h).int()
        [x, y] = torch.meshgrid(x, y)
        uv = torch.vstack((x.flatten(), y.flatten(), torch.ones_like(x.flatten())))
        self.uv = to_cuda(uv)

        I = torch.tensor([
            [1, 0, 0],
            [0, 1, 0]
        ]).float()
        self.I = to_cuda(I)
        
    
    @property
    def K(self):
        return self._K
    
    @property
    def K_inv(self):
        return self._K_inv
    
    @property
    def scale(self):
        return self._scale
    
    @property
    def fx(self):
        return self.K[0,0]
    
    @property
    def fy(self):
        return self.K[1,1]
    
    @property
    def cx(self):
        return self.K[0,2]

    @property
    def cy(self):
        return self.K[1,2]

    def generate_pixels(self, h, w):
        """
        Generates 3x(hw) grid of pixels
        [0, 1, 2, ..., (w-1), 0, 1, ..., (w-1)]
        [0, 0, 0, ...,     0, 1, 1, ..., (h-1)]
        [1, 1, 1, ...,     1, 1, 1, ...,     1]
        """
        x = torch.linspace(0, w - 1, w).int()
        y = torch.linspace(0, h - 1, h).int()
        [x, y] = torch.meshgrid(x, y)
        uv = torch.vstack((x.flatten(), y.flatten(), torch.ones_like(x.flatten())))

        return to_cuda(uv)

    def to_cartesian(self, x_h):
        """
        Converts homogenous vectors into cartesian vectors
        Args:
            x_h: (4xN) tensor containing points in homogenous coordinates
        Returns:
            x_c: (3xN) tensor containing points in cartesian coordinates
        """
        return x_h[:3]/x_h[-1]
    
    def to_homogenous(self, x_c):
        """
        Converts homogenous vectors into cartesian vectors
        Args:
            x_c: (3xN) tensor containing points in cartesian coordinates
        Returns:
            x_h: (4xN) tensor containing points in homogenous coordinates
        """
        w = x_c.size()[1]
        
        return torch.cat([x_c, to_cuda(torch.ones(1,w))]).double()

    def project(self, x_c):
        """
        Projects points in the world (3D) into the image plane (2D)
        Args:
            x_c: (3xN) tensor containing 3D points in the camera frame
        Returns:
            u: (2xN) tensor containing cartesian 2D pixel coordinates
        """
        return self.I @ self.K @ x_c.float() / x_c[2].float()
    
    def backproject(self, u, z):
        """
        Projects points in the image plane (2D) into the world (3D)
        Args:
            u: (3xN) tensor containing homogeneous pixels locations
            z: (1xN) tensor containing depth at each pixel location
        Returns:
            x_c: (3xN) tensor containing 3D points in the camera frame
        """
        return self.K_inv @ u.float() * z

    def warp(self, T_ab, u_b, z_b):
        """
        Warps pixels u_b in Image b to corresponding pixels u_a in Image a
        Args:
            T_ab: (4x4) tensor relative pose between cameras a and b
            u_b: (3xN) tensor containing homogeneous pixel locations in camera b
            z_b: (1xN) tensor containing depth at each pixel location
        Returns:
            u_a: (2xN) tensor containing pixel locations in camera a
        """
        return self.project(self.to_cartesian(T_ab @ self.to_homogenous(self.backproject(u_b, z_b))))

    def bilinear_interpolation(self, x: float, y: float, I: np.ndarray):
        """
        Bilinearly interpolate from four nearest points
        Args:
            x: subpixel along x-axis
            y: subpixel along y-axis
            I: (HxW) tensor intensity image (or other map to bilinearly interpolate)
        Returns:
            fxy: bilinearly interpolated value
        """
        assert x >= 0 and y >= 0 and x < I.shape[1] and y < I.shape[0],\
            f'Subpixels are outside image dimensions'

        x1 = int(torch.floor(x))
        y1 = int(torch.floor(y))
        x2 = int(torch.ceil(x))
        y2 = int(torch.ceil(y))

        q11 = I[y1, x1]
        q12 = I[y2, x1]
        q21 = I[y1, x2]
        q22 = I[y2, x2]

        fxy1 = (x2 - x)/max((x2 - x1),1e-6) * q11 + (x - x1)/max((x2 - x1),1e-6) * q21
        fxy2 = (x2 - x)/max((x2 - x1),1e-6) * q12 + (x - x1)/max((x2 - x1),1e-6) * q22
        fxy = (y2 - y)/max((y2 - y1),1e-6) * fxy1 + (y - y1)/max((y2 - y1),1e-6) * fxy2

        return fxy
    
    def geometric_error(self, T_ab, D_a_gt, D_b_gt, D_b, depth_threshold=25, error_threshold=0.1, npoints=200):
        """
        Computes the total geometric error between two depth maps
        Args:
            T_ab: (4x4) tensor transformation matrix from b to a
            D_a: (HxW) tensor of ground truth depth values in view a (metric scale)
            D_b: (HxW) tensor of predicted depth values in view b (metric scale)
            D_b_gt: (HxW) tensor of ground truth depth values in view b (metric scale)
            depth_threshold: scalar [m] to threshold which depth values to keep (keep D < threshold)
            error_threshold: scalar [m] determining how large the error must be for it to count as an occlusion (metric scale)
            npoints: scalar number of (random) points to calculate geometric error at
        Return:
            e_g: scalar geometric error
        """
        h, w = D_a_gt.shape
        uv_b = self.uv

        x_b = self.to_homogenous(self.backproject(uv_b, D_b.flatten()))
        x_a = self.to_cartesian(T_ab @ x_b)
        D_a_projected = x_a[2,:].reshape(h,w)

        x_b_gt = self.to_homogenous(self.backproject(uv_b, D_b_gt.flatten()))
        x_a_gt = self.to_cartesian(T_ab @ x_b_gt)
        D_a_gt_projected = x_a_gt[2,:].reshape(h,w)

        uv_a = self.warp(T_ab, uv_b, D_b.flatten())
        e_g = 0
        n = 0

        sampled_points = random.sample(range(h*w), npoints)

        for i, (u_a, u_b) in enumerate(zip(uv_a.T[sampled_points], uv_b.T[sampled_points])):
            x_a, y_a = u_a[0], u_a[1]
            x_b, y_b = u_b[0], u_b[1]
            # Mask non-overlapping boundary areas
            if x_a < 0 or y_a < 0 or x_a >= (w-1) or y_a >= (h-1):
                pass
            else:
                d_a_projected = D_a_projected[y_b, x_b]
                d_a_gt_projected = D_a_gt_projected[y_b, x_b]
                # Mask depth values greater than allowed
                if d_a_gt_projected > depth_threshold:
                    pass
                else:
                    d_a = self.bilinear_interpolation(x_a, y_a, D_a_gt)
                    error = torch.abs(d_a - d_a_projected)
                    occlusion_error = torch.abs(d_a - d_a_gt_projected)
                    # Mask occlusions
                    if occlusion_error > error_threshold:
                        pass
                    else:
                        e_g += error / (d_a + d_a_projected)
                        n += 1

        return e_g/n if n > 0 else 0