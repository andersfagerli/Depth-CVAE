import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia

from depthcvae.utils.camera_utils import PerspectiveCamera

class Criterion(nn.Module):
    def __init__(self, cfg):
        super(Criterion, self).__init__()
        self.camera = PerspectiveCamera(cfg)
        self.depth_mean = cfg.OUTPUT.PIXEL_MEAN
        self.depth_max = cfg.OUTPUT.PIXEL_MAX
    
    def kl_divergence(self, mu, logvar):
        loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1), dim=0)
        return loss
    
    def reconstruction_loss(self, input, target, b=None, eps=1e-06):
        if b is not None:
            b_clamped = torch.clamp(b, min=eps)
            loss = torch.mean(torch.mean(F.l1_loss(input, target, reduction='none') / b_clamped + torch.log(b_clamped), dim=(2,3)), dim=(0,1))
        else:
            loss = F.l1_loss(input, target)
    
        return loss

    def edge_aware_smoothness(self, depth, image):
        def gradient_x(im):
            return im[:, :, :, :-1] - im[:, :, :, 1:]
        def gradient_y(im):
            return im[:, :, :-1, :] - im[:, :, 1:, :]

        dIx = gradient_x(image)
        dIy = gradient_y(image)

        dDx = gradient_x(depth)
        dDy = gradient_y(depth)

        weights_x = torch.exp(-torch.mean(torch.abs(dIx), dim=1, keepdim=True))
        weights_y = torch.exp(-torch.mean(torch.abs(dIy), dim=1, keepdim=True))

        return torch.mean(torch.abs(dDx)*weights_x) + torch.mean(torch.abs(dDy)*weights_y)
    
    def uncertainty_aware_smoothness(self, depth, b):
        def gradient_x(im):
            return im[:, :, :, :-1] - im[:, :, :, 1:]
        def gradient_y(im):
            return im[:, :, :-1, :] - im[:, :, 1:, :]
        def uncertainty_aware_edges(b):
            b_normalized = torch.clone(b)
            b_normalized = b_normalized.view(b_normalized.size(0),-1)
            b_normalized /= b_normalized.max(1, keepdim=True)[0]
            b_normalized = b_normalized.view(b.size())
            edges = kornia.filters.canny(b_normalized, low_threshold=0.05, high_threshold=0.4, kernel_size=(7, 7))[1]
            return edges
   
        dDx = gradient_x(depth)
        dDy = gradient_y(depth)

        E = uncertainty_aware_edges(b)

        weights_x = (1 - E)[:,:,:,:-1]
        weights_y = (1 - E)[:,:,:-1,:]

        return torch.mean(torch.abs(dDx)*weights_x) + torch.mean(torch.abs(dDy)*weights_y)
    
    def geometric_consistency_loss(self, prediction, target, Ts):
        def proximity_to_depth(x):
            return torch.clip(self.depth_mean * (1.0-x) / torch.clip(x, min=1e-6, max=None), min=0, max=self.depth_max)
        
        batch_size = target.shape[0]
        num_images = target.shape[1]

        loss = 0
        for b in range(batch_size):
            # Target image (current)
            D_t = proximity_to_depth(prediction[b,0,:,:]) * self.camera.scale
            D_t_gt = proximity_to_depth(target[b,0,:,:]) * self.camera.scale
            T_wt = Ts[0,0,:,:]
            
            # Compute consistency with previous images
            for i in range(1, num_images):
                D_i_gt = proximity_to_depth(target[b,i,:,:]) * self.camera.scale
                T_wi = Ts[b,i,:,:]

                T_it = torch.linalg.solve(T_wi, T_wt)

                loss += self.camera.geometric_error(T_it, D_i_gt, D_t_gt, D_t) / (num_images-1)
        loss /= batch_size
        return loss
    
    def forward(self, depth, mu, logvar, target, b=None, Ts=None, image=None):
        kl_div = self.kl_divergence(mu, logvar)
        reconstruction_loss = self.reconstruction_loss(depth, target, b)
        edge_smoothness_loss = self.edge_aware_smoothness(depth, image) if image is not None else torch.tensor(0.0).to(mu.device)
        uncertainty_smoothness_loss = self.uncertainty_aware_smoothness(depth, b) if b is not None else torch.tensor(0.0).to(mu.device)
        geometric_loss = self.geometric_consistency_loss(depth, target, Ts) if Ts is not None else torch.tensor(0.0).to(mu.device)

        loss_dict = dict(
            kl_div=kl_div,
            recon_loss=reconstruction_loss,
            edgeaware_loss=edge_smoothness_loss,
            uncertaintyaware_loss = uncertainty_smoothness_loss,
            geometric_loss=geometric_loss
        )

        return loss_dict