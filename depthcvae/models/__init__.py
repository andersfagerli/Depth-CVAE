import torch

from .unet import UNet
from .cvae import CVAE
from .depthcvae import DepthCVAE

_MODELS = {
    'unet': UNet,
    'cvae': CVAE,
    'depthcvae': DepthCVAE
}

def make_model(cfg):
    if cfg.MODEL.NAME not in _MODELS:
        raise RuntimeError("Model \"{}\" in config is not supported, check models/__init__.py for the supported models".format(cfg.MODEL.NAME))
    
    model = _MODELS[cfg.MODEL.NAME](cfg)
    
    return model

def load_pretrained_params(cfg, model):
    params = dict(model.named_parameters())
    pretrained_params = torch.load(cfg.MODEL.PRETRAINED_PATH)["model"]
    
    for name in pretrained_params:
        if name in params and "cvae.mu" not in name and "cvae.logvar" not in name\
             and "cvae.decoder_in" not in name and "cvae.decoder_in" not in name: # Cannot copy code parameters, as they are dependent on input dimensions

            params[name].data.copy_(pretrained_params[name].data)
    
    del pretrained_params