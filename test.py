import logging

from depthcvae.models import make_model
from depthcvae.utils.parser import get_parser
from depthcvae.config.default import cfg
from depthcvae.trainer.inference import do_evaluation
from depthcvae.utils.checkpointer import CheckPointer
from depthcvae.utils.logger import setup_logger
from depthcvae.utils import torch_utils

from depthcvae.data.dataset.varos import VAROSDataset
from depthcvae.data.build import build_transforms
from depthcvae.data.transforms import Resize, ProximityToDepth
import matplotlib.pyplot as plt
import numpy as np
import cv2
import kornia
import torch

def evaluation(cfg):
    logger = logging.getLogger("DepthCVAE.inference")

    model = make_model(cfg)
    model = torch_utils.to_cuda(model)

    ckpt = cfg.PRETRAINED_WEIGHTS if len(cfg.PRETRAINED_WEIGHTS) > 0 else None 
    if ckpt is None:
        raise RuntimeError("Specify file with model weights in config")

    checkpointer = CheckPointer(model, save_dir=cfg.OUTPUT_DIR, logger=logger)
    checkpointer.load(ckpt, use_latest=ckpt is None)
    weight_file = ckpt if ckpt else checkpointer.get_checkpoint_file()
    print('Loaded weights from {}'.format(weight_file))

    # do_evaluation(cfg, model)
    transforms = build_transforms(cfg, is_train=False)
    dataset = VAROSDataset("datasets/VAROS2021Dataset/2021-08-17_SEQ1/train/vehicle0/cam0/", split="val", transform=transforms)

    resize = Resize(cfg.INPUT.IMAGE_SIZE)
    proximity_to_depth_transform = ProximityToDepth(cfg.OUTPUT.PIXEL_MEAN, cfg.OUTPUT.PIXEL_MAX)
    index = 700

    fig, (ax1, ax2) = plt.subplots(1,2)
    #viewer = fig.add_subplot(121)
    plt.ion() # Turns interactive mode on (probably unnecessary)
    fig.show() # Initially shows the figure

    model.eval()
    for i in range(index, len(dataset)):
        images, _, _ = dataset[i]
        
        images = images.unsqueeze(0)

        # Load data to GPU if available
        images = torch_utils.to_cuda(images)

        # Forward
        results = model(images)

        depth = results["depth"]
        b = results["b"]
        
        depth = torch_utils.to_numpy(depth)

        _, depth = proximity_to_depth_transform(None, depth)

        original_image = dataset._read_rgb_image(i)
        original_image, _ = resize(original_image)

        b_normalized = torch.clone(b)
        b_normalized = b_normalized.view(b_normalized.size(0),-1)
        b_normalized /= b_normalized.max(1, keepdim=True)[0]
        b_normalized = b_normalized.view(b.size())

        edges = kornia.filters.canny(b_normalized, low_threshold=0.05, high_threshold=0.4, kernel_size=(7, 7))[1]
        
        b = torch_utils.to_numpy(b)
        b = b[0,:,:,:].transpose((1,2,0))
        b = b / b.max()

        original_image = original_image[:,:,0]*0.299 + original_image[:,:,1]*0.587 + original_image[:,:,2]*0.114
        img_blur = cv2.GaussianBlur(b*255, (7,7), 0)
        edges_cv = cv2.Canny(image=np.uint8(img_blur), threshold1=20, threshold2=150)
        im = np.uint8(original_image)*0.9 + edges_cv
        ax1.clear()
        ax1.imshow(im, cmap="gray")

        ax2.clear() 
        ax2.imshow(np.uint8(original_image)*0.9 + kornia.tensor_to_image(edges.byte())*255, cmap="gray")
        plt.pause(.01) 
        fig.canvas.draw() 

        


def main():
    args = get_parser().parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    logger = setup_logger("DepthCVAE", cfg.OUTPUT_DIR)
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    evaluation(cfg)


if __name__ == '__main__':
    main()