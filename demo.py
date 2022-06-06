import os
import pathlib
import time
import numpy as np
import torch
import glob
import PIL.Image as Image

from depthcvae.config.default import cfg
from depthcvae.models import make_model
from depthcvae.utils.checkpointer import CheckPointer
from depthcvae.utils.parser import get_parser
from depthcvae.utils import torch_utils
from depthcvae.data.transforms import build_transforms
from depthcvae.data.transforms.transforms import ProximityToDepth, Resize
from utils.visualization import plot_comparison, depth_to_xyz, display_pointcloud

@torch.no_grad()
def run_demo(cfg):
    model = make_model(cfg)
    model = torch_utils.to_cuda(model)

    ckpt = cfg.PRETRAINED_WEIGHTS if len(cfg.PRETRAINED_WEIGHTS) > 0 else None 
    if ckpt is None:
        raise RuntimeError("Specify file with model weights in config")
    
    demo_dir = cfg.OUTPUT_DIR + '/demo'
    if not os.path.exists(demo_dir):
        os.makedirs(demo_dir)

    checkpointer = CheckPointer(model, save_dir=cfg.OUTPUT_DIR)
    checkpointer.load(ckpt, use_latest=ckpt is None)
    weight_file = ckpt if ckpt else checkpointer.get_checkpoint_file()
    print('Loaded weights from {}'.format(weight_file))

    image_paths_photo = sorted(glob.glob(os.path.join(cfg.DEMO_RGB_PATH, '*.png')))
    image_paths_depth = sorted(glob.glob(os.path.join(cfg.DEMO_DEPTH_PATH, '*.png')))

    transforms = build_transforms(cfg, is_train=False)
    proximity_to_depth_transform = ProximityToDepth(cfg.OUTPUT.PIXEL_MEAN, cfg.OUTPUT.PIXEL_MAX)
    resize = Resize(cfg.INPUT.IMAGE_SIZE)
    
    model.eval()

    inference_time = 0
    comparisons = []
    for i, (image_path_photo, image_path_depth) in enumerate(zip(image_paths_photo, image_paths_depth)):
        original_image = np.array(Image.open(image_path_photo).convert("RGB"))
        original_target = np.array(Image.open(image_path_depth))
        original_image, original_target = resize(image, target)

        image, target = transforms(image, target)

        image = image.unsqueeze(0)
        target = target.unsqueeze(0)
        
        image = torch_utils.to_cuda(image)
        target = torch_utils.to_cuda(target)

        start = time.time()

        # Forward
        with torch.no_grad():
            results = model(image)

        inference_time += time.time() - start

        depth = results["depth"]
        b = results["b"]
        
        depth = torch_utils.to_numpy(depth)
        b = torch_utils.to_numpy(b)

        depth = proximity_to_depth_transform(depth)

        comparisons.extend([[np.int32(original_image), original_target, depth.squeeze((0,1)), b.squeeze((0,1))]])

    plot_comparison(comparisons)
        
    print(f'Average inference time: {inference_time/len(comparisons)}')

    # Pointcloud visualization
    resize_factor_y = cfg.INPUT.IMAGE_SIZE[0]/cfg.INPUT.ORIGINAL_IMAGE_SIZE[0]
    resize_factor_x = cfg.INPUT.IMAGE_SIZE[1]/cfg.INPUT.ORIGINAL_IMAGE_SIZE[1]

    fx = cfg.INPUT.CAMERA_INTRINSICS.FX * resize_factor_x
    fy = cfg.INPUT.CAMERA_INTRINSICS.FY * resize_factor_y
    cx = cfg.INPUT.CAMERA_INTRINSICS.CX * resize_factor_x
    cy = cfg.INPUT.CAMERA_INTRINSICS.CY * resize_factor_y 
    scale = cfg.OUTPUT.RESOLUTION

    for comparison in comparisons:
        xyz = depth_to_xyz(comparison[2], fx, fy, cx, cy, scale)
        rgb = comparison[0]

        display_pointcloud(xyz, rgb/255)


def main():
    # Parse config file
    args = get_parser().parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = pathlib.Path(cfg.OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    run_demo(cfg)


if __name__ == '__main__':
    main()