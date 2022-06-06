import logging
import pathlib
import torch
import numpy as np
import random

from depthcvae.config.default import cfg
from depthcvae.models import make_model, load_pretrained_params
from depthcvae.trainer.trainer import start_train
from depthcvae.utils.checkpointer import CheckPointer
from depthcvae.utils.parser import get_parser
from depthcvae.utils.logger import setup_logger
from depthcvae.utils import torch_utils
from depthcvae.data.build import make_data_loader
from depthcvae.trainer.optimizer import make_optimizer
from depthcvae.trainer.scheduler import LinearMultiStepWarmUp

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

def train(cfg):
    logger = logging.getLogger('DepthCVAE.trainer')

    # Load model
    model = make_model(cfg)
    if cfg.MODEL.PRETRAINED:
        load_pretrained_params(cfg, model)
    logger.info("Running with model:\n{}".format(model))
    model = torch_utils.to_cuda(model)
    
    # Load data
    data_loader = make_data_loader(cfg, is_train=True, shuffle=True)
    
    # Optimizer
    optimizer = make_optimizer(cfg, model)

    # Learning rate scheduler
    iterations = len(data_loader)
    milestones = (np.array(cfg.TRAINER.OPTIMIZER.MILESTONES)-1)*iterations
    scheduler = LinearMultiStepWarmUp(cfg, optimizer, milestones) if cfg.TRAINER.OPTIMIZER.GAMMA < 1 or len(milestones) > 0 else None

    # Checkpointer for saving model during and after training
    checkpointer = CheckPointer(
        model, optimizer, save_dir=cfg.OUTPUT_DIR_MODEL, save_to_disk=True, logger=logger,
    )

    arguments = {"global_step": 0, "iteration": 0, "epoch": 0}
    extra_checkpoint_data = checkpointer.load()
    arguments.update(extra_checkpoint_data)
    
    # Begin training
    model = start_train(
        cfg, model, data_loader, optimizer,
        checkpointer, arguments, scheduler
    )

    return model
    

def main():
    # Parse config file
    args = get_parser().parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # Make directory for outputs (trained models, etc.)
    output_dir = pathlib.Path(cfg.OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True, parents=True)
    output_dir_model = pathlib.Path(cfg.OUTPUT_DIR_MODEL)
    output_dir_model.mkdir(exist_ok=True, parents=True)

    # Setup logger for displaying information during training
    logger = setup_logger("DepthCVAE", output_dir)
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))
    
    # Train
    model = train(cfg)


if __name__ == '__main__':
    main()