from yacs.config import CfgNode as CN

cfg = CN()

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
cfg.INPUT = CN()
cfg.INPUT.ORIGINAL_IMAGE_SIZE = [720, 1280] # (HxW)
cfg.INPUT.IMAGE_SIZE = [288, 512]           # After resizing
cfg.INPUT.IMAGE_CHANNELS = 3
cfg.INPUT.PIXEL_MEAN = [29.07345509, 59.31016239, 82.7120999]
cfg.INPUT.PIXEL_STD = [43.10742488, 54.9318506,  49.03897355]
cfg.INPUT.RGB2GRAY = False

cfg.INPUT.CAMERA_INTRINSICS = CN()
cfg.INPUT.CAMERA_INTRINSICS.FX = 3.4*1280/4.416
cfg.INPUT.CAMERA_INTRINSICS.FY = 3.4*720/2.484
cfg.INPUT.CAMERA_INTRINSICS.CX = 1280/2
cfg.INPUT.CAMERA_INTRINSICS.CY = 720/2

# -----------------------------------------------------------------------------
# OUTPUT
# -----------------------------------------------------------------------------
cfg.OUTPUT = CN()
cfg.OUTPUT.IMAGE_SIZE = [288, 512]
cfg.OUTPUT.CHANNELS = 1
cfg.OUTPUT.PIXEL_MEAN = 21490.507800145235
cfg.OUTPUT.PIXEL_STD = 2435.2189247928313
cfg.OUTPUT.PIXEL_MAX = 65535
cfg.OUTPUT.RESOLUTION = 25/(2**16-1) # [m]


# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
cfg.MODEL = CN()
cfg.MODEL.NAME = "depthcvae"
cfg.MODEL.DEVICE = "cpu"            # Used during evaluation
cfg.MODEL.NUM_OUTPUT_SCALES = 1     # Number of output scales to compute loss over (image pyramid)
cfg.MODEL.PRETRAINED = False        # Whole model is pretrained
cfg.MODEL.PRETRAINED_PATH = ""      # Path to pretrained model

### UNET ###
cfg.MODEL.UNET = CN()
cfg.MODEL.UNET.USE_SPARSE = False
cfg.MODEL.UNET.ENCODER = CN()
cfg.MODEL.UNET.ENCODER.PRETRAINED = False

### CVAE ###
cfg.MODEL.CVAE = CN()

cfg.MODEL.CVAE.LATENT = CN()
cfg.MODEL.CVAE.LATENT.DIMENSIONS = 128      # Dimension of mean vector
cfg.MODEL.CVAE.LATENT.INPUT_DIM = [9, 16]   # Size of feature map (HxW) before latent code

cfg.MODEL.CVAE.CONDITION = CN()
cfg.MODEL.CVAE.CONDITION.DIMENSIONS = 1


# -----------------------------------------------------------------------------
# TRAINER
# -----------------------------------------------------------------------------
cfg.TRAINER = CN()
cfg.TRAINER.MAX_ITER = 100000
cfg.TRAINER.BATCH_SIZE = 16
cfg.TRAINER.EPOCHS = 10
cfg.TRAINER.EVAL_STEP = -1 
cfg.TRAINER.SAVE_STEP = -1
cfg.TRAINER.LOG_STEP = 10
cfg.TRAINER.KL_MILESTONE = 0

cfg.TRAINER.OPTIMIZER = CN()
cfg.TRAINER.OPTIMIZER.TYPE = "adam"
cfg.TRAINER.OPTIMIZER.LR = 3e-4
cfg.TRAINER.OPTIMIZER.WEIGHT_DECAY = 0.0
cfg.TRAINER.OPTIMIZER.MOMENTUM = 0.9
cfg.TRAINER.OPTIMIZER.GAMMA = 1.0
cfg.TRAINER.OPTIMIZER.WARMUP_PERIOD = 0.0
cfg.TRAINER.OPTIMIZER.MILESTONES = []


# -----------------------------------------------------------------------------
# DATASET
# -----------------------------------------------------------------------------
cfg.DATASETS = CN()
# List of the dataset names for training, as present in pathscfgatalog.py
cfg.DATASETS.TRAIN = ()
# List of the dataset names for testing, as present in pathscfgatalog.py
cfg.DATASETS.TEST = ()
cfg.DATASETS.DATASET_DIR = "datasets"

# -----------------------------------------------------------------------------
# DATALOADER
# -----------------------------------------------------------------------------
cfg.DATA_LOADER = CN()
# Number of data loading threads
cfg.DATA_LOADER.NUM_WORKERS = 4
cfg.DATA_LOADER.PIN_MEMORY = True


# -----------------------------------------------------------------------------
# OUTPUT DIRECTORY
# -----------------------------------------------------------------------------
cfg.OUTPUT_DIR = "outputs"
cfg.OUTPUT_DIR_MODEL = "outputs/model"
cfg.PRETRAINED_WEIGHTS = ""

# ---------------------------------------------------------------------------- #
# TEST CONFIGURATION
# ---------------------------------------------------------------------------- #
cfg.TEST = CN()
cfg.TEST.BATCH_SIZE = 1

# -----------------------------------------------------------------------------
# DEMO CONFIGURATION
# -----------------------------------------------------------------------------
cfg.DEMO_RGB_PATH = ""
cfg.DEMO_DEPTH_PATH = ""