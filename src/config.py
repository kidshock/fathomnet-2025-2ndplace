import os
import torch

# Paths
DATA_DIR = "data"
TRAIN_ANNOTATIONS_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_ANNOTATIONS_CSV = os.path.join(DATA_DIR, "test.csv")
TAXONOMY_TREE_PATH = os.path.join(DATA_DIR, "taxonomy_tree.nh")
DISTANCE_MATRIX_PATH = os.path.join(DATA_DIR, "distance_matrix.csv")

CHECKPOINT_BASE_DIR = "models/kfold_checkpoints"
LOGS_DIR = "logs"

# Data & Image Settings
ROI_FOLDER_NAME = "train_images"
FULL_IMAGE_FOLDER_NAME = "images"
ROI_IMAGE_SIZE = (224, 224)
FULL_IMAGE_SIZE = (384, 384)

# Model & Training Hyperparameters
MODEL_NAME = "TwoStream-EfficientNetV2-M-S"
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
H_LOSS_WEIGHT = 0.1
LABEL_SMOOTHING = 0.1
MIXUP_ALPHA = 1.0

# K-Fold Cross-Validation
N_FOLDS = 5

# Training Loop Settings
MAX_EPOCHS = 50
BATCH_SIZE = 32
EARLY_STOPPING_PATIENCE = 15
MIN_DELTA = 0.005

# System & Reproducibility
SEED = 42
MAX_DISTANCE = 12
NUM_WORKERS = os.cpu_count() // 2 if os.cpu_count() and os.cpu_count() > 1 else 2
NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0
ACCELERATOR = "gpu" if NUM_GPUS > 0 else "cpu"
DEVICES = NUM_GPUS if NUM_GPUS > 0 else 1
STRATEGY = "ddp" if NUM_GPUS > 1 else "auto"
PRECISION = "16-mixed" if ACCELERATOR == "gpu" else 32