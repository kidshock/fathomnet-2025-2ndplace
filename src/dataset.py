import os
import numpy as np
from PIL import Image, ImageFile
from tqdm.auto import tqdm

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from . import utils
from . import config

ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- Transforms ---
train_transforms_roi = transforms.Compose([
    transforms.RandomResizedCrop(config.ROI_IMAGE_SIZE, scale=(0.35, 1.0), ratio=(0.75, 1.33)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.3, 0.3, 0.2, 0.1),
    transforms.RandomRotation(15),
    transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_test_transforms_roi = transforms.Compose([
    transforms.Resize(config.ROI_IMAGE_SIZE),
    transforms.CenterCrop(config.ROI_IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

full_image_transforms = transforms.Compose([
    transforms.Resize(config.FULL_IMAGE_SIZE),
    transforms.CenterCrop(config.FULL_IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# --- Dataset Class ---
class FathomNetDataset(Dataset):
    def __init__(self, df_subset, transform_roi, transform_full, label_encoder, is_test=False):
        self.data = df_subset.reset_index(drop=True)
        self.transform_roi = transform_roi
        self.transform_full = transform_full
        self.is_test = is_test
        self.label_encoder = label_encoder

        self.roi_image_paths = self.data['path'].tolist()

        if 'full_image_path' not in self.data.columns:
            raise ValueError("DataFrame must contain a 'full_image_path' column.")
        self.full_image_paths = self.data['full_image_path'].tolist()

        if not is_test:
            self.labels = self.data["label"].tolist()
            self.label_ids = self.label_encoder.transform(self.labels)
        else:
            self.annotation_ids = self.data['annotation_id'].tolist()

    def __len__(self):
        return len(self.roi_image_paths)

    def __getitem__(self, idx):
        roi_path = self.roi_image_paths[idx]
        full_path = self.full_image_paths[idx]

        try:
            roi_img = Image.open(roi_path).convert("RGB")
            full_img = Image.open(full_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image for ROI: {roi_path} or Full: {full_path}. Error: {e}")
            dummy_roi = torch.zeros((3, *config.ROI_IMAGE_SIZE))
            dummy_full = torch.zeros((3, *config.FULL_IMAGE_SIZE))
            return (dummy_roi, dummy_full), -1

        if self.transform_roi:
            roi_img = self.transform_roi(roi_img)
        if self.transform_full:
            full_img = self.transform_full(full_img)

        images_tuple = (roi_img, full_img)

        if self.is_test:
            return images_tuple, self.annotation_ids[idx]

        return images_tuple, self.label_ids[idx]
