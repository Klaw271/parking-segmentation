import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class ParkingDataset(Dataset):

    def __init__(
        self,
        filelist,
        image_dir="data/coco_cars/images",
        mask_dir="data/coco_cars/masks",
        transform=None,
    ):
        if isinstance(filelist, str):
            with open(filelist, "r", encoding="utf-8-sig") as f:
                self.files = [line.strip() for line in f if line.strip()]
        else:
            self.files = list(filelist)

        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]

        image_path = os.path.join(self.image_dir, fname)
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256))

        mask_name = os.path.splitext(fname)[0] + ".png"
        mask_path = os.path.join(self.mask_dir, mask_name)
        mask = cv2.imread(mask_path, 0)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        mask = cv2.resize(mask, (256, 256))
        mask = (mask > 0).astype(np.float32)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        image = image.astype(np.float32) / 255.0

        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        return image, mask