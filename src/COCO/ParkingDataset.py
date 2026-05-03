import cv2
import torch
from torch.utils.data import Dataset
import os
import numpy as np

class ParkingDataset(Dataset):

    def __init__(self, filelist):

        with open(filelist) as f:
            self.files = f.read().splitlines()

        self.image_dir="coco_cars/images"
        self.mask_dir="coco_cars/masks"

    def __len__(self):
        return len(self.files)

    def __getitem__(self,idx):

        fname=self.files[idx]

        image=cv2.imread(
            os.path.join(self.image_dir,fname)
        )

        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image=cv2.resize(image,(256,256))/255.0

        mask=cv2.imread(
            os.path.join(
               self.mask_dir,
               fname.replace(".jpg",".png")
            ),
            0
        )

        mask=cv2.resize(mask,(256,256))
        mask=(mask>0).astype(np.float32)

        image=torch.tensor(
            image,
            dtype=torch.float32
        ).permute(2,0,1)

        mask=torch.tensor(
            mask,
            dtype=torch.float32
        ).unsqueeze(0)

        return image,mask