import os
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import albumentations as A
import cv2
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.self_made_dataset.BalancedParkingDataset import BalancedParkingDataset
from src.COCO.ParkingDataset import ParkingDataset


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE:", DEVICE)


# ----------------
# PATHS
# ----------------
OLD_TRAIN_LIST = "data/coco_cars/train.txt"
NEW_TRAIN_LIST = "data/self_made_dataset/train.txt"
NEW_VAL_LIST = "data/self_made_dataset/valid.txt"
OLD_IMAGE_DIR = "data/coco_cars/images"
OLD_MASK_DIR = "data/coco_cars/masks"
NEW_IMAGE_DIR = "data/self_made_dataset/images"
NEW_MASK_DIR = "data/self_made_dataset/masks"
CHECKPOINT = "src/models/best_linknet.pth"
SAVE_PATH = "src/models/best_linknet_finetuned.pth"

BATCH_SIZE = 4
EPOCHS = 15
LR = 1e-5
NEW_RATIO = 0.7
FREEZE_ENCODER = True


# ----------------
# AUGMENTATIONS
# ----------------
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(
        shift_limit=0.05,
        scale_limit=0.1,
        rotate_limit=25,
        border_mode=cv2.BORDER_CONSTANT,
        p=0.5,
    ),
    A.Perspective(scale=(0.02, 0.05), p=0.3),
    A.RandomBrightnessContrast(p=0.3),
])

val_transform = A.Compose([])


# ----------------
# DATA
# ----------------
old_train_ds = ParkingDataset(
    OLD_TRAIN_LIST,
    image_dir=OLD_IMAGE_DIR,
    mask_dir=OLD_MASK_DIR,
    transform=train_transform,
)

new_train_ds = ParkingDataset(
    NEW_TRAIN_LIST,
    image_dir=NEW_IMAGE_DIR,
    mask_dir=NEW_MASK_DIR,
    transform=train_transform,
)

train_ds = BalancedParkingDataset(old_train_ds, new_train_ds, new_ratio=NEW_RATIO)

val_ds = ParkingDataset(
    NEW_VAL_LIST,
    image_dir=NEW_IMAGE_DIR,
    mask_dir=NEW_MASK_DIR,
    transform=val_transform,
)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
)
val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
)


# ----------------
# MODEL
# ----------------
model = smp.Linknet(
    encoder_name="efficientnet-b1",
    encoder_weights=None,
    classes=1,
    activation=None,
)

model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model = model.to(DEVICE)
print(f"Loaded checkpoint: {CHECKPOINT}")

if FREEZE_ENCODER:
    for param in model.encoder.parameters():
        param.requires_grad = False
    print("Encoder frozen")


# ----------------
# LOSS
# ----------------

dice_loss = smp.losses.DiceLoss(mode="binary")
bce_loss = torch.nn.BCEWithLogitsLoss()


def loss_fn(pred, target):
    return 0.7 * dice_loss(pred, target) + 0.3 * bce_loss(pred, target)


# ----------------
# METRIC
# ----------------

def iou_score(pred, target, threshold=0.5):
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()

    inter = (pred * target).sum(dim=(1, 2, 3))
    union = (pred + target - pred * target).sum(dim=(1, 2, 3))
    return (inter / (union + 1e-6)).mean()


# ----------------
# OPTIMIZER
# ----------------
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR,
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2,
)

scaler = torch.cuda.amp.GradScaler()


# ----------------
# TRAIN LOOP
# ----------------
def main():
    global best_iou
    best_iou = 0.0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        loop = tqdm(train_loader, desc=f"TRAIN {epoch+1}/{EPOCHS}")
        for images, masks in loop:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                preds = model(images)
                loss = loss_fn(preds, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        val_iou = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)
                preds = model(images)
                loss = loss_fn(preds, masks)
                val_loss += loss.item()
                val_iou += iou_score(preds, masks).item()

        val_loss /= len(val_loader)
        val_iou /= len(val_loader)

        scheduler.step(val_loss)

        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print(f"Train loss: {train_loss:.4f}")
        print(f"Val loss:   {val_loss:.4f}")
        print(f"Val IoU:    {val_iou:.4f}")

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"Saved best model: {SAVE_PATH}")

    print("\nFinetuning finished. Best IoU:", best_iou)


if __name__ == "__main__":
    main()
