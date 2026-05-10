import os
import cv2
import torch
import segmentation_models_pytorch as smp
import albumentations as A

from src.ParkingDataset import ParkingDataset
from torch.utils.data import DataLoader
from tqdm import tqdm


# =========================================================
# CONFIG
# =========================================================
TRAIN_DIR = "dataset/train"
VAL_DIR   = "dataset/valid"

CHECKPOINT = "src/models/best_linknet.pth"
SAVE_DIR   = "src/models/"

BATCH_SIZE = 4
EPOCHS = 10
LR = 1e-5

FINE_TUNE = True
FREEZE_ENCODER = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(SAVE_DIR, exist_ok=True)

print("DEVICE:", DEVICE)


# =========================================================
# AUGMENTATIONS
# =========================================================
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),

    A.RandomRotate90(p=0.5),

    A.ShiftScaleRotate(
        shift_limit=0.05,
        scale_limit=0.1,
        rotate_limit=25,
        border_mode=cv2.BORDER_CONSTANT,
        p=0.5
    ),

    A.Perspective(scale=(0.02, 0.05), p=0.3),
    A.RandomBrightnessContrast(p=0.3),
])

val_transform = A.Compose([])


# =========================================================
# DATASET
# =========================================================
train_dataset = ParkingDataset(
    TRAIN_DIR,
    transform=train_transform
)

val_dataset = ParkingDataset(
    VAL_DIR,
    transform=val_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)


# =========================================================
# MODEL
# =========================================================
model = smp.Linknet(
    encoder_name="efficientnet-b1",
    encoder_weights=None,
    classes=1,
    activation=None
)

# load pretrained weights
model.load_state_dict(
    torch.load(CHECKPOINT, map_location=DEVICE)
)

print("Loaded pretrained checkpoint")

model = model.to(DEVICE)


# =========================================================
# FREEZE ENCODER (fine-tuning core)
# =========================================================
if FINE_TUNE and FREEZE_ENCODER:
    for param in model.encoder.parameters():
        param.requires_grad = False
    print("Encoder frozen")


# =========================================================
# LOSS
# =========================================================
dice_loss = smp.losses.DiceLoss(mode='binary')
bce_loss = torch.nn.BCEWithLogitsLoss()

def loss_fn(pred, target):
    return 0.7 * dice_loss(pred, target) + 0.3 * bce_loss(pred, target)


# =========================================================
# METRIC
# =========================================================
def iou_score(pred, target):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

    inter = (pred * target).sum(dim=(1, 2, 3))
    union = (pred + target - pred * target).sum(dim=(1, 2, 3))

    return (inter / (union + 1e-6)).mean()


# =========================================================
# OPTIMIZER
# =========================================================
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=2
)


# =========================================================
# AMP
# =========================================================
scaler = torch.cuda.amp.GradScaler()


# =========================================================
# TRAIN LOOP
# =========================================================
best_iou = 0

for epoch in range(EPOCHS):

    # ---------------------
    # TRAIN
    # ---------------------
    model.train()

    train_loss = 0

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


    # ---------------------
    # VALIDATION
    # ---------------------
    model.eval()

    val_loss = 0
    val_iou = 0

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


    # ---------------------
    # LOGS
    # ---------------------
    print("\n" + "=" * 50)
    print(f"EPOCH {epoch+1}")
    print(f"Train loss: {train_loss:.4f}")
    print(f"Val loss:   {val_loss:.4f}")
    print(f"Val IoU:    {val_iou:.4f}")
    print("=" * 50)


    # ---------------------
    # SAVE BEST
    # ---------------------
    if val_iou > best_iou:
        best_iou = val_iou

        save_path = os.path.join(SAVE_DIR, "best_finetuned.pth")

        torch.save(model.state_dict(), save_path)

        print("Saved best model")


print("\nFINETUNING FINISHED")
print("BEST IOU:", best_iou)