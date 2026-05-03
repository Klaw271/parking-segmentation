import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.COCO.ParkingDataset import ParkingDataset

# ------------------
# DEVICE
# ------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# ------------------
# DATA
# ------------------
train_ds = ParkingDataset("./train.txt")
val_ds   = ParkingDataset("./val.txt")

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)

# ------------------
# MODEL
# ------------------
model = smp.Linknet(
    encoder_name="efficientnet-b1",
    encoder_weights="imagenet",
    classes=1,
    activation=None
).to(device)

# ------------------
# LOSS (важно!)
# ------------------
dice_loss = smp.losses.DiceLoss(mode='binary')
bce_loss  = torch.nn.BCEWithLogitsLoss()

def loss_fn(pred, target):
    return dice_loss(pred, target) + bce_loss(pred, target)

# ------------------
# OPTIMIZER
# ------------------
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=2, factor=0.5
)

# ------------------
# AMP (ускорение RTX)
# ------------------
scaler = torch.cuda.amp.GradScaler()

# ------------------
# METRIC IoU
# ------------------
def iou_score(pred, target, threshold=0.5):
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()

    inter = (pred * target).sum(dim=(1,2,3))
    union = (pred + target - pred * target).sum(dim=(1,2,3))

    return (inter / (union + 1e-6)).mean()

# ------------------
# TRAIN LOOP
# ------------------
EPOCHS = 30
best_iou = 0

for epoch in range(EPOCHS):

    # ===== TRAIN =====
    model.train()
    train_loss = 0

    loop = tqdm(train_loader, desc=f"Train {epoch+1}")

    for imgs, masks in loop:
        imgs = imgs.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            preds = model(imgs)
            loss = loss_fn(preds, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    train_loss /= len(train_loader)

    # ===== VALIDATION =====
    model.eval()
    val_loss = 0
    val_iou = 0

    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            preds = model(imgs)

            loss = loss_fn(preds, masks)
            val_loss += loss.item()

            val_iou += iou_score(preds, masks).item()

    val_loss /= len(val_loader)
    val_iou /= len(val_loader)

    print(f"\nEpoch {epoch+1}")
    print(f"Train loss: {train_loss:.4f}")
    print(f"Val loss:   {val_loss:.4f}")
    print(f"Val IoU:    {val_iou:.4f}")

    # scheduler
    scheduler.step(val_loss)

    # save best model
    if val_iou > best_iou:
        best_iou = val_iou
        torch.save(model.state_dict(), "src/models/new_best_linknet1.pth")
        print("Saved best model")

print("Training finished")
