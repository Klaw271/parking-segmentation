"""
Обучение модели семантической сегментации Linknet на датасете припаркованных автомобилей.

Использует датасет COCO Cars, комбинированную функцию потерь (Dice + BCE),
аугментацию данных через AMP и сохранение лучшей модели по метрике IoU.

Конфигурация:
- Модель: Linknet с энкодером EfficientNet-B1
- Функция потерь: DiceLoss + BCEWithLogitsLoss
- Оптимизатор: Adam с learning rate 1e-4
- Scheduler: ReduceLROnPlateau (уменьшает LR при плато валидации)
- Эпохи: 30
- Batch size: 8
- Ускорение: AMP (Automatic Mixed Precision) для GPU

Сохраняет лучшую модель в best_linknet1.pth по метрике validation IoU.
"""

import torch
from typing import Tuple
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.COCO.ParkingDataset import ParkingDataset

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# Загрузка данных
train_ds = ParkingDataset("data/coco_cars/train.txt")
val_ds   = ParkingDataset("data/coco_cars/val.txt")

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)

# Инициализация модели
model = smp.Linknet(
    encoder_name="efficientnet-b1",
    encoder_weights="imagenet",
    classes=1,
    activation=None
).to(device)

# Функции потерь
dice_loss = smp.losses.DiceLoss(mode='binary')
bce_loss  = torch.nn.BCEWithLogitsLoss()

def loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Комбинированная функция потерь для семантической сегментации.

    Объединяет Dice Loss (улучшает IoU) и Binary Cross-Entropy Loss (улучшает стабильность).

    :param pred: предсказания модели формата (B, 1, H, W)
    :param target: целевые маски формата (B, 1, H, W)
    :return: скалярный тензор общей потери
    """
    return dice_loss(pred, target) + bce_loss(pred, target)


def iou_score(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    Вычисляет метрику Intersection over Union (IoU) для батча.

    Применяет sigmoid и пороговую бинаризацию, затем считает IoU для каждого элемента батча.

    :param pred: предсказания модели формата (B, 1, H, W)
    :param target: целевые маски формата (B, 1, H, W)
    :param threshold: порог для бинаризации (по умолчанию 0.5)
    :return: средний IoU по батчу (скалярный тензор в диапазоне 0-1)
    """
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()

    inter = (pred * target).sum(dim=(1, 2, 3))
    union = (pred + target - pred * target).sum(dim=(1, 2, 3))

    return (inter / (union + 1e-6)).mean()


# Оптимизатор и планировщик learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=2, factor=0.5
)

# Ускорение вычислений на GPU (AMP)
scaler = torch.cuda.amp.GradScaler()

# Обучающий цикл
EPOCHS = 30
best_iou = 0

for epoch in range(EPOCHS):
    # Обучение
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

    # Валидация
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

    scheduler.step(val_loss)

    # Сохранение лучшей модели по IoU
    if val_iou > best_iou:
        best_iou = val_iou
        torch.save(model.state_dict(), "best_linknet1.pth")
        print("Saved best model")

print("Training finished")