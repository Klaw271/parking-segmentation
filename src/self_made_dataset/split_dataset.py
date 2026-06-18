"""
Разбиение собственного датасета припаркованных мест на train/validation подмножества.

Загружает все изображения из папки (форматы: JPG, PNG, BMP), случайно перемешивает их
и разбивает на train (80%) и validation (20%) наборы. Сохраняет списки имён файлов
в текстовые файлы с кодировкой UTF-8.

Конфигурация:
- Директория изображений: data/self_made_dataset/images
- Выходная директория: data/self_made_dataset
- Соотношение train/val: 80% / 20%
- Случайное зерно: 42 (для воспроизводимости)
- Поддерживаемые форматы: .jpg, .jpeg, .png, .bmp

Выходные файлы:
- data/self_made_dataset/train.txt (список имён файлов для обучения)
- data/self_made_dataset/valid.txt (список имён файлов для валидации)
"""

import os
import random
from pathlib import Path

IMAGE_DIR = "data/self_made_dataset/images"
OUTPUT_DIR = "data/self_made_dataset"

TRAIN_RATIO = 0.8  # 80% train, 20% val
RANDOM_SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ПОЛУЧЕНИЕ СПИСКА ИЗОБРАЖЕНИЙ
image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

image_files = [
    f.name
    for f in Path(IMAGE_DIR).iterdir()
    if f.is_file() and f.suffix.lower() in image_extensions
]

if not image_files:
    print(f"❌ No images found in {IMAGE_DIR}")
    exit(1)

print(f"✓ Found {len(image_files)} images")

# РАЗБИЕНИЕ НА TRAIN/VAL
random.seed(RANDOM_SEED)
random.shuffle(image_files)

split_idx = int(len(image_files) * TRAIN_RATIO)
train_files = image_files[:split_idx]
val_files = image_files[split_idx:]

print(f"✓ Train: {len(train_files)} images")
print(f"✓ Val:   {len(val_files)} images")

# СОХРАНЕНИЕ СПИСКОВ
train_path = os.path.join(OUTPUT_DIR, "train.txt")
val_path = os.path.join(OUTPUT_DIR, "valid.txt")

with open(train_path, "w") as f:
    f.write("\n".join(train_files))

with open(val_path, "w") as f:
    f.write("\n".join(val_files))

print(f"\n✓ Saved: {train_path}")
print(f"✓ Saved: {val_path}")
