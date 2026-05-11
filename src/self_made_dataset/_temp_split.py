import os
import random

IMAGE_DIR = 'data/self_made_dataset/images'
OUTPUT_DIR = 'data/self_made_dataset'
TRAIN_RATIO = 0.8
RANDOM_SEED = 42

# Get all images
image_files = []
for item in os.listdir(IMAGE_DIR):
    path = os.path.join(IMAGE_DIR, item)
    if os.path.isfile(path):
        ext = os.path.splitext(item)[1].lower()
        if ext in {'.jpg', '.jpeg', '.png', '.bmp'}:
            image_files.append(item)

print(f"Found {len(image_files)} images")

# Split
random.seed(RANDOM_SEED)
random.shuffle(image_files)

split_idx = int(len(image_files) * TRAIN_RATIO)
train_files = image_files[:split_idx]
val_files = image_files[split_idx:]

# Write with explicit UTF-8
train_path = os.path.join(OUTPUT_DIR, 'train.txt')
val_path = os.path.join(OUTPUT_DIR, 'valid.txt')

with open(train_path, 'w', encoding='utf-8') as f:
    for line in train_files:
        f.write(line + '\n')

with open(val_path, 'w', encoding='utf-8') as f:
    for line in val_files:
        f.write(line + '\n')

print(f'Train: {len(train_files)}, Val: {len(val_files)}')

# Verify
with open(train_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()[:5]
    print("First 5 files in train.txt:")
    for line in lines:
        print(f"  {line.strip()}")
