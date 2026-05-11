import os
import random
from pathlib import Path

# Простая транслитерация кириллицы
CYRILLIC_TO_LATIN = {
    'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'е': 'e', 'ё': 'yo', 'ж': 'zh',
    'з': 'z', 'и': 'i', 'й': 'y', 'к': 'k', 'л': 'l', 'м': 'm', 'н': 'n', 'о': 'o',
    'п': 'p', 'р': 'r', 'с': 's', 'т': 't', 'у': 'u', 'ф': 'f', 'х': 'h', 'ц': 'ts',
    'ч': 'ch', 'ш': 'sh', 'щ': 'sch', 'ъ': '', 'ы': 'y', 'ь': '', 'э': 'e', 'ю': 'yu', 'я': 'ya'
}

def transliterate(text):
    result = []
    for char in text.lower():
        if char in CYRILLIC_TO_LATIN:
            result.append(CYRILLIC_TO_LATIN[char])
        else:
            result.append(char)
    return ''.join(result)

IMAGE_DIR = 'data/self_made_dataset/images'
OUTPUT_DIR = 'data/self_made_dataset'

# Переименование файлов
print("Renaming files with Cyrillic characters...")
for filename in os.listdir(IMAGE_DIR):
    old_path = os.path.join(IMAGE_DIR, filename)
    if not os.path.isfile(old_path):
        continue
    
    # Проверяем, есть ли кириллица
    if any(ord(c) > 127 for c in filename):
        name, ext = os.path.splitext(filename)
        new_name = transliterate(name) + ext
        new_path = os.path.join(IMAGE_DIR, new_name)
        
        os.rename(old_path, new_path)
        print(f"  {filename} -> {new_name}")

# Пересоздание списков
print("\nCreating train/val lists...")
image_files = []
for item in os.listdir(IMAGE_DIR):
    path = os.path.join(IMAGE_DIR, item)
    if os.path.isfile(path):
        ext = os.path.splitext(item)[1].lower()
        if ext in {'.jpg', '.jpeg', '.png', '.bmp'}:
            image_files.append(item)

print(f"Found {len(image_files)} images")

random.seed(42)
random.shuffle(image_files)

split_idx = int(len(image_files) * 0.8)
train_files = image_files[:split_idx]
val_files = image_files[split_idx:]

with open(os.path.join(OUTPUT_DIR, 'train.txt'), 'w', encoding='utf-8') as f:
    for line in train_files:
        f.write(line + '\n')

with open(os.path.join(OUTPUT_DIR, 'valid.txt'), 'w', encoding='utf-8') as f:
    for line in val_files:
        f.write(line + '\n')

print(f"Train: {len(train_files)}, Val: {len(val_files)}")
print("Done!")
