"""
Проверка консистентности датасета: поиск несовпадений между изображениями и масками.

Сравнивает списки файлов в директориях изображений и масок:
- Ищет маски, которым нет соответствующего изображения (missing masks)
- Ищет маски без соответствующих изображений (orphaned masks)
- Поддерживает различные расширения файлов (.jpg, .jpeg, .png, .bmp для изображений)

Конфигурация:
- Директория изображений: data/self_made_dataset/images
- Директория масок: data/self_made_dataset/masks
- Формат изображений: JPG, JPEG, PNG, BMP
- Формат масок: PNG

Выходная информация:
- Количество недостающих масок
- Количество осиротелых масок
- Детальный список проблемных файлов
"""

import os
from pathlib import Path

IMAGE_DIR = 'data/self_made_dataset/images'
MASK_DIR = 'data/self_made_dataset/masks'

image_files = set(os.listdir(IMAGE_DIR))
mask_files = set(os.listdir(MASK_DIR))

print(f"Images: {len(image_files)}, Masks: {len(mask_files)}")
print("\nChecking for unmatched masks...")

# ЗАГРУЗКА СПИСКОВ ФАЙЛОВ
missing_count = 0
for img in sorted(image_files):
    if not img.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        continue
    
    # Ожидаемое имя маски
    mask_name = os.path.splitext(img)[0] + '.png'
    
    if mask_name not in mask_files:
        print(f"Missing mask for: {img} (expected: {mask_name})")
        missing_count += 1

print(f"\nTotal missing masks: {missing_count}")

# ПРОВЕРКА ОСИРОТЕЛЫХ МАСОК (БЕЗ СООТВЕТСТВУЮЩИХ ИЗОБРАЖЕНИЙ)
orphaned_count = 0
for mask in sorted(mask_files):
    if not mask.lower().endswith('.png'):
        continue
    
    img_name = os.path.splitext(mask)[0] + '.jpg'
    
    # Проверяем разные расширения образов
    found = False
    for img in image_files:
        if os.path.splitext(img)[0] == os.path.splitext(mask)[0]:
            found = True
            break
    
    if not found:
        print(f"Orphaned mask: {mask}")
        orphaned_count += 1

print(f"Total orphaned masks: {orphaned_count}")
