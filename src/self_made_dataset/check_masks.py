import os
from pathlib import Path

IMAGE_DIR = 'data/self_made_dataset/images'
MASK_DIR = 'data/self_made_dataset/masks'

# Получаем все текущие файлы в разных директориях
image_files = set(os.listdir(IMAGE_DIR))
mask_files = set(os.listdir(MASK_DIR))

print(f"Images: {len(image_files)}, Masks: {len(mask_files)}")
print("\nChecking for unmatched masks...")

# Для каждого образа ищем соответствующую маску
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

# Проверяем orphaned маски (без соответствующих образов)
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
