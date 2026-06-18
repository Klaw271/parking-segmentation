"""
Извлечение датасета припаркованных автомобилей из COCO Cars.

Загружает все изображения, содержащие категорию "car" из датасета COCO 2017 (train),
генерирует бинарные маски припаркованных автомобилей используя аннотации COCO,
сохраняет как изображения так и маски в выходную директорию.

Конфигурация:
- Входной датасет COCO: COCO 2017 train (instances_train2017.json)
- Категория для обработки: car
- Выходная директория: coco_cars (будет создана, если не существует)

Структура выхода:
- coco_cars/images/ — исходные изображения с автомобилями
- coco_cars/masks/ — бинарные маски припаркованных машин (PNG формат)
- coco_cars/train_car.txt — список имён обработанных файлов

Процесс:
1. Загружает COCO датасет через pycocotools
2. Находит все изображения с категорией "car"
3. Для каждого изображения копирует исходный файл
4. Генерирует маску, объединяя все сегментации машин
5. Сохраняет маску с тем же именем в формате PNG
6. Создаёт файл train_car.txt со списком обработанных файлов
"""

import os
import cv2
import numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm
import shutil

# ПУТИ К ДАТАСЕТУ И ВЫХОДНЫМ ФАЙЛАМ
coco_images_dir = "E:/ProgramFiles/Downloads/coco2017/train2017"
coco_ann_file = "E:/ProgramFiles/Downloads/coco2017/annotations/instances_train2017.json"

output_dir = "coco_cars"
out_img_dir = os.path.join(output_dir, "images")
out_mask_dir = os.path.join(output_dir, "masks")

os.makedirs(out_img_dir, exist_ok=True)
os.makedirs(out_mask_dir, exist_ok=True)

coco = COCO(coco_ann_file)

# КАТЕГОРИЯ "CAR"
cat_ids = coco.getCatIds(catNms=['car'])
img_ids = coco.getImgIds(catIds=cat_ids)

print(f"Found {len(img_ids)} images with cars")

# ОБРАБОТКА И СОХРАНЕНИЕ ИЗОБРАЖЕНИЙ И МАСОК
for img_id in tqdm(img_ids):

    img_info = coco.loadImgs(img_id)[0]
    file_name = img_info['file_name']

    img_path = os.path.join(coco_images_dir, file_name)

    if not os.path.exists(img_path):
        continue

    # читаем изображение
    image = cv2.imread(img_path)
    h, w = image.shape[:2]

    # создаём маску
    mask = np.zeros((h, w), dtype=np.uint8)

    ann_ids = coco.getAnnIds(imgIds=img_id, catIds=cat_ids, iscrowd=False)
    anns = coco.loadAnns(ann_ids)

    for ann in anns:
        m = coco.annToMask(ann)
        mask = np.maximum(mask, m)

    # сохраняем изображение
    out_img_path = os.path.join(out_img_dir, file_name)
    shutil.copy(img_path, out_img_path)

    # сохраняем маску
    mask_path = os.path.join(out_mask_dir, file_name.replace(".jpg", ".png"))
    cv2.imwrite(mask_path, mask * 255)

# СОХРАНЕНИЕ СПИСКА ОБРАБОТАННЫХ ФАЙЛОВ
with open(os.path.join(output_dir, "train_car.txt"), "w") as f:
    for file_name in os.listdir(out_img_dir):
        f.write(file_name + "\n")

print("Done: COCO car dataset created")
