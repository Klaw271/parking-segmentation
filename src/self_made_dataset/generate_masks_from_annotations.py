"""
Генерация бинарных масок припаркованных автомобилей из аннотаций COCO.

Загружает объединённый JSON файл аннотаций (в формате COCO), для каждого изображения
извлекает полигональные сегментации машин (категория ID=1), заполняет бинарные маски
чёрным фоном и белыми пикселями автомобилей, сохраняет маски в PNG формате.

Конфигурация:
- Входной файл аннотаций: data/self_made_dataset/annotations/merged_annotations.json
- Выходная директория масок: data/self_made_dataset/masks
- Категория для обработки: car (ID=1)
- Формат выходных файлов: PNG (8-бит, чёрно-белые)

Процесс:
1. Загружает JSON с изображениями и аннотациями
2. Группирует аннотации по ID изображения
3. Для каждого изображения создаёт маску нулевого размера
4. Заполняет маску белыми пикселями (255) для всех сегментаций машин
5. Сохраняет маску с тем же именем, что и изображение, но в формате PNG
"""

import os
import json
import cv2
import numpy as np

DATASET_DIR = "data/self_made_dataset"

OUTPUT_MASK_DIR = os.path.join(
    DATASET_DIR,
    "masks"
)

os.makedirs(
    OUTPUT_MASK_DIR,
    exist_ok=True
)

ANNOTATION_FILE = os.path.join(
    DATASET_DIR,
    "annotations/merged_annotations.json"
)


# ЗАГРУЗКА АННОТАЦИЙ
with open(ANNOTATION_FILE, "r") as f:
    coco = json.load(f)


images = coco["images"]
annotations = coco["annotations"]


# ГРУППИРОВКА АННОТАЦИЙ ПО ID ИЗОБРАЖЕНИЯ
ann_dict = {}

for ann in annotations:

    img_id = ann["image_id"]

    ann_dict.setdefault(
        img_id,
        []
    ).append(ann)


# ГЕНЕРАЦИЯ И СОХРАНЕНИЕ МАСОК
CAR_ID = 1

for img_info in images:

    image_id = img_info["id"]

    width = img_info["width"]
    height = img_info["height"]

    filename = img_info["file_name"]

    mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    anns = ann_dict.get(
        image_id,
        []
    )

    for ann in anns:

        if ann["category_id"] != CAR_ID:
            continue

        for seg in ann["segmentation"]:

            pts = np.array(seg).reshape(-1, 2)

            pts = pts.astype(np.int32)

            cv2.fillPoly(
                mask,
                [pts],
                255
            )

    # -----------------------------------------------------
    # SAVE
    # -----------------------------------------------------
    mask_name = os.path.splitext(filename)[0] + ".png"

    save_path = os.path.join(
        OUTPUT_MASK_DIR,
        mask_name
    )

    cv2.imwrite(
        save_path,
        mask
    )

    print("Saved:", save_path)


print()
print("DONE")