"""
Объединение нескольких COCO JSON файлов аннотаций в один.

Загружает аннотации из нескольких источников, переиндексирует ID изображений и аннотаций
чтобы избежать конфликтов, объединяет все данные в единый JSON файл. Сохраняет 
объединённый датасет в формате COCO (images, annotations, categories).

Конфигурация:
- Входные JSON файлы: три датасета из папок segm1, segm2, segm3
- Выходной файл: data/self_made_dataset/annotations/merged_annotations.json

Процесс:
1. Загружает категории из первого файла (предполагаем, что они одинаковы)
2. Для каждого входного JSON:
   - Переиндексирует ID изображений (стартуя с 0)
   - Переиндексирует ID аннотаций
   - Обновляет ссылки image_id в аннотациях
3. Объединяет все в единую структуру
4. Сохраняет результат в JSON файл
"""

import json

json_files = [
    "E:/ProgramFiles/Downloads/segm1/annotations/instances_default.json",
    "E:/ProgramFiles/Downloads/segm2/annotations/instances_default.json",
    "E:/ProgramFiles/Downloads/segm3/annotations/instances_default.json"
]

# ИТОГОВЫЙ ВЫХОДНОЙ ФАЙЛ
output_file = "data/self_made_dataset/annotations/merged_annotations.json"


# ИНИЦИАЛИЗАЦИЯ СТРУКТУРЫ ОБЪЕДИНЁННЫХ АННОТАЦИЙ
merged = {
    "images": [],
    "annotations": [],
    "categories": None
}


new_image_id = 0
new_annotation_id = 0

image_id_map = {}


# ОБРАБОТКА И ПЕРЕИНДЕКСАЦИЯ КАЖДОГО ВХОДНОГО ФАЙЛА
for json_path in json_files:

    print(f"Processing: {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    # ------------------------------
    # катеогрии
    # ------------------------------
    if merged["categories"] is None:
        merged["categories"] = data["categories"]

    # ------------------------------
    # изображения
    # ------------------------------
    for img in data["images"]:

        old_id = img["id"]

        img["id"] = new_image_id

        image_id_map[(json_path, old_id)] = new_image_id

        merged["images"].append(img)

        new_image_id += 1

    # ------------------------------
    # аннотации
    # ------------------------------
    for ann in data["annotations"]:

        old_img_id = ann["image_id"]

        ann["id"] = new_annotation_id

        ann["image_id"] = image_id_map[(json_path, old_img_id)]

        merged["annotations"].append(ann)

        new_annotation_id += 1


# СОХРАНЕНИЕ ОБЪЕДИНЁННОГО ФАЙЛА
with open(output_file, "w") as f:
    json.dump(merged, f)

print()
print("DONE")
print("Images:", len(merged["images"]))
print("Annotations:", len(merged["annotations"]))
print("Saved to:", output_file)