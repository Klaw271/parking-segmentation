import json
import os


# ==========================================
# ВХОДНЫЕ ФАЙЛЫ
# ==========================================
json_files = [
    "E:/ProgramFiles/Downloads/segm1/annotations/instances_default.json",
    "E:/ProgramFiles/Downloads/segm2/annotations/instances_default.json",
    "E:/ProgramFiles/Downloads/segm3/annotations/instances_default.json"
]

# ==========================================
# ИТОГОВЫЙ ФАЙЛ
# ==========================================
output_file = "data/self_made_dataset/annotations/merged_annotations.json"


# ==========================================
# ФИНАЛЬНЫЙ ФАЙЛ АННОТАЦИЙ
# ==========================================
merged = {
    "images": [],
    "annotations": [],
    "categories": None
}


new_image_id = 0
new_annotation_id = 0

image_id_map = {}


# ==========================================
# ОБРАБОТКА ФАЙЛОВ
# ==========================================
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


# ==========================================
# СОХРАНЕНИЕ
# ==========================================
with open(output_file, "w") as f:
    json.dump(merged, f)

print()
print("DONE")
print("Images:", len(merged["images"]))
print("Annotations:", len(merged["annotations"]))
print("Saved to:", output_file)