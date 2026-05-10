import os
import json
import cv2
import numpy as np


# =========================================================
# CONFIG
# =========================================================
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


# =========================================================
# LOAD COCO
# =========================================================
with open(ANNOTATION_FILE, "r") as f:
    coco = json.load(f)


images = coco["images"]
annotations = coco["annotations"]


# =========================================================
# GROUP ANNOTATIONS
# =========================================================
ann_dict = {}

for ann in annotations:

    img_id = ann["image_id"]

    ann_dict.setdefault(
        img_id,
        []
    ).append(ann)


# =========================================================
# CREATE MASKS
# =========================================================
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