"""
Разбиение датасета COCO на train/validation подмножества.

Загружает изображения из директории COCO, случайно перемешивает их
и разбивает на train (80%) и validation (20%) наборы. Сохраняет
списки имён файлов в текстовые файлы.

Конфигурация:
- Директория изображений: data/coco_cars/images
- Соотношение train/val: 80% / 20%

Выходные файлы:
- data/coco_cars/train.txt (список для обучения)
- data/coco_cars/val.txt (список для валидации)
"""

# =====================================================
# РАЗБИЕНИЕ НА TRAIN/VAL (COCO)
# =====================================================


files = os.listdir("data/coco_cars/images")
random.shuffle(files)

split = int(0.8*len(files))

train = files[:split]
val = files[split:]

with open("data/coco_cars/train.txt","w") as f:
    f.write("\n".join(train))

with open("data/coco_cars/val.txt","w") as f:
    f.write("\n".join(val))
