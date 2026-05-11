import os
import random

files = os.listdir("data/coco_cars/images")
random.shuffle(files)

split = int(0.8*len(files))

train = files[:split]
val = files[split:]

with open("data/coco_cars/train.txt","w") as f:
    f.write("\n".join(train))

with open("data/coco_cars/val.txt","w") as f:
    f.write("\n".join(val))
