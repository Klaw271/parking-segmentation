import os
import random

files = os.listdir("data/self_made_dataset/images")
random.shuffle(files)

split = int(0.8*len(files))

train = files[:split]
val = files[split:]

with open("src/self_made_dataset/train.txt","w") as f:
    f.write("\n".join(train))

with open("src/self_made_dataset/val.txt","w") as f:
    f.write("\n".join(val))
