import os
import shutil
import random

source_dir = "dataset/train"
validation_dir = "dataset/validation"

classes = ["MS", "No_MS"]

split_ratio = 0.8

for cls in classes:

    src_folder = os.path.join(source_dir, cls)
    val_folder = os.path.join(validation_dir, cls)

    os.makedirs(val_folder, exist_ok=True)

    images = os.listdir(src_folder)

    random.shuffle(images)

    split_index = int(len(images) * split_ratio)

    train_images = images[:split_index]
    val_images = images[split_index:]

    for img in val_images:
        src = os.path.join(src_folder, img)
        dst = os.path.join(val_folder, img)
        shutil.move(src, dst)

print("Dataset split completed")