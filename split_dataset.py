import os
import random
import shutil

img_dir = 'data/raw_images_yolo/'
label_dir = 'data/raw_labels_yolo/'
out_img_train = 'data/images/train/'
out_img_val = 'data/images/val/'
out_lbl_train = 'data/labels/train/'
out_lbl_val = 'data/labels/val/'

os.makedirs(out_img_train, exist_ok=True)
os.makedirs(out_img_val, exist_ok=True)
os.makedirs(out_lbl_train, exist_ok=True)
os.makedirs(out_lbl_val, exist_ok=True)

images = [f for f in os.listdir(img_dir) if f.endswith('.png')]
random.shuffle(images)
n = len(images)
split = int(n * 0.9)

for i, img in enumerate(images):
    name = os.path.splitext(img)[0]
    lbl = name + '.txt'
    if i < split:
        shutil.copy(os.path.join(img_dir, img), os.path.join(out_img_train, img))
        shutil.copy(os.path.join(label_dir, lbl), os.path.join(out_lbl_train, lbl))
    else:
        shutil.copy(os.path.join(img_dir, img), os.path.join(out_img_val, img))
        shutil.copy(os.path.join(label_dir, lbl), os.path.join(out_lbl_val, lbl))

print(f"Split complete: {split} train / {n-split} val")
