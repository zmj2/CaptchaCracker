import os
import shutil
import random
import string
from collections import defaultdict
import json

raw_data_dir = 'data/chars'
dataset_dir = 'data/char_dataset'
train_dir = os.path.join(dataset_dir, 'train')
val_dir = os.path.join(dataset_dir, 'val')
char_classes = list(string.digits + string.ascii_uppercase + string.ascii_lowercase)
char_to_index = {c: i for i, c in enumerate(char_classes)}
index_to_char = {i: c for c, i in char_to_index.items()}

validation_rate = 0.1

if not os.path.exists(train_dir):
    print("📦 正在构建数据集目录结构...")
    os.makedirs(train_dir)
    os.makedirs(val_dir)
    for i in range(len(char_classes)):
        os.makedirs(os.path.join(train_dir, str(i)), exist_ok=True)
        os.makedirs(os.path.join(val_dir, str(i)), exist_ok=True)

    label_map = defaultdict(list)
    for fname in os.listdir(raw_data_dir):
        if fname.endswith(".png"):
            label = fname.split('_')[0]
            if label in char_to_index:
                label_map[label].append(fname)

    for label_char, files in label_map.items():
        idx = char_to_index[label_char]
        random.shuffle(files)
        split_idx = int((1 - validation_rate) * len(files))
        for i, f in enumerate(files):
            src = os.path.join(raw_data_dir, f)
            dst_dir = train_dir if i < split_idx else val_dir
            dst = os.path.join(dst_dir, str(idx), f)
            shutil.copy(src, dst)

    print("✅ 数据整理完成")

with open("idx_to_char.json", "w") as f:
    json.dump(index_to_char, f)