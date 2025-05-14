import torch
import cv2
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm

img_dir = "data/raw"  
save_dir = "data/chars"
model_path = "yolov5/runs/train/char-detector/weights/best.pt"
img_size = 320

os.makedirs(save_dir, exist_ok=True)

model = torch.hub.load('yolov5', 'custom', path=model_path, source='local')
model.conf = 0.25 
model.iou = 0.45

max_size = 0
print("🔎 正在扫描最大字符尺寸...")

for img_name in tqdm(os.listdir(img_dir)):
    if not img_name.endswith(".png") or '_' not in img_name:
        continue

    char_label, _ = img_name.split('_')
    img_path = os.path.join(img_dir, img_name)
    img = cv2.imread(img_path)
    results = model(img, size=img_size)
    boxes = results.xyxy[0].cpu().numpy()

    if len(boxes) != len(char_label):
        continue

    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        h, w = y2 - y1, x2 - x1
        max_size = max(max_size, max(h, w))

print(f"✅ 最大字符尺寸为：{max_size} × {max_size}")

print("🚀 开始提取字符图像并统一尺寸...")

for img_name in tqdm(os.listdir(img_dir)):
    if not img_name.endswith(".png") or '_' not in img_name:
        continue

    char_label, suffix = img_name.split('_')
    suffix = suffix.replace(".png", "")
    img_path = os.path.join(img_dir, img_name)
    img = cv2.imread(img_path)

    results = model(img, size=img_size)
    boxes = results.xyxy[0].cpu().numpy()

    if len(boxes) != len(char_label):
        print(f"⚠️ {img_name} 检测字符数为 {len(boxes)}，应为 {len(char_label)}，跳过。")
        continue

    boxes = sorted(boxes, key=lambda x: x[0])  # 从左到右

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box[:4])
        crop = img[y1:y2, x1:x2]

        h, w = crop.shape[:2]
        pad_size = max(h, w)
        square = np.ones((pad_size, pad_size, 3), dtype=np.uint8) * 255
        y_offset = (pad_size - h) // 2
        x_offset = (pad_size - w) // 2
        square[y_offset:y_offset + h, x_offset:x_offset + w] = crop

        resized = cv2.resize(square, (max_size, max_size), interpolation=cv2.INTER_AREA)

        label_char = char_label[i]
        out_name = f"{label_char}_{i}_{suffix}.png"
        cv2.imwrite(os.path.join(save_dir, out_name), resized)

print("✅ 所有字符提取完成，保存路径：", save_dir)
