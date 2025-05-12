import os
import cv2
import numpy as np
from tqdm import tqdm

input_dir = "data/raw/"
output_dir = "data/masks/"
os.makedirs(output_dir, exist_ok=True)

def is_valid_label(label):
    return len(label) == 6 and all(c.isalnum() for c in label)

for filename in tqdm(os.listdir(input_dir), desc="Generating masks"):
    if not filename.endswith(".png"):
        continue

    filepath = os.path.join(input_dir, filename)
    label = filename.split("_")[0]

    if not is_valid_label(label):
        print(f"[SKIP] Invalid label for {filename}")
        continue

    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    binary = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if 10 < bw < 100 and 15 < bh < 100:
            boxes.append((x, y, bw, bh))

    boxes = sorted(boxes, key=lambda b: b[0])

    for i, (x, y, bw, bh) in enumerate(boxes[:6]):
        cv2.rectangle(mask, (x, y), (x + bw, y + bh), 255, -1)  # 填白色

    cv2.imwrite(os.path.join(output_dir, filename), mask)

print("✅ 所有掩码图已生成完成。")
