import os
import json
import numpy as np
import cv2
from tqdm import tqdm
from labelme.utils import shapes_to_label
import PIL.Image
import PIL.ImageDraw

input_dir = "data/labelme_json/"
output_dir = "data/masks/"
os.makedirs(output_dir, exist_ok=True)

json_files = [f for f in os.listdir(input_dir) if f.endswith(".json")]

for json_file in tqdm(json_files, desc="Converting masks"):
    json_path = os.path.join(input_dir, json_file)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    image_path = os.path.join("data/raw/", data["imagePath"])
    image = np.array(PIL.Image.open(image_path).convert("RGB"))
    h, w = image.shape[:2]

    mask = PIL.Image.new("L", (w, h), 0)
    draw = PIL.ImageDraw.Draw(mask)

    for shape in data["shapes"]:
        points = shape["points"]
        label = shape["label"]

        if label != "char":
            continue

        if shape["shape_type"] == "rectangle":
            (x1, y1), (x2, y2) = points
            draw.rectangle([x1, y1, x2, y2], outline=255, fill=255)
        elif shape["shape_type"] == "polygon":
            draw.polygon(points, outline=255, fill=255)

    output_name = os.path.splitext(json_file)[0] + ".png"
    output_path = os.path.join(output_dir, output_name)
    mask.save(output_path)

print("✅ 所有掩码图已生成。")
