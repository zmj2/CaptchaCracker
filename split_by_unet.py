import os
import sys
import cv2
import torch
import numpy as np
from torch.nn.functional import sigmoid
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
from utils.unet_model import UNet
sys.path.pop(0)

MODEL_PATH = "checkpoints/unet_epoch20.pth"
INPUT_DIR = "data/raw/"
OUTPUT_DIR = "data/chars"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels=1, out_channels=1).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

def preprocess_image(img_path):
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    image_resized = cv2.resize(image, (200, 80))
    tensor = torch.tensor(image_resized / 255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    return tensor, image

def extract_characters(mask, raw_img, filename):
    binary = (mask > 0.5).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(cnt) for cnt in contours]
    boxes = sorted(boxes, key=lambda x: x[0])  # 按 x 排序

    for i, (x, y, w, h) in enumerate(boxes[:6]):  # 最多保留6个
        char_img = raw_img[y:y+h, x:x+w]
        out_name = f"{os.path.splitext(filename)[0]}_char{i+1}.png"
        cv2.imwrite(os.path.join(OUTPUT_DIR, out_name), char_img)

for filename in os.listdir(INPUT_DIR):
    if not filename.endswith(".png"):
        continue

    img_path = os.path.join(INPUT_DIR, filename)
    input_tensor, raw_img = preprocess_image(img_path)

    with torch.no_grad():
        pred = model(input_tensor)
        pred_mask = sigmoid(pred).squeeze().cpu().numpy()
        pred_mask_resized = cv2.resize(pred_mask, (raw_img.shape[1], raw_img.shape[0]))
        extract_characters(pred_mask_resized, raw_img, filename)

print("✅ 字符分割完成，结果已保存到 data/chars/")
