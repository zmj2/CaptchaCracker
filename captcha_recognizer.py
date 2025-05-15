import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import json

image_dir = "data/test"
output_txt = "37220222203879.txt"
yolo_path = "yolov5/runs/train/char-detector/weights/best.pt"
resnet_path = "char_resnet18.pth"

with open("idx_to_char.json", "r") as f:
    idx_to_char = json.load(f)


img_size = 64
yolo_size = 320
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor()
])


cnn_model = models.resnet18(weights=None)
cnn_model.fc = nn.Linear(cnn_model.fc.in_features, 62)
cnn_model.load_state_dict(torch.load(resnet_path, map_location='cpu'))
cnn_model.eval()

yolo_model = torch.hub.load('yolov5', 'custom', path=yolo_path, source='local')
yolo_model.conf = 0.6
yolo_model.iou = 0.45

def recognize_captcha(img_path):
    img = cv2.imread(str(img_path))
    results = yolo_model(img, size=yolo_size)
    boxes = results.xyxy[0].cpu().numpy()
    boxes = [box for box in boxes if box[4] >= 0.65]

    if len(boxes) != 6:
        return "[ERR: bad box count]"

    boxes = sorted(boxes, key=lambda x: x[0])  
    chars = []

    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        crop = img[y1:y2, x1:x2]
        h, w = crop.shape[:2]
        pad_size = max(h, w)
        square = np.ones((pad_size, pad_size, 3), dtype=np.uint8) * 255
        y_off, x_off = (pad_size - h) // 2, (pad_size - w) // 2
        square[y_off:y_off + h, x_off:x_off + w] = crop
        resized_square = cv2.resize(square, (64, 64), interpolation=cv2.INTER_AREA)

        pil_img = Image.fromarray(resized_square)
        tensor = transform(pil_img).unsqueeze(0)
        with torch.no_grad():
            pred = cnn_model(tensor)
            label = pred.argmax(1).item()
            chars.append(idx_to_char[str(label)])

    return ''.join(chars)


result = recognize_captcha("data/raw/0Fm2HT_205.png")
print("Prediction results:", result)

'''
files = sorted([f for f in os.listdir(image_dir) if f.endswith(".png)])

with open(output_txt, "w", encoding="utf-8") as out:
    for i fname in enumerate(files, 1):
        img_path = os.path.join(image_dir, fname)
        pred = recognize_captcha(img_path)
        out.write(f"{i}\t{pred}\n")

print(f"Batch recognition completed, results saved to: {output_txt}")
'''

