import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from tqdm import tqdm
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
from utils.unet_model import UNet
sys.path.pop(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

IMG_DIR = "data/raw"
MASK_DIR = "data/masks"
BATCH_SIZE = 4
EPOCHS = 50
LR = 1e-3
IMG_SIZE = (80, 200)

class CaptchaDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        img_files = set(f for f in os.listdir(img_dir) if f.endswith(".png"))
        mask_files = set(f for f in os.listdir(mask_dir) if f.endswith(".png"))
        self.filenames = sorted(list(img_files & mask_files))  # 取交集
        print(f"🧩 Loaded {len(self.filenames)} image-mask pairs.")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img_path = os.path.join(self.img_dir, fname)
        mask_path = os.path.join(self.mask_dir, fname)

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        image = cv2.resize(image, IMG_SIZE)
        mask = cv2.resize(mask, IMG_SIZE)

        image = image.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.float32)

        image = torch.tensor(image).unsqueeze(0)  # [1, H, W]
        mask = torch.tensor(mask).unsqueeze(0)

        return image, mask

dataset = CaptchaDataset(IMG_DIR, MASK_DIR)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = UNet(in_channels=1, out_channels=1).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

def compute_iou(pred, mask):
    pred = (torch.sigmoid(pred) > 0.5).float()
    intersection = (pred * mask).sum(dim=(1,2,3))
    union = ((pred + mask) > 0).float().sum(dim=(1,2,3))
    return (intersection / (union + 1e-6)).mean().item()

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    epoch_iou = 0

    for images, masks in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        images, masks = images.to(device), masks.to(device)

        preds = model(images)
        loss = criterion(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_iou += compute_iou(preds, masks)

    avg_loss = epoch_loss / len(loader)
    avg_iou = epoch_iou / len(loader)

    print(f"🧪 Epoch {epoch+1}: Loss = {avg_loss:.4f}, IoU = {avg_iou:.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), f"checkpoints/unet_epoch{epoch+1}.pth")
