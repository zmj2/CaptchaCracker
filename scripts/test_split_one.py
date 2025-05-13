import os
import sys
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.functional import sigmoid
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
from utils.unet_model import UNet
sys.path.pop(0)
from scipy.signal import find_peaks

# ======= 配置 =======
MODEL_PATH = "checkpoints/unet_epoch50.pth"
IMAGE_PATH = "data/raw/0gQtnV_10.png"  # ← 改成你要测试的图片路径
OUTPUT_DIR = "test"
os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====== 加载模型 ======
model = UNet(in_channels=1, out_channels=1).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ====== 预处理 ======
def preprocess_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (200, 80))
    tensor = torch.tensor(img_resized / 255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    return tensor, img

# ====== 分割函数 ======
def split_by_projection(mask, raw_img, filename, num_chars=6):
    h, w = mask.shape
    projection = np.sum(mask > 0.5, axis=0).astype(np.float32)
    kernel_width = max(5, min(31, int(w / num_chars * 1.5) // 2 * 2 + 1))
    projection_smooth = cv2.GaussianBlur(projection, (kernel_width, 1), 0).flatten()

    inverted = projection_smooth.max() - projection_smooth
    threshold = inverted.mean() + inverted.std() * 0.8
    above = inverted > threshold
    segments = []
    start = None

    for i in range(len(above)):
        if above[i]:
            if start is None:
                start = i
        else:
            if start is not None:
                if i - start >= 3:  # 至少3像素长
                    segments.append((start, i - 1))
                start = None
    if start is not None and len(above) - start >= 3:
        segments.append((start, len(above) - 1))

    valid_segments = [seg for seg in segments if seg[0] > w * 0.05 and seg[1] < w * 0.95]
    valid_segments.sort(key=lambda x: x[1] - x[0], reverse=True)
    selected_segments = valid_segments[:5]
    peaks = sorted([(s + e) // 2 for s, e in selected_segments])

    rough_split = [0] + peaks + [w]
    if len(rough_split) != num_chars + 1:
        rough_split = np.linspace(0, w, num_chars + 1, dtype=int).tolist()

    refined_split = []
    for x in rough_split:
        x = int(x)
        local_start = max(x - 5, 0)
        local_end = min(x + 6, w)
        local = projection_smooth[local_start:local_end]
        best_offset = np.argmin(local)
        refined_split.append(local_start + best_offset)

    for i in range(num_chars):
        x1, x2 = refined_split[i], refined_split[i + 1]
        char_img = raw_img[:, x1:x2]
        out_name = f"{os.path.splitext(os.path.basename(filename))[0]}_char{i+1}.png"
        cv2.imwrite(os.path.join(OUTPUT_DIR, out_name), char_img)

    fig, axs = plt.subplots(3, 1, figsize=(12, 8))
    axs[0].imshow(raw_img, cmap='gray')
    axs[0].set_title("Raw Image + Split Lines")
    for x in refined_split[1:-1]:
        axs[0].axvline(x=x, color='r', linestyle='--')

    # 掩码
    axs[1].imshow(mask, cmap='gray')
    axs[1].set_title("Predicted Mask")

    # 投影曲线
    axs[2].plot(projection_smooth, label="Smoothed Projection")
    axs[2].plot(inverted, label="Inverted")
    axs[2].scatter(peaks, inverted[peaks], color='red', label="Peaks")
    axs[2].set_title("Vertical Projection + Split Points")
    axs[2].legend()

    plt.tight_layout()
    plt.show()

# ====== 执行预测 ======
tensor, raw_img = preprocess_image(IMAGE_PATH)

with torch.no_grad():
    pred = model(tensor)
    mask = sigmoid(pred).squeeze().cpu().numpy()
    mask_resized = cv2.resize(mask, (raw_img.shape[1], raw_img.shape[0]))

split_by_projection(mask_resized, raw_img, IMAGE_PATH)

print("✅ 单图分割完成，已保存为 test/char1.png ~ test/char6.png")
