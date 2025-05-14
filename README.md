
# CAPTCHA Recognition via YOLOv5 + ResNet18

This is a complete pipeline for recognizing 6-character alphanumeric CAPTCHAs using a combination of YOLOv5 (for character detection) and ResNet18 (for character classification). The system is trained on only ~300 labeled images and performs robustly on synthetic CAPTCHA images.

---

## 🔧 Requirements

- Python 3.8+
- PyTorch 1.10+
- OpenCV
- torchvision
- labelImg (for annotation)
- yolov5 (cloned from Ultralytics)

---

## 🛠️ Setup Instructions

1. **Clone YOLOv5**
   ```bash
   git clone https://github.com/ultralytics/yolov5.git
   cd yolov5
   pip install -r requirements.txt
   ```

2. **Install other dependencies**

   ```bash
   pip install torch torchvision opencv-python pillow tqdm
   ```

---

## 🚀 How to Use This Project

### 1. Annotate Characters Using labelImg

* Launch `labelImg`
* Annotate 200–400 images from `data/raw/` (out of 1000, unzip from `data/train.zip`), drawing boxes around each character
* All labels should be saved as class `char`
* Save images (e.g. 300 samples) into `data/raw_images_yolo/`
* Save corresponding YOLO-format `.txt` files into `data/raw_labels_yolo/`

---

### 2. Split into Train and Validation Sets

Run:

```bash
python split_dataset.py
```

This creates:

```
data/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
```

---

### 3. Train YOLOv5 to Detect Characters

Modify or create a `yolo_data.yaml` to point to your split dataset. Then run from the `yolov5` directory:

```bash
python train.py --img 320 --batch 16 --epochs 100 \
  --data ../yolo_data.yaml --weights yolov5s.pt \
  --name char-detector
```

This trains a detector to localize 6 character boxes per image.

---

### 4. Extract Individual Characters

Use the trained YOLO model to extract characters:

```bash
python extract_chars.py
```

This produces cropped, square, resized character images saved in `data/chars/`, named like:

```
V_0_311.png, 5_1_311.png, a_2_311.png ...
```

---

### 5. Build Classification Dataset

Run:

```bash
python build_index_dataset.py
```

This:

* Converts characters to 62 class indices (`0–9`, `A–Z`, `a–z`)
* Creates `char_dataset/train/0/`, `char_dataset/train/1/`, ..., `char_dataset/train/61/`
* Saves mapping file `idx_to_char.json`

---

### 6. Train ResNet18 for Character Classification

```bash
python train_cnn.py
```

This trains a ResNet18 model to classify each character image into 62 classes.

---

### 7. Recognize CAPTCHA

```bash
python captcha_recognizer.py
```

This script:

* Loads the YOLOv5 model for detection
* Uses the trained ResNet18 to classify each character
* Sorts boxes left to right
* Outputs the predicted 6-character string

You can also switch the script to batch mode to recognize all images in `data/test/` and save results as:

```
1    jhPvN5
2    xKa4Zm
3    7gYVbC
...
```

---

## 🧠 Model Summary

| Component      | Model    | Purpose                              |
| -------------- | -------- | ------------------------------------ |
| Detection      | YOLOv5   | Detect 6 character boxes in CAPTCHA  |
| Classification | ResNet18 | Classify each character (62 classes) |

---

## 📝 Acknowledgements

* [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5)
* PyTorch, torchvision

---

## 📂 Project Structure

```
data/
├── raw/                     # Original 1000 CAPTCHA images
├── raw_images_yolo/        # Selected 300 annotated images
├── raw_labels_yolo/        # Corresponding YOLO labels
├── images/labels/          # Split train/val sets for YOLO
├── chars/                  # Cropped character images
├── char_dataset/train/     # Classification dataset (62 folders)
```

---

## 📢 License

MIT License

---

## 👤 Author

**Barry Chao**  
Undergraduate, Department of Artificial Intelligence, School of Information, Xiamen University  
📧 Email: barryjoth@gmail.com  
💬 WeChat: zmj_418

