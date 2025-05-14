import os
from torch.utils.data import Dataset
from PIL import Image


class IndexedCharDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.img_paths = []
        self.labels = []
        self.transform = transform

        for i in range(62):  
            label_dir = os.path.join(root_dir, str(i))
            for fname in os.listdir(label_dir):
                if fname.endswith(".png"):
                    self.img_paths.append(os.path.join(label_dir, fname))
                    self.labels.append(i)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]
