import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
from folder import IndexedCharDataset

img_size = 64
batch_size = 64
epochs = 20
lr = 1e-3
seed = 42
use_pretrained = True

random.seed(seed)
torch.manual_seed(seed)

dataset_dir = 'data/char_dataset'
train_dir = os.path.join(dataset_dir, 'train')
val_dir = os.path.join(dataset_dir, 'val')


if __name__ == '__main__':

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor()
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])

    train_dataset = IndexedCharDataset(train_dir, transform=train_transform)
    val_dataset = IndexedCharDataset(val_dir, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if use_pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, 62)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    def evaluate(model, loader):
        model.eval()
        correct = 0
        total = 0
        total_loss = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                total_loss += loss.item() * y.size(0)
                pred = out.argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        return correct / total, total_loss / total

    print("🚀 开始训练...")
    for epoch in range(1, epochs+1):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for x, y in loop:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())

        acc, val_loss = evaluate(model, val_loader)
        print(f"🎯 验证准确率：{acc*100:.2f}% | 验证Loss: {val_loss:.4f}")

    torch.save(model.state_dict(), "char_resnet18.pth")
    print("✅ 模型训练完成并保存为 char_resnet18.pth")
