import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter
from PIL import Image

torch.backends.cudnn.benchmark = True

# ---------------- CONFIG ---------------- #
BASE_PATH = r"D:\minorcollege\real_vs_fake\real-vs-fake"
TRAIN_DIR = os.path.join(BASE_PATH, "train")
VAL_DIR = os.path.join(BASE_PATH, "valid")

BATCH_SIZE = 16   # ✅ stable for 4GB GPU
EPOCHS = 5
NUM_WORKERS = 4


# ---------------- DATASET ---------------- #
class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.paths = []
        self.labels = []
        self.transform = transform

        for label_name, label in [("real", 0), ("fake", 1)]:
            folder = os.path.join(root_dir, label_name)

            if not os.path.exists(folder):
                continue

            for file in os.listdir(folder):
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.paths.append(os.path.join(folder, file))
                    self.labels.append(label)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.paths[idx])
        if img is None:
            raise FileNotFoundError(self.paths[idx])

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(self.labels[idx], dtype=torch.long)


# ---------------- MAIN ---------------- #
def main():

    # 📊 Graph storage
    train_losses = []
    val_accuracies = []

    # ---------------- TRANSFORMS ---------------- #
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # ---------------- DATA ---------------- #
    train_dataset = DeepfakeDataset(TRAIN_DIR, train_transform)
    val_dataset = DeepfakeDataset(VAL_DIR, val_transform)

    print("Train size:", len(train_dataset))
    print("Val size:", len(val_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # ---------------- DEVICE ---------------- #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🖥️ Using:", device)

    if torch.cuda.is_available():
        print("🔥 GPU:", torch.cuda.get_device_name(0))

    # ---------------- MODEL ---------------- #
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    # ---------------- LOSS ---------------- #
    counts = Counter(train_dataset.labels)
    print("Class distribution:", counts)

    weights = torch.tensor([
        1.0 / counts[0],
        1.0 / counts[1]
    ]).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    scaler = torch.amp.GradScaler("cuda")

    best_acc = 0

    # ---------------- TRAIN ---------------- #
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        print(f"\n🚀 Epoch {epoch+1}/{EPOCHS}")

        for i, (images, labels) in enumerate(train_loader):

            if i % 50 == 0:
                percent = (i / len(train_loader)) * 100
                print(f"📦 Batch {i}/{len(train_loader)} ({percent:.1f}%)")

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            torch.cuda.empty_cache()

        # ---------------- VALIDATION ---------------- #
        model.eval()
        preds, targets = [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)

                with torch.amp.autocast("cuda"):
                    outputs = model(images)

                pred = torch.argmax(outputs, dim=1)
                preds.extend(pred.cpu().numpy())
                targets.extend(labels.numpy())

        acc = accuracy_score(targets, preds)
        avg_loss = total_loss / len(train_loader)

        train_losses.append(avg_loss)
        val_accuracies.append(acc)

        print(f"✅ Loss: {avg_loss:.4f} | Val Acc: {acc:.4f}")
        print(classification_report(targets, preds))

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "best_resnet_model.pth")
            print("💾 Saved best model")

    # ---------------- GRAPHS ---------------- #
    plt.figure()
    plt.plot(train_losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("loss_graph.png")

    plt.figure()
    plt.plot(val_accuracies)
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig("accuracy_graph.png")

    print("\n📊 Graphs saved!")
    print("🎯 Training Completed!")


# ---------------- RUN ---------------- #
if __name__ == "__main__":
    main()