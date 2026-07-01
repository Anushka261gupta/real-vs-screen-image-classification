from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from dataset import RealVsScreenDataset
from model import RealVsScreenClassifier

# -----------------------
# Config
# -----------------------

BATCH_SIZE = 8
EPOCHS = 20
LR = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = Path(__file__).resolve().parent.parent

REAL_DIR = BASE_DIR / "dataset" / "real"
SCREEN_DIR = BASE_DIR / "dataset" / "screen"

# -----------------------
# Dataset
# -----------------------

real_images = sorted(REAL_DIR.glob("*.jpg"))
screen_images = sorted(SCREEN_DIR.glob("*.jpg"))

all_images = real_images + screen_images
labels = [0]*len(real_images) + [1]*len(screen_images)

train_paths, val_paths, train_labels, val_labels = train_test_split(
    all_images,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

train_dataset = RealVsScreenDataset(
    train_paths,
    train_labels,
    train=True
)

val_dataset = RealVsScreenDataset(
    val_paths,
    val_labels,
    train=False
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# -----------------------
# Model
# -----------------------

model = RealVsScreenClassifier().to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR
)

best_acc = 0


for epoch in range(EPOCHS):

    # ---------------- TRAIN ----------------

    model.train()

    train_loss = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    # ---------------- VALIDATION ----------------

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Loss: {train_loss:.4f} | "
        f"Val Acc: {accuracy:.4f}"
    )

    if accuracy > best_acc:

        best_acc = accuracy

        torch.save(
            model.state_dict(),
            BASE_DIR / "models" / "best_model.pth"
        )

print("\nBest Validation Accuracy:", best_acc)