from pathlib import Path

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from dataset import RealVsScreenDataset
from model import RealVsScreenClassifier

BASE_DIR = Path(__file__).resolve().parent.parent

REAL_DIR = BASE_DIR / "dataset" / "real"
SCREEN_DIR = BASE_DIR / "dataset" / "screen"

real_images = sorted(REAL_DIR.glob("*.jpg"))
screen_images = sorted(SCREEN_DIR.glob("*.jpg"))

all_images = real_images + screen_images
labels = [0] * len(real_images) + [1] * len(screen_images)

_, val_paths, _, val_labels = train_test_split(
    all_images,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels,
)

val_dataset = RealVsScreenDataset(
    val_paths,
    val_labels,
    train=False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False
)

device = torch.device("cpu")

model = RealVsScreenClassifier().to(device)

model.load_state_dict(
    torch.load(BASE_DIR / "models" / "best_model.pth", map_location=device)
)

model.eval()

y_true = []
y_pred = []

with torch.no_grad():

    for images, labels in val_loader:

        images = images.to(device)

        outputs = model(images)

        preds = torch.argmax(outputs, dim=1)

        y_true.extend(labels.numpy())
        y_pred.extend(preds.cpu().numpy())

print("\nAccuracy :", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred))
print("Recall   :", recall_score(y_true, y_pred))
print("F1 Score :", f1_score(y_true, y_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_true, y_pred))

print("\nClassification Report")
print(classification_report(y_true, y_pred))