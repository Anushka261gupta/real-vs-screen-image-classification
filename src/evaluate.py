from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import torch
import numpy as np
from sklearn.metrics import roc_curve, auc
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
y_scores = []
y_pred = []

with torch.no_grad():

    for images, labels in val_loader:

        images = images.to(device)

        outputs = model(images)

        preds = torch.argmax(outputs, dim=1)

        probs = torch.softmax(outputs, dim=1)

        y_true.extend(labels.numpy())
        y_pred.extend(preds.cpu().numpy())

        # Probability of Screen class (class = 1)
        y_scores.extend(probs[:, 1].cpu().numpy())


print("\nAccuracy :", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred))
print("Recall   :", recall_score(y_true, y_pred))
print("F1 Score :", f1_score(y_true, y_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_true, y_pred))

print("\nClassification Report")
print(classification_report(y_true, y_pred))

# ------------------------
# Save Confusion Matrix
# ------------------------

cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Real", "Screen"]
)

fig, ax = plt.subplots(figsize=(5,5))

disp.plot(
    cmap="Blues",
    ax=ax,
    colorbar=False
)

plt.title("Confusion Matrix")

results_dir = BASE_DIR / "results"
results_dir.mkdir(exist_ok=True)

plt.savefig(
    results_dir / "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nConfusion matrix saved.")


# ------------------------
# ROC Curve
# ------------------------

fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,6))

plt.plot(
    fpr,
    tpr,
    linewidth=2,
    label=f"AUC = {roc_auc:.3f}"
)

plt.plot([0,1],[0,1],'k--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")

plt.savefig(
    results_dir / "roc_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"\nROC AUC Score : {roc_auc:.4f}")