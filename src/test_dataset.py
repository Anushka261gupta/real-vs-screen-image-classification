from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from dataset import RealVsScreenDataset

REAL_DIR = Path(r"C:\Users\anush\SalesCode\dataset\real")
SCREEN_DIR = Path(r"C:\Users\anush\SalesCode\dataset\Screen")

real_images = sorted(REAL_DIR.glob("*.jpg"))
screen_images = sorted(SCREEN_DIR.glob("*.jpg"))

all_images = real_images + screen_images
labels = [0] * len(real_images) + [1] * len(screen_images)

train_paths, val_paths, train_labels, val_labels = train_test_split(
    all_images,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

train_dataset = RealVsScreenDataset(train_paths, train_labels, train=True)

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True
)

images, labels = next(iter(train_loader))

print(images.shape)
print(labels)