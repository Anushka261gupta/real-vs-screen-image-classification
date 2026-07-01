from pathlib import Path
import argparse

import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms

from model import RealVsScreenClassifier

# ---------------------------------
# Configuration
# ---------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_model.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = {
    0: "Real Camera Image",
    1: "Screen Photo"
}

# ---------------------------------
# Load Model
# ---------------------------------

model = RealVsScreenClassifier().to(DEVICE)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE)
)

model.eval()

# ---------------------------------
# Image Transform
# ---------------------------------

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------------------------
# Prediction Function
# ---------------------------------

def predict(image_path):

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(str(image_path))

    if img is None:
        raise ValueError(f"Unable to read image: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = transform(img)
    img = img.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img)
        probs = F.softmax(outputs, dim=1)

    real_prob = probs[0][0].item()
    screen_prob = probs[0][1].item()

    pred_idx = torch.argmax(probs, dim=1).item()

    prediction = CLASS_NAMES[pred_idx]
    confidence = probs[0][pred_idx].item()

    return prediction, confidence, real_prob, screen_prob


# ---------------------------------
# Main
# ---------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Real vs Screen Image Classifier"
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to input image"
    )

    args = parser.parse_args()

    try:

        prediction, confidence, real_prob, screen_prob = predict(args.image)

        print("\n" + "=" * 45)
        print("        Real vs Screen Classifier")
        print("=" * 45)
        print(f"Prediction         : {prediction}")
        print(f"Confidence         : {confidence*100:.2f}%")
        print(f"Real Probability   : {real_prob*100:.2f}%")
        print(f"Screen Probability : {screen_prob*100:.2f}%")
        print("=" * 45)

    except Exception as e:
        print(f"\nError: {e}")