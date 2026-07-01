# Real vs Screen Image Classification

A deep learning-based binary image classification system that determines whether an image is:

- 📷 **Real Camera Image**
- 🖥️ **Photo of a Screen**

The project uses **transfer learning with MobileNetV3-Small** and is trained on a manually collected dataset consisting of real camera photographs and recaptured screen images.

---

## Project Overview

Images captured directly from a camera and photographs taken of digital screens often exhibit different visual characteristics such as edge sharpness, frequency patterns, reflections, and display artifacts.

This project investigates these differences and builds a lightweight deep learning classifier capable of distinguishing between the two categories.

---

## Dataset

The dataset was manually collected for this assignment.

| Class | Images |
|-------|--------|
| Real Camera Images | 109 |
| Screen Photos | 108 |
| **Total** | **217** |

Images were collected using a smartphone camera under different lighting conditions and object types.

---

## Exploratory Data Analysis (EDA)

Before training the model, exploratory analysis was performed to understand the visual differences between the two classes.

The analysis included:

- Image resolution analysis
- FFT (Fast Fourier Transform) visualization
- Edge density analysis
- Visual inspection of reflections and display artifacts

These observations motivated the use of a transfer learning approach with MobileNetV3.

---

## Model Architecture

- MobileNetV3-Small (ImageNet Pretrained)
- Transfer Learning
- Cross Entropy Loss
- AdamW Optimizer

---

## Project Structure

```
real-vs-screen-image-classification/
│
├── dataset/
│   ├── real/
│   └── screen/
│
├── models/
│   └── best_model.pth
│
├── notebooks/
│   └── analysis.ipynb
│
├── report/
│
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/Anushka261gupta/real-vs-screen-image-classification.git

cd real-vs-screen-image-classification
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Training

```bash
python src/train.py
```

---

## Evaluation

```bash
python src/evaluate.py
```

Example validation results:

```
Accuracy : 1.00

Precision : 1.00

Recall : 1.00

F1 Score : 1.00
```

> These metrics were obtained on the held-out validation split used during experimentation.

---

## Prediction

Predict a single image using:

```bash
python src/predict.py --image dataset/real/IMG_5121.jpg
```

Example Output

```
Prediction         : Real Camera Image

Confidence         : 99.13%

Real Probability   : 99.13%

Screen Probability : 0.87%
```

---

## Technologies Used

- Python
- PyTorch
- Torchvision
- OpenCV
- NumPy
- Scikit-learn
- Matplotlib

---

## Future Improvements

- Larger and more diverse dataset
- Cross-validation on multiple splits
- Grad-CAM visualization
- Confidence calibration
- Deployment as a web application

---

## Author

**Anushka Gupta**

B.Tech Computer Science (AI)

Bennett University
