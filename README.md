# Real vs Screen Image Classification

## Overview

This project classifies whether an input image is

• Real Camera Image
• Photograph of a Screen

using MobileNetV3 Transfer Learning.

---

## Dataset

217 manually collected images

Real Images : 109

Screen Images : 108

---

## Exploratory Data Analysis

- FFT Analysis
- Edge Density Analysis
- Resolution Analysis

---

## Model

MobileNetV3-Small (ImageNet Pretrained)

---

## Results

Validation Accuracy : 100%

Precision : 100%

Recall : 100%

F1 Score : 100%

---

## Usage

python src/predict.py --image path/to/image.jpg