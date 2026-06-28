# Lung Cancer Classification with Deep Learning

A deep-learning pipeline that classifies lung cancer types from **CT-scan images**, built with PyTorch. It implements and compares a **Convolutional Neural Network (CNN)** and a **Multi-Layer Perceptron (MLP)** to study how each handles medical image features.

## Overview

- **Task:** classify CT-scan images into 4 classes — Adenocarcinoma, Large Cell Carcinoma, Squamous Cell Carcinoma, and Normal.
- **Goal:** automate diagnostic classification and compare dense vs. convolutional feature extraction.

## Models

- **CNN** — 2 convolutional layers (Conv2d + MaxPool2d + ReLU) followed by a classification head.
- **MLP** — 3-layer fully connected network with ReLU and dropout (0.2).

## Results

- The CNN reached **~77% validation accuracy** with far fewer parameters than the MLP, confirming the value of spatial feature extraction for imaging.
- Both models showed overfitting after ~15 epochs on the small dataset — an honest limitation, and a clear direction for future work (more data, augmentation, regularization).

## Tech Stack

- Python, PyTorch, Torchvision, scikit-learn, Matplotlib

## Run

```bash
# Train the CNN
python CnnGrayscaleModel.py

# Train the MLP
python MlpGrayscaleModel.py
```

CT-scan images are organized under `ct_scan/` (train/test splits by class).

## Project Structure

- `CnnGrayscaleModel.py` — CNN model, training, and evaluation
- `MlpGrayscaleModel.py` — MLP model, training, and evaluation
- `ct_scan/` — dataset (CT images by class)

## Notes

Deep-learning project applying computer vision to medical imaging, with a focus on comparing architectures and being transparent about dataset limitations.
