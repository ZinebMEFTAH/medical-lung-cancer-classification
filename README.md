# Lung Cancer Classification using Deep Learning

This project implements a diagnostic tool for classifying lung cancer types from CT-scan images using PyTorch. It evaluates and compares the performance of a Multi-Layer Perceptron (MLP) and a Convolutional Neural Network (CNN).

## Overview
- **Dataset**: CT-scan images categorized into 4 classes: Adenocarcinoma, Large Cell Carcinoma, Squamous Cell Carcinoma, and Normal.
- **Goal**: Automate medical diagnosis and compare feature extraction efficiency between dense and convolutional layers.

## Architectures
- **MLP**: 3-layer architecture with Dropout (0.2) and ReLU activation.
- **CNN**: 2 Convolutional layers (Conv2d) with MaxPool2d and ReLU, followed by a classification head.

## Results
- **CNN Performance**: Reached ~77% validation accuracy with significantly fewer parameters than the MLP.
- **Key Observation**: The CNN showed superior spatial feature extraction, though both models exhibited overfitting after 15+ epochs on the small dataset.

## Tech Stack
- Python, PyTorch, Torchvision, Matplotlib, Scikit-learn.

## How to use
Run the training script:
`python sup_train.py --model CNN --epochs 10 --lr 1e-4`