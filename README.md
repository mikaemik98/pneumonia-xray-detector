# Pneumonia Detection from Chest X-rays

## Overview

A deep learning model that detects pneumonia from chest X-ray images using
a pretrained ResNet18 neural network fine-tuned on medical imaging data.

## Dataset

- Source: Chest X-Ray Images (Pneumonia) by Paul Mooney on Kaggle
- 5,216 training images, 624 test images
- Two classes: Normal and Pneumonia
- Dataset is imbalanced — 3x more pneumonia than normal images

## Results

|                    | First Model | Second Model |
| ------------------ | ----------- | ------------ |
| Overall Accuracy   | 79.0%       | 92.5%        |
| Normal Accuracy    | 44.4%       | 84.2%        |
| Pneumonia Accuracy | 99.7%       | 97.4%        |

## Confusion Matrix (Second Model)

|                    | Predicted Normal | Predicted Pneumonia |
| ------------------ | ---------------- | ------------------- |
| Actually Normal    | 185 ✅           | 49 ⚠️               |
| Actually Pneumonia | 6 ❌             | 384 ✅              |

- 185 healthy patients correctly cleared
- 384 pneumonia cases correctly detected
- 49 healthy patients over-flagged (false positive — safer error)
- 6 pneumonia cases missed (false negative — most dangerous error)

## Key Findings

- Initial model achieved 79% accuracy but only 44.4% on normal cases
  showing that overall accuracy is misleading in medical AI
- Class imbalance (3x more pneumonia than normal images) caused the
  model to over-predict pneumonia
- Adding data augmentation and stronger class weights improved normal
  accuracy from 44.4% to 84.2% without significantly hurting pneumonia detection
- Model makes the right kind of mistakes — it over-flags rather than
  under-flags, which is the safer error in clinical settings
- Only 6 out of 390 pneumonia cases were missed (1.5% miss rate)

## What I Learned

- Deep learning with PyTorch and pretrained models (transfer learning)
- Why class imbalance is dangerous in medical datasets
- Data augmentation to improve model generalization
- Difference between overall accuracy and per-class accuracy
- Why sensitivity and specificity both matter in clinical AI
- How to interpret a confusion matrix in a medical context

## How to Run

1. Download dataset from Kaggle:
   https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
2. Extract to data/chest_xray/
3. Install dependencies:
   pip install torch torchvision matplotlib pillow scikit-learn
4. Run exploration.ipynb for data visualization
5. Run model.ipynb for training and evaluation
   - After first run the model is saved as pneumonia_model.pth
   - Future runs can skip retraining by loading the saved model

## Project Structure

pneumonia-xray-detector/
├── data/ ← download dataset here (not included)
├── exploration.ipynb
├── model.ipynb
└── README.md

## Tools Used

Python, PyTorch, torchvision, Scikit-learn, Matplotlib, Jupyter, GitHub
