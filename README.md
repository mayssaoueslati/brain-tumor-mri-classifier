# brain-tumor-mri-classifier


A convolutional neural network that classifies brain MRI scans into four categories: **glioma**, **meningioma**, **pituitary tumor**, or **no tumor**. Includes a Streamlit web app for interactive predictions.

## Overview

This project uses a CNN built from scratch with TensorFlow/Keras to classify brain tumors from MRI images. It includes a full pipeline from data preprocessing to model evaluation, plus a simple web interface for testing predictions on new images.

## Dataset

- **Source**: [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) (Kaggle)
- **Size**: 7,200 images total — 5,600 training, 1,600 testing
- **Classes**: glioma, meningioma, pituitary, notumor (balanced, 1,400 train / 400 test per class)
- The dataset is a curated combination of the figshare, SARTAJ, and Br35H brain MRI datasets

## Model Architecture

A from-scratch CNN with 4 convolutional blocks:

- Conv2D (32 filters) → MaxPooling
- Conv2D (64 filters) → MaxPooling
- Conv2D (128 filters) → MaxPooling
- Conv2D (128 filters) → MaxPooling
- Flatten → Dropout (0.5) → Dense (256) → Dense (4, softmax)

**Input**: 224×224 RGB images, rescaled to [0, 1]
**Loss**: Categorical cross-entropy
**Optimizer**: Adam

## Results

| Metric | Score |
|---|---|
| Test Accuracy | 88.9% |
| Macro F1-score | 0.89 |

**Per-class performance:**

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| Glioma | 0.89 | 0.79 | 0.84 |
| Meningioma | 0.85 | 0.85 | 0.85 |
| No Tumor | 0.87 | 0.98 | 0.92 |
| Pituitary | 0.95 | 0.94 | 0.95 |

**Key finding**: The model performs excellently on "no tumor" and "pituitary" classes, but shows more confusion between **glioma and meningioma** — a known challenge in brain tumor classification due to visual similarity between these tumor types on MRI scans.

## Project Structure

```
brain_tumor_project/
├── main.py                    # Data pipeline, model training, and evaluation
├── app.py                     # Streamlit web app for predictions
├── brain_tumor_cnn_v1.keras   # Trained model weights
└── README.md
```

## Setup & Usage

### 1. Install dependencies

```bash
pip install tensorflow kagglehub matplotlib pillow scikit-learn seaborn streamlit
```

### 2. Train the model (optional — pretrained model included)

Run `main.py` cell by cell to download the dataset, preprocess images, build the CNN, train, and evaluate.

### 3. Launch the web app

```bash
streamlit run app.py
```

Upload an MRI image and get an instant prediction with confidence scores across all four classes.

## Future Improvements

- Add transfer learning comparison (ResNet50/EfficientNet) as a benchmark against the from-scratch CNN
- Grad-CAM visualization to interpret model predictions and understand glioma/meningioma confusion
- Early stopping and data augmentation to reduce overfitting and improve generalization
- K-fold cross-validation for more robust performance estimates

## Author

Mayssa Oueslati
