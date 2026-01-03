# Image Classification with Classical ML, CNNs, and Transfer Learning (PyTorch)

This project implements and compares multiple image classification approaches,
ranging from classical machine learning to deep learning and transfer learning.
All experiments are implemented in Python using PyTorch and scikit-learn.

The project covers:
- Classical machine learning (SVM and Logistic Regression) on MNIST
- Convolutional Neural Networks (CNNs) trained from scratch on MNIST
- Transfer learning using pretrained CNNs on a monkey species dataset

---

## Project Structure

.
├── SVM_LR.py
├── CNN_MNIST.py
├── Transfer_Learning.py
└── README.md

---

## 1. Classical Machine Learning on MNIST (SVM_LR.py)

This script implements multiclass classification on the MNIST dataset using
Support Vector Machines (SVM) and Logistic Regression. Dimensionality reduction
is applied before classification to improve efficiency and performance.

Supported methods:
- SVM with linear, polynomial, and RBF kernels
- Multiclass logistic regression with softmax
- Dimensionality reduction using PCA or LDA

Configuration parameters:
- kernel_type: 'linear', 'poly', or 'rbf'
- dim_reduction: 'pca' or 'lda'
- pca_dim: PCA dimension (e.g., 80)
- lda_dim: LDA dimension (maximum = number of classes − 1)

Dataset:
- MNIST is downloaded automatically using torchvision.

Run:
python SVM_LR.py

---

## 2. Convolutional Neural Networks on MNIST (CNN_MNIST.py)

This script trains and compares two convolutional neural network architectures
on the MNIST dataset.

Models:
- LeNet-5 (classic CNN)
- Custom CNN with Batch Normalization, Dropout, and Global Average Pooling

Configuration parameters:
- model_select: 'LeNet5' or 'Custom'
- epoch_count: number of training epochs (e.g., 10)

Dataset:
- MNIST is downloaded automatically and padded for LeNet compatibility.

Run:
python CNN_MNIST.py

Output:
- Training and test accuracy curves saved as PNG files.

---

## 3. Transfer Learning on Monkey Species Dataset (Transfer_Learning.py)

This script compares training a simple CNN from scratch with transfer learning
using pretrained models.

Pretrained models:
- ResNet-18
- VGG16

Training modes:
- last_FC_only: train only the final classification layer
- fine_tune: fine-tune all layers with a small learning rate

Configuration parameters:
- model_select: 'Resnet18' or 'VGG'
- train_option: 'last_FC_only' or 'fine_tune'
- epoch_count: number of training epochs (e.g., 10)

Dataset:
- The monkey species dataset must be downloaded manually.
- Dataset must follow the ImageFolder structure.

Expected directory structure:

training/
  ├── class1/
  ├── class2/
  └── ...

validation/
  ├── class1/
  ├── class2/
  └── ...

Dataset paths must be set at the top of the script:
- train_dataset_path
- val_dataset_path

Run:
python Transfer_Learning.py

Output:
- Validation accuracy plots saved as PNG files
- Accuracy values saved as JSON files

---

## Dependencies

- Python 3.7 or higher
- PyTorch
- torchvision
- scikit-learn
- matplotlib

Install dependencies with:
pip install torch torchvision scikit-learn matplotlib

---

## Notes

- MNIST is downloaded automatically by PyTorch.
- The monkey species dataset must be downloaded separately.
- GPU is used automatically if available; otherwise, training runs on CPU.
