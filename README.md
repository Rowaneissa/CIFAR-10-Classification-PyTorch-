# CNN for CIFAR-10 Classification (PyTorch)

## Project overview
This notebook implements a simple Convolutional Neural Network (CNN) in PyTorch to classify images from the CIFAR-10 dataset. It is a beginner-friendly end-to-end example that covers data loading and augmentation, model definition, training loop, evaluation, and simple visualization of predictions.

This README describes what was built (architecture, data pipeline, training & evaluation) and how to reproduce the results.

---

## What I built (concise)
- A PyTorch notebook that trains a 3-layer convolutional neural network on CIFAR-10.
- Data pipeline:
  - Training transforms: Random horizontal flip, Random crop (32, padding=4), normalization with CIFAR-10 mean/std.
  - Test transforms: ToTensor + normalization (test shown as ToTensor in the notebook).
- Model architecture (class `CIFAR`):
  - Conv2d(3 → 25) → BatchNorm2d → ReLU → MaxPool2d(2)
  - Conv2d(25 → 50) → BatchNorm2d → ReLU → Dropout(0.2) → MaxPool2d(2)
  - Conv2d(50 → 75) → BatchNorm2d → ReLU → MaxPool2d(2)
  - Flatten → Linear(1200 → 512) → Dropout(0.3) → ReLU → Linear(512 → 10)
- Training loop:
  - Loss: CrossEntropyLoss
  - Optimizer: SGD with lr=0.1
  - Batch size: 32
  - Epochs: 15
  - Moves model/data to GPU if available
- Evaluation:
  - Metrics: macro Precision and Recall (using `torchmetrics`)
  - Simple visualization showing predicted vs true labels for a few images

---

## Key hyperparameters and choices
- Batch size: 32
- Epochs: 15
- Optimizer: SGD, lr = 0.1
- Loss: CrossEntropyLoss
- Input normalization: mean = (0.4914, 0.4822, 0.4465), std = (0.2023, 0.1994, 0.2010)
- RandomHorizontalFlip and RandomCrop were used for training augmentation

---

## Results (from the notebook run)
- Training loss decreased from ~1.59 → ~0.61 over 15 epochs (training loss logged each epoch).
- Example test metrics (computed in the notebook):
  - Precision (macro): ~0.2981
  - Recall (macro): ~0.1890

These numbers indicate the model learns but has room for improvement (standard for a small, beginner CNN on CIFAR-10).

---

## How to run (reproduce the notebook)
1. Install dependencies (example):
```bash
pip install torch torchvision torchmetrics matplotlib
```

2. Run the notebook (or equivalent script) with a GPU if available. Basic steps:
```python
# load CIFAR-10, define transforms
# instantiate model: model = CIFAR()
# move to device: model.to(device)
# define loss and optimizer
# run training loop for N epochs
# evaluate with torchmetrics Precision & Recall
# visualize a few predictions
```

(See the notebook for exact code; the notebook already contains runnable code cells.)

---

## Files in the project
- `CNN_for_CIFAR_10_Classification_(PyTorch_)ipynb.ipynb` — the Jupyter/Colab notebook with full code, training logs, images, and evaluation.

---

## License & attribution
- Dataset: CIFAR-10 (Krizhevsky et al.)
- This notebook is written for learning and demonstration purposes.







---
