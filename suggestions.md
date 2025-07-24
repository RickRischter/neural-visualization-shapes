# Suggestions and Extensions

This file lists possible improvements, modifications, and extensions to the current project. These are not required for the main functionality, but can serve as inspiration for students or contributors who wish to explore further.

---

## 🧠 Model Improvements

### 1. Enable GPU Usage

Currently, the model runs entirely on CPU. You could modify the training and evaluation code to use a GPU (if available) by moving the model and tensors to `cuda` when possible.

### 2. Penalize Large Weights

Incorporate **L2 regularization** (also known as weight decay) in the optimizer to discourage overly large weights and improve generalization.

### 3. Experiment with Network Architecture

Try changing the number of hidden layers and the number of neurons per layer. For example:

- Use fewer or more neurons
- Add or remove layers
- Use different activation functions (e.g., LeakyReLU, Tanh)

Observe how these changes affect training speed, accuracy, and overfitting. This is an excellent way to understand model capacity and inductive bias.

## 🏷️ Training Objectives

### 3. Train Model to Predict Shape Area

Instead of binary classification, you could train a regression model to estimate the area covered by the shape in the image. You have to make the area labels.

### 4. Train Model to Predict RGB Color

Train a model to output the RGB color of the shape. 

---

## 🖼️ Dataset Enhancements

### 5. Add Noise to the Images

To test the model's robustness, you can apply random noise (Gaussian, salt-and-pepper, etc.) during training. You can also add random occlusions, distortions, or color jitter for more data augmentation.

---

These improvements are optional but can help expand your understanding of neural networks and experimentation with model design and training techniques.
