# My Neural Network Framework from Scratch

A modular deep learning framework built entirely from scratch using **Python** and **NumPy**, designed to replicate the core mechanisms of modern neural networks without relying on high-level frameworks like TensorFlow or PyTorch.

---

## 🏗️ Architecture & Code Structure

The project is structured into logical components, following standard object-oriented programming paradigms for deep learning:

1. **Data Pipeline:** Automated data downloading and preprocessing utilities.
2. **Mathematical Operations:** Activation functions, and loss formulations.
3. **Parameter Management:** Base classes for handling learnable weights and biases, function class
4. **Layer Implementation:** Modular building blocks for network construction.
5. **Network Orchestration (`Net`):** The primary container class managing forward passes, backpropagation, and training loops.

---

## 🚀 Supported Components

### Activation Functions
* **ReLU** (ReLU)
* **Sigmoid** (sigmoid)
* **Softmax** (softmax)
* **Hyperbolic Tangent** (tanh)
* **Linear** (linear)

### Loss Functions
* **Negative Log-Likelihood** (`NLL`)
* **Mean Squared Error** (`MSE`)
* **Hinge Loss** (`Hinge`)

### Neural Network Layers
* **Linear (Dense) Layer:** Standard fully-connected transformation.
* **Activation Layer:** Encapsulates non-linear transformations.
* **Batch Normalization:** Stabilizes and speeds-up learning by normalizing layer inputs.
* **Dropout:** Regularization technique to prevent overfitting.
  > *Note:* Best practices recommend placing `Batch Normalization` *before* the activation function and `Dropout` *after* the activation function, though the architecture supports flexible ordering.

### Metrics
* **Accuracy:** Computes the overall percentage of correct predictions made by the model across the entire dataset, giving a quick, high-level view of general performance.
* **Mistake Matrix (Confusion Matrix):** Generates a detailed breakdown of correct versus incorrect predictions per class. This helps identify specific classes or patterns where the model struggles (e.g., confusing digits or categories).
* **Per-Class Accuracy:** Calculates individual accuracy scores for each target class separately, preventing imbalanced datasets from masking poor performance on minority classes.

### Optimization
* **Optimizer:** Implements the **Adam optimization algorithm** utilizing **Mini-batch Gradient Descent** for efficient weight updates.
