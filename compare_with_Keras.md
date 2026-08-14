# Neural Network Benchmark: Custom Implementation vs. Keras (TensorFlow)

This document compares a custom Neural Network (built from scratch) against a Keras/TensorFlow implementation across various architecture configurations.

## 1. Benchmark Results

| # | Architecture | Custom Acc. | Custom Time | Keras Acc. | Keras Time |
|---|---|:---:|:---:|:---:|:---:|
| 1 | No hidden layers (Single layer) | **0.926** | 4.4s | **0.926** | 20.7s |
| 2 | 1 Hidden Layer (ReLU, 256 units) | **0.981** | 142.0s | **0.980** | 40.0s |
| 3 | 1 Hidden Layer (Tanh, 256 units) | **0.980** | 141.0s | **0.981** | 40.0s |
| 4 | 1 Hidden Layer + BatchNorm (256) | **0.979** | 139.0s | **0.979** | 43.0s |
| 5 | 4 Hidden Layers (ReLU, 256 units) | **0.976** | 200.0s | **0.981** | 55.0s |

---

## 2. Conclusions & Analysis

* **Accuracy Parity:**
  * The custom network achieves nearly identical accuracy to Keras across all tested configurations, verifying the mathematical correctness of the custom forward/backward propagation and optimization logic.

* **Training Time & Framework Overhead:**
  * **Trivial Architecture (0 Hidden Layers):** The custom model is faster (`4.4s` vs `20.7s`) because it avoids high-level framework overhead (such as computational graph setup, Keras abstractions, and initialization overhead), which dominates execution time when computational workload is minimal.
  * **Multi-layer Networks:** Keras significantly outperforms the custom model (~3.5x to 3.6x speedup) due to TensorFlow's highly optimized C++ backend, hardware acceleration, and efficient BLAS matrix operations.

* **Performance Saturation & Future Improvements:**
  * Accuracy hits a ceiling around **~98.1%** across all Fully Connected (FC) setups, regardless of depth (1 to 4 layers) or activation choice (ReLU vs. Tanh).
  * This suggests that dense layers have reached their representation limit on this dataset.
  * **Next Steps:**
    * Shift to **Convolutional Neural Networks (CNNs)** to extract spatial features.
    * Experiment with higher initial layer capacities (more neurons in the first layer).
    * Apply data preprocessing and augmentation techniques to boost generalization.
