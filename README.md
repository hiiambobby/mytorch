# 🔥 MyTorch

<div align="center">

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![NumPy](https://img.shields.io/badge/backend-NumPy-orange.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)
![License](https://img.shields.io/badge/license-MIT-green)

**A Lightweight, Numpy-based Deep Learning Framework built from scratch.** *Mimics the PyTorch API for educational purposes and understanding the internals of Backpropagation.*

[Features](#-features) •
[Installation](#-installation) •


</div>

---

## 📖 About

**MyTorch** is a custom deep learning library designed to demystify the "black box" of modern AI frameworks. It implements a dynamic computational graph with automatic differentiation (Autograd) purely using NumPy.

It allows you to build, train, and evaluate complex neural networks (including CNNs) using an API that feels right at home for PyTorch users.

## 🚀 Features

* **🧠 Autograd Engine:** Automatic gradient calculation (`tensor.py`) using a DAG for reverse-mode differentiation.
* **🏗️ Modular Architecture:**
    * **Layers:** `Linear`, `Conv2d` (with `im2col`), `MaxPool2d`, `AvgPool2d`.
    * **Activations:** Dedicated module for `ReLU`, `Sigmoid`, etc.
    * **Losses:** Modular loss functions like `CrossEntropyLoss`.
* **📉 Optimizers:** Stochastic Gradient Descent (`SGD`) with learning rate scheduling support.
* **🛠️ Utilities:** Custom `DataLoader` for handling binary datasets (like STL-10) and `Initializers` (Xavier, He).

---

## 📦 Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/hiiambobby/mytorch.git](https://github.com/hiiambobby/mytorch.git)
    cd mytorch
    ```

2.  **Install dependencies:**
    MyTorch relies on `numpy` for computation. `torchvision` is used only for downloading datasets (like STL-10).
    ```bash
    pip install numpy matplotlib torchvision torch
    ```

---

