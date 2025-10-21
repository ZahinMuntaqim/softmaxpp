# SoftMax++ Activation Function

## Overview
SoftMax++ is a novel activation function designed to enhance the standard Softmax function by introducing additional flexibility through three key parameters: `alpha`, `beta`, and `tau`. This function modifies the typical output of the Softmax by applying a non-linear transformation that can adjust the sharpness, smoothness, and overall behavior of the activation, providing a more controlled response for various machine learning tasks.

This repository provides an implementation of the SoftMax++ activation function in Python using `NumPy` and `PyTorch`. The function can be easily integrated into neural networks to replace or enhance the typical Softmax activation layer.

## SoftMax++ Formula

The SoftMax++ activation function, denoted as `g(z)`, is defined as follows:

\[
g(z) = \left( 1 + \alpha \cdot \tanh(\beta \cdot z) \right)^{\frac{1}{\tau}}
\]

Where:
- `z`: Input logits (values passed into the activation function)
- `alpha (α)`: Controls the magnitude of the non-linearity. Higher values of `alpha` introduce stronger deviations from linear behavior.
- `beta (β)`: Controls the sharpness of the transition. Larger values of `beta` make the transition between low and high values of `z` sharper.
- `tau (τ)`: A smoothing factor that adjusts the overall output. Smaller values of `tau` lead to sharper outputs, while larger values smooth the output more.

## Motivation

While the standard Softmax function works well for many tasks, its limitations in terms of sharpness and smoothness can hinder its performance in more complex models. SoftMax++ allows for the adjustment of these characteristics, providing a more flexible activation function that can be customized to suit the specific requirements of different machine learning problems.

## Installation

To use SoftMax++, you will need to have Python installed along with the required dependencies:

- `torch`
- `numpy`
- `matplotlib`

You can install the required dependencies using pip:

```bash
pip install torch numpy matplotlib


## 🧩 Implementation

```python
import torch
import matplotlib.pyplot as plt
import numpy as np

# Define Softmax++ non-linearity (before normalization)
def softmaxpp_g(z, alpha=1.0, beta=1.0, tau=1.0):
    return (1 + alpha * np.tanh(beta * z)) ** (1.0 / tau)

# Generate input range
z = np.linspace(-5, 5, 500)

# Plot for various parameters
plt.figure(figsize=(10, 6))

# Set of parameters to visualize effect
params = [
    {"alpha": 1.0, "beta": 1.0, "tau": 1.0},   # Mild nonlinearity
    {"alpha": 2.0, "beta": 1.0, "tau": 1.0},   # Stronger nonlinearity
    {"alpha": 1.0, "beta": 2.0, "tau": 1.0},   # Sharper transition
    {"alpha": 1.0, "beta": 1.0, "tau": 0.5},   # Sharpened output
    {"alpha": 1.0, "beta": 1.0, "tau": 2.0},   # Smoothed output
]

for p in params:
    g_vals = softmaxpp_g(z, alpha=p["alpha"], beta=p["beta"], tau=p["tau"])
    label = f"α={p['alpha']}, β={p['beta']}, τ={p['tau']}"
    plt.plot(z, g_vals, label=label)

plt.title("Softmax++ Non-linearity Function g(z)")
plt.xlabel("Logits z")
plt.ylabel("g(z)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
