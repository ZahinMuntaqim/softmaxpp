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
    # {"alpha": 0.0, "beta": 1.0, "tau": 1.0},   # Baseline (linear)
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
