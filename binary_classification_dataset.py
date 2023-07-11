import numpy as np
import matplotlib.pyplot as plt

def make_binary_classification_dataset(n_samples, decision_point, direction):
    X = np.random.uniform(-10, 10, n_samples)
    if direction == 'pos':
        Y = (X > decision_point).astype(int)
    elif direction == 'neg':
        Y = (X < decision_point).astype(int)
    return X, Y

def sigmoid(x, a, b):
    return 1 / (1 + np.exp(-a*x-b))

n_samples = 100
decision_point = 3
direction = 'pos'

a, b = 10, -30

X, Y = make_binary_classification_dataset(n_samples, decision_point, direction)

x = np.linspace(-decision_point -5, decision_point +5, 100)
y = sigmoid(x, a=a, b=b)

plt.figure(figsize=(10, 6))
plt.scatter(X, Y, c=Y, cmap='bwr', edgecolor='k', alpha=0.6)
plt.axvline(x=decision_point, color='k', linestyle='--')
plt.show()
