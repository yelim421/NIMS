import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def moved_sigmoid(x, a, b):
    return 1 / (1 + np.exp(-a*x-b))

def get_decision_point(a,b):
    x = -b/a
    return x, 0.5

a = 5
b = 10
x = np.linspace(-10, 10, 1000)
y = moved_sigmoid(x, a, b)
decision_point = get_decision_point(a,b)

plt.figure(figsize=(8, 6))
plt.plot(x, y, label="moved_sigmoid")
plt.plot(decision_point[0], decision_point[1], 'ro')

plt.show()
