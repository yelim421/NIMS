import numpy as np
import matplotlib.pyplot as plt

target_a = 2
target_b = 1
n_samples= 100

def make_dataset(n_samples, target_a, target_b):
    X = np.random.normal(loc=0, scale=1, size=(n_samples,))
    noise = 0.3*np.random.normal(loc=0, scale=1, size=(n_samples,))
    Y = target_a * X + target_b +noise
    return X, Y

x_i, y_i = make_dataset(n_samples, target_a, target_b)

class LinearFunction:
    def __init__(self, a, b):
        self.a, self.b = a, b

    def __call__(self, x):
        return self.a * x + self.b

    def update_params(self, dloss_da, dloss_db, lr):
        self.a = self.a - lr * dloss_da
        self.b = self.b - lr * dloss_db

    def get_params(self):
        return self.a, self.b


model = LinearFunction(a = -2, b = -5)

lr = 0.001
epochs = 1000

a_vals = []
b_vals = []

for _ in range(epochs):
    for x, y in zip(x_i, y_i):
        y_pred = model(x)

        loss = (y_pred-y)**2

        dloss_da = 2*x*(y_pred-y)
        dloss_db = 2*(y_pred-y)

        model.update_params(dloss_da, dloss_db, lr)
    a, b = model.get_params()
    a_vals.append(a)
    b_vals.append(b)

plt.figure(figsize=(12,6))
plt.plot(a_vals)
plt.plot(b_vals)
plt.show()

print(a, b)