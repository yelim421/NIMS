import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib

class GetDataset:
    def __init__(self, n_samples):
        self.n_samples = n_samples

    def __call__(self,n_samples, decision_boundary, direction):
        X = np.random.normal(loc=decision_boundary, scale=2, size=(n_samples, ))
        if direction == 'pos':
            Y = (X > decision_boundary).astype(int)
        elif direction == 'neg':
            Y = (X < decision_boundary).astype(int)
        return X, Y

class LinearFunction:
    def __init__(self, w, b):
        self.w, self.b = w, b

    def __call__(self, x):
        self.x = x
        return self.w * x + self.b

    def cal_deri(self):
        dz_dw = self.x
        dz_db = 1
        return dz_dw, dz_db

    def backward(self, from_sigmoid):
        dz_dw, dz_db = self.cal_deri()

        self.dJ_dw = from_sigmoid * dz_dw
        self.dJ_db = from_sigmoid * dz_db

    def update_weights(self, learning_rate):
        self.w = self.w - learning_rate * np.mean(self.dJ_dw)
        self.b = self.b - learning_rate * np.mean(self.dJ_db)

class Sigmoid:
    def __call__(self, z):
        self.z = z
        self.y_pred = 1 / (1 + np.exp(-z))
        return self.y_pred

    def backward(self, from_BCELoss):
        return from_BCELoss * self.y_pred * (1 - self.y_pred)

class BCELoss:
    def __call__(self, y_hat, y):
        self.y_hat = y_hat
        self.y = y
        return -1*(y*np.log(y_pred) + (1-y)*np.log(1 - y_pred))

    def backward(self):
        return - (self.y / self.y_hat - (1 - self.y) / (1 - self.y_hat))




n_samples = 100
decision_boundary = 0.0
direction = 'pos'
learning_rate = 0.05
epochs = 100

get_dataset = GetDataset(n_samples)
x_data, y_data = get_dataset(n_samples, decision_boundary, direction)

fig, ax = plt.subplots(figsize=(13, 8))
ax.scatter(x_data, y_data)
ax.tick_params(labelsize=20)

xlim = ax.get_xlim()
x_model = np.linspace(xlim[0], xlim[1], 100)


linear = LinearFunction(w=0.0, b=0.0)
sigmoid = Sigmoid()
loss_fn = BCELoss()

y_model = sigmoid(linear(x_model))
ax.plot(x_model, y_model, color='black', linewidth=3)

n_iterations = n_samples
cmap = matplotlib.cm.rainbow

for iter in range(n_iterations):

    linear_output = linear(x_data)
    y_pred = sigmoid(linear_output)
    loss = loss_fn(y_pred, y_data)

    loss_grad = loss_fn.backward()
    sigmoid_grad = sigmoid.backward(loss_grad)
    linear.backward(sigmoid_grad)

    linear.update_weights(learning_rate)

    y_model = sigmoid(linear(x_model))
    ax.plot(x_model, y_model, color=cmap(iter/n_iterations), alpha=0.5)

plt.show()