import numpy as np
import matplotlib.pyplot as plt

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
decision_boundary = 7
direction = 'pos'

get_dataset = GetDataset(n_samples)
X, Y = get_dataset(n_samples, decision_boundary, direction)


fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(X[Y == 0], Y[Y == 0], color='red')
ax.scatter(X[Y == 1], Y[Y == 1], color='blue')

linear = LinearFunction(w=0.0, b=0.0)
sigmoid = Sigmoid()
loss_fn = BCELoss()

xlim = ax.get_xlim()
x_model = np.linspace(xlim[0], xlim[1], 100)
y_model = sigmoid(linear(x_model))
ax.plot(x_model, y_model)

for x, y in zip(X,Y):
    z = linear(x)
    y_pred = sigmoid(x)
    loss = loss_fn(y_pred, y)
    print(loss)