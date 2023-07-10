import numpy as np
import matplotlib.pyplot as plt

# target function의 파라미터
a_star = 2
b_star = 1
n_samples = 100

def make_dataset(n_samples, a_star, b_star):
    X = np.random.normal(loc=0, scale=1, size=(n_samples,))
    Y = a_star * X + b_star
    noise = 0.3*np.random.normal(loc=0, scale=1, size=(n_samples,))
    Y = Y+noise
    return X, Y

x, y_star = make_dataset(n_samples, a_star, b_star)

class LinearFunction:
    def __init__(self, a,b):
        self.a, self.b = a,b

    def __call__(self, x):
        return self.a * x + self.b

    def update_params(self, dloss_da, dloss_db, lr):
        self.a = self.a - lr * dloss_da
        self.b = self.b - lr * dloss_db

    def get_params(self):
        return self.a, self.b

# 학습률
lr = 0.001

# 반복 횟수
epochs = 1000

# 파라미터의 변화를 저장할 리스트
a_vals = []
b_vals = []

model = LinearFunction(a=-2,b=-1)

# 경사 하강법
for _ in range(epochs):
    for x_sample, y in zip(x, y_star):
        # 예측값
        y_pred = model(x_sample)
        loss = (y_pred-y)**2

        # 그래디언트
        dloss_da = 2*x_sample*(y_pred-y)
        dloss_db = 2*(y_pred-y)

        model.update_params(dloss_da, dloss_db, lr)

    a,b = model.get_params()

    # 파라미터 값 저장
    a_vals.append(a)
    b_vals.append(b)

# 파라미터의 변화를 시각화
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(a_vals)
plt.title("a values")
plt.subplot(1, 2, 2)
plt.plot(b_vals)
plt.title("b values")
plt.show()

print("Final values of a and b: ", a, b)
