import numpy as np
import matplotlib.pyplot as plt

# Sigmoid 함수
def sigmoid(x, a=1, b=0):
    return 1 / (1 + np.exp(-a * (x - b)))

# 데이터셋
decision_boundaries = [3, 1, 5, -2, -1, 7]
directions = ['pos', 'neg', 'pos', 'neg', 'pos', 'neg']
Y = [1 if direction == 'pos' else 0 for direction in directions]

# 파라미터 설정
a = 1
b = np.mean(decision_boundaries)

# 각 decision_boundary에 대해 데이터셋을 생성하고 sigmoid 함수를 플롯
for decision_boundary, y in zip(decision_boundaries, Y):
    plt.scatter(decision_boundary, y, color='b')

x_values = np.linspace(-10, 10, 400)
y_values = sigmoid(x_values, a, b)
plt.plot(x_values, y_values, color='r')
plt.show()
