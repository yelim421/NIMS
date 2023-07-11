import numpy as np

class Sigmoid:
    def __call__(self, z):
        self.a = 1 / (1 + np.exp(-z))
        return self.a

    def derivative(self):
        return self.a * (1 - self.a)