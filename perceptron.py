import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Perceptron():
    def __init__(self, input_size, lr=1, epochs=100):
        self.W = np.zeros(input_size+1)
        self.epochs = epochs
        self.lr = lr
        self.histW = []

    def activation_fn(self, x):
        return 1 if x >= 0 else 0

    def predict(self, x):
        z = self.W.T.dot(x)
        a = self.activation_fn(z)
        return a

    def fit(self, X, d):
        for _ in range(self.epochs):
            for i in range(d.shape[0]):
                x = np.insert(X[i], 0, -1)
                y = self.predict(x)
                e = d[i] - y
                self.W = self.W + self.lr * e * x
                self.histW.append(self.W)
        self.histW = np.array(self.histW)

if __name__ == '__main__':
    train_set = pd.DataFrame(np.random.rand(100, 2)*100, columns=['x', 'y'])
    train_set['color'] = 1
    train_set.loc[train_set.y - 1.8*train_set.x > -6, 'color'] = 0
    X = train_set[['x','y']].to_numpy()
    d = train_set.color.to_numpy()

    perceptron = Perceptron(input_size=2)
    perceptron.fit(X, d)
    print(perceptron.histW)
    print(perceptron.W)

    x1 = np.linspace(0, 100, 3)
    x2 = -(perceptron.W[1]/perceptron.W[2])*x1 + perceptron.W[0]/perceptron.W[2]

    # Plotting
    train_plt = plt.scatter(train_set.x, train_set.y, c=train_set.color)
    line_plt = plt.plot(x1, x2, linewidth=2.0, c='r')
    plt.tick_params(axis='x', labelsize=15)
    plt.tick_params(axis='y', labelsize=15)

    plt.xlim(0,100)
    plt.ylim(0,100)

    plt.show()