import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation

class Perceptron():
    def __init__(self, input_size, lr=1, epochs=300):
        self.W = np.zeros(input_size+1)
        self.epochs = epochs
        self.lr = lr
        self.W_ls = []
        self.RMSE_ls = []

    def act_fn(self, x):
        return 1 if x >= 0 else 0

    def predict(self, x):
        net = self.W.T.dot(x)
        act_fn = self.act_fn(net)
        return act_fn

    def lrate(self, it):
        return self.lr*(1 - (it/self.epochs))

    def rmsd(self, err):
        return np.sqrt(np.sum(np.square(err))/err.shape[0])

    def fit(self, X, d):
        for it in range(self.epochs):
            lrate = self.lrate(it)
            err = np.array([])
            for i in range(d.shape[0]):
                x = np.insert(X[i], 0, -1)
                y = self.predict(x)
                e = d[i] - y
                self.W = self.W + lrate * e * x
                err = np.append(err, e)
            self.RMSE_ls.append(self.rmsd(err))
            print(F'{self.rmsd(err):.2f}')
            self.W_ls.append(self.W)
        self.W_ls = np.array(self.W_ls)
    
    def evaluate(self, X, d):
            err = np.array([])
            RMSE = np.array([])
            for i in range(d.shape[0]):
                x = np.insert(X[i], 0, -1)
                y = self.predict(x)
                e = d[i] - y
                err = np.append(err, e)
            RMSE = np.append(RMSE, self.rmsd(err))
            return np.mean(RMSE)

if __name__ == '__main__':
    ### Training set
    train_set = pd.DataFrame(np.random.rand(1000, 2)*100, columns=['x', 'y'])
    train_set['color'] = 1
    train_set.loc[train_set.y -1.8*train_set.x > -35, 'color'] = 0
    X = train_set[['x','y']].to_numpy()
    d = train_set.color.to_numpy()
    
    ### Test set
    test_set = pd.DataFrame(np.random.rand(100, 2)*100, columns=['x', 'y'])
    test_set['color'] = 1
    test_set.loc[test_set.y -1.8*test_set.x > -35, 'color'] = 0
    X_test = test_set[['x','y']].to_numpy()
    d_test = test_set.color.to_numpy()
    
    ### NeuralNet
    perceptron = Perceptron(input_size=2)
    perceptron.fit(X, d)

    #print(F'Training set global error: {perceptron.evaluate(X, d):.2f}')
    #print(F'Test set global error: {perceptron.evaluate(X_test, d_test):.2f}')

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    # plot training set
    train_plt = ax.scatter(train_set.x, train_set.y, c=train_set.color)
    
    # plot final line
    #x1 = np.linspace(0, 100, 3)
    #x2 = -(perceptron.W[1]/perceptron.W[2])*x1 + perceptron.W[0]/perceptron.W[2]
    #line_plt, = ax.plot(x1, x2, linewidth=2.0, c='r')
    
    ###############################################################
    ########################## ANIMATION ##########################
    ###############################################################

    line_plt, = ax.plot([], [], linewidth=2.0, c='r')
   
    # initialization function: plot the background of each frame
    def init():
        line_plt.set_data([], [])
        return line_plt,

    # animation function.  This is called sequentially
    def animate(i):
        x = np.linspace(0, 100, 3)
        w = perceptron.W_ls
        y = -(w[i][1]/w[i][2])*x + w[i][0]/w[i][2]
        line_plt.set_data(x, y)
        return line_plt,

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=perceptron.W_ls.shape[0], interval=1, blit=True, repeat=False)

    ###############################################################
    ####################### END ANIMATION #########################
    ###############################################################    

    plt.tick_params(axis='x', labelsize=15)
    plt.tick_params(axis='y', labelsize=15)
    plt.xlim(0,100)
    plt.ylim(0,100)
    plt.show()
