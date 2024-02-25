import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import randint


Crime = pd.read_csv("communities.data", header=None, sep=",")
# We find ViolentCrimesPerPop in column 127
print(Crime)
dependent = Crime.loc[:, [127]]
# Convert dataframe to matrix (ndarray)
dependent = dependent.to_numpy()
# All the independent variables are in columns 17,26,27,31,32,37,76,90,95
independentVars = Crime.loc[:, [17, 26, 27, 31, 32, 37, 76, 90, 95]]
# Convert dataframe to matrix (ndarray)
independentVars = independentVars.to_numpy()

independentVars = np.concatenate([np.ones((len(independentVars),1)), independentVars], axis= 1)
theta_init = np.zeros((independentVars.shape[1],1))
learning_rate = 0.01
num_iteration = 10000

def cost_function (X,y,theta):
    m= len(y)
    y_hat = X.dot(theta)
    J = (1/m)*np.sum(np.square(y_hat - y))

    return J

def gradient(X,y,theta):
    m= len(y)
    y_hat = X.dot(theta)
    grad = (1/m)*X.T.dot(y_hat - y)

    return grad

def Stochastic_Gradient_Descent(X,y,theta_init, learning_rate, num_iteration):
    m = len(y)

    theta = theta_init.copy()
    J_history = np.zeros(num_iteration)

    for i in range(num_iteration):
        rand_index = np.random.randint(0,m)

        x_i = np.array([X[rand_index,:]]).reshape(1, -1)
        y_i = np.array([y[rand_index,:]]).reshape(1, -1)

        grad = gradient(x_i,y_i, theta)

        theta = theta - learning_rate* grad

        J_history [i] = cost_function(X, y, theta)
    return theta, J_history

theta, J_history = Stochastic_Gradient_Descent(independentVars, dependent, theta_init, learning_rate, num_iteration)


plt.plot(J_history)
plt.xlabel('No. of iterations')
plt.ylabel('Cost')
plt.show()

print(">>> Estimated thetas")
print(theta)
print(">>> Minimum cost value")
print(J_history[num_iteration-1])




