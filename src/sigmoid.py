import numpy as np




def sigmoid(__z):
    return  1 / (1 + np.exp(-__z))

def compute_cost(X, y, w, b, *argv):
    m,n = X.shape
    cost = 0
    for i in range(m):
        y_hat_i = sigmoid(np.dot(X[i], w) + b)
        y_hat_i = np.clip(y_hat_i, 1e-15, 1 - 1e-15)
        cost +=(-y[i] * np.log(y_hat_i) - (1 - y[i]) * np.log(1 - y_hat_i))
    return cost/m

#compute_cost & compute_cost_2 valent la même chose (same result)
def compute_cost_v2(X, y, w, b, *argv):
    m,n = X.shape
    cost = 0
    y_hat = sigmoid(np.dot(X, w) + b)
    for i in range(m):
        y_hat_i = np.clip(y_hat[i], 1e-15, 1 - 1e-15)
        cost +=(-y[i] * np.log(y_hat_i) - (1 - y[i]) * np.log(1 - y_hat_i))
    return cost/m