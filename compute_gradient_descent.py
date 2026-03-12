import numpy as np
from gradient_descent import gradient_descent
from cost_function import cost_function
def compute_gradient(x_train, y_train, iterations):
    w = np.zeros(x_train.shape[1])
    b = 0
    alpha = 0.03
    m = x_train.shape[0]
    n = x_train.shape[1]
    i = 0
    history = []

    while i < iterations:
        w, b = gradient_descent(x_train, y_train, alpha, w, b)
        i = i + 1
        history.append(cost_function(x_train, y_train, w, b))

    return w, b,history