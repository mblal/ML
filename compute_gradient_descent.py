import numpy as np
from gradient_descent import gradient_descent

def compute_gradient(x_train, y_train, iterations):
    w = np.zeros(x_train.shape[1])
    b = 0
    alpha = 0.01
    m = x_train.shape[0]
    n = x_train.shape[1]
    i = 0

    while i < iterations:
        w, b = gradient_descent(x_train, y_train, alpha, w, b)
        i = i + 1