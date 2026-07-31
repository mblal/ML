import numpy as np
from sigmoid import sigmoid
def compute_gradient(X, y, w, b, *agrv):
    m,n = X.shape
    dj_dw = np.zeros(w.shape)
    dj_db = 0.
    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i], w) + b)
        error_i = f_wb_i - y[i]
        for j in range(n):
            dj_dw[j] += error_i * X[i][j]
        dj_db += error_i

    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db

# V2 is the vectorized version of the function above "compute_gradient"
def compute_gradient_v2(X, y, w, b, *agrv):
    m,n = X.shape

    f_wb = sigmoid(np.dot(X, w) + b)
    error = f_wb - y
    dj_dw = np.dot(X.T, error)


    dj_dw = dj_dw / m
    dj_db = np.sum(error) / m

    return dj_dw, dj_db