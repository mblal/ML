import numpy as np
def gradient_descent(x_train, y_train, alpha = 0.01, w=None, b=None):

    m = x_train.shape[0]
    n = x_train.shape[1]

    if w is None:
        w = np.zeros(n)

    dj_w = np.zeros(n)
    dj_b = 0
    for i in range(m):
        err = ((np.dot(w, x_train[i]) + b) - y_train[i])
        for j in range(n):
            dj_w[j] +=  err * x_train[i][j]
        dj_b += err

    tmp_w = w - alpha * dj_w / m
    tmp_b = b - alpha * dj_b / m


    w = tmp_w
    b = tmp_b

    return w, b