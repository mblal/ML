import numpy as np


def cost_function(x, y, w, base):
    m = x.shape[0]

    squared_error = 0
    for i in range(m):
        squared_error += ((np.dot(w, x[i]) + base) - y[i]) ** 2
    return squared_error / (2 * m)