from compute_gradient import *
from sigmoid import *
import math as math
def gradient_descent(X, y, init_w, init_b, alpha, num_iters):

    J_histoty = []
    w = init_w.copy()
    b = init_b
    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient_v2(X, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        if i <= num_iters:
            J_histoty.append(compute_cost(X, y, w, b))
        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_histoty[i]}    ")
    return w,b, J_histoty