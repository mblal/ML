def cost_function(x_train, y, w, b):
    m = x_train.shape[0]
    squared_error = 0
    for i in range(m):
        current_squared_error = ((w*x_train[i] + b) - y[i]) ** 2
        squared_error += current_squared_error
    return squared_error / (2 * m)
