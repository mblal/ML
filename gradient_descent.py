def gradient_descent(x_train, y):
    m = x_train.shape[0]
    w = 0
    b = 0
    alpha = 0.01
    learning_effort = 0
    while True:
        dj_w_part1 = 0
        dj_b_part1 = 0
        dj_w = 0
        dj_b = 0
        for i in range(m):
            dj_w_part1 += ((w * x_train[i] + b) - y[i]) * x_train[i]
            dj_b_part1 += ((w * x_train[i] + b) - y[i])
        dj_w = dj_w_part1 / m
        dj_b = dj_b_part1 / m

        tmp_w = w - alpha * dj_w
        b = b - alpha * dj_b
        learning_effort += 1
        # attention, the simple condition below (w == tmp_w) may never be verified on a high precision computer,
        #if w == tmp_w:
        if abs(w - tmp_w) < 1e-9:
            return w, b, learning_effort -1

        w = tmp_w