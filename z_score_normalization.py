import numpy as np
def z_score_normalization(x_train):

    # Average per column, HEY average per line np.mean(x_train, axis=1)
    mu = np.mean(x_train, axis=0)

    # Find de standard deviation of each column/feature
    sigma = np.std(x_train, axis=0)

    # Element-wise, subtract mu for that column from each example, divide by std for that column

    x_norm = (x_train - mu) / sigma

    return x_norm, mu, sigma