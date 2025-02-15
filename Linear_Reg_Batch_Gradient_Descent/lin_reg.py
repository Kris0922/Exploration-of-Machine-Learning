import numpy as np


def lin_reg(X, y, eta=0.1, n_iter=1000, m=5):
    theta = np.random.randn(2, 1)  # Correct shape for 2 features (intercept + feature)

    for iteration in range(n_iter):
        gradients = 2/m * X.T.dot(X.dot(theta) - y)  # Now X is (m, 2), so this works
        theta = theta - eta * gradients
        if iteration % 100 == 0:  # Print every 100 iterations
            print(f"Iteration {iteration}: theta = {theta.flatten()}")

    return theta
