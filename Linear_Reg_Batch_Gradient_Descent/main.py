import matplotlib.pyplot as plt  # Correct import
import pandas as pd
import lin_reg as ln
import numpy as np

# Load data
df = pd.read_csv("data/LinData_3_2_Noise6.csv")
X = df["X"].values
y = df["Y"].values

# Add intercept term to X and reshape y
X = np.c_[np.ones((len(X), 1)), X]  # Add a column of ones for the intercept
y = y.reshape(-1, 1)  # Reshape y to (m, 1)

# Normalize and Scale the features
X[:, 1] = (X[:, 1] - np.mean(X[:, 1])) / np.std(X[:, 1])

# Perform linear regression
theta = ln.lin_reg(X, y, eta=0.1, n_iter=1000, m=len(X))
print("Theta:", theta)
# Predict y values using the learned theta
y_pred = theta[0] + theta[1] * X[:, 1]  # Use the second column of X (original feature)

# Plot the data and the regression line
plt.scatter(X[:, 1], y, color="blue", label="Data Points")  # Plot original data
plt.plot(X[:, 1], y_pred, color="red", label="Regression Line")  # Plot regression line
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Linear Regression")
plt.legend()
plt.show()