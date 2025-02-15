import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_data(X, y,title = "Here is the plot"):
    """
    Plots the generated dataset.

    param X: Input features
    param y: Target values
    """
    plt.scatter(X, y, color='blue', label='Data points')
    plt.xlabel("X values")
    plt.ylabel("Y values")
    plt.title(title)
    plt.legend()
    plt.show()

# df = pd.read_csv("linear_data.csv")
# X_loaded = df["X"].values
# y_loaded = df["Y"].values

#plot_data(X_loaded, y_loaded)