import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import data_vis

def generate_linear_data(n, m=2, b=5, noise_level=1.0):
    """
    Generates linearly separable data with added Gaussian noise.

    param n: Number of data points
    param m: Slope of the line
    param b: Intercept of the line
    param noise_level: Standard deviation of Gaussian noise
    return: Tuple (X, y) of generated data
    """
    X = np.linspace(0, 10, n)  # Generate n evenly spaced points
    noise = np.random.normal(0, noise_level, n)  # Gaussian noise
    y = m * X + b + noise  # Apply linear function with noise
    return X, y


def save_data(X, y, filename="linear_data.csv"):
    """
    Saves the dataset to a CSV file.

    param X: Input features
    param y: Target values
    param filename: Name of the CSV file
    """
    df = pd.DataFrame({"X": X, "Y": y})
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")

"""
Moderate slope, low noise (clean data)

m = 2, b = 5, noise_level = 1
Steep slope, moderate noise (clear trend but some spread)

m = 5, b = 1, noise_level = 3
Shallow slope, high noise (harder to fit)

m = 0.5, b = 10, noise_level = 7
Negative slope, medium noise (downward trend)

m = -3, b = 15, noise_level = 4
Flat line with high noise (almost random-looking data)

m = 0, b = 8, noise_level = 10
Extreme slope with low noise (clear pattern, strong change)

m = 10, b = -5, noise_level = 2
"""

# Define different parameter sets
parameter_sets = [
    {"m": 2, "b": 5, "noise_level": 1, "filename": "data_m2_b5_n1.csv"},
    {"m": 5, "b": 1, "noise_level": 3, "filename": "data_m5_b1_n3.csv"},
    {"m": 0.5, "b": 10, "noise_level": 7, "filename": "data_m0.5_b10_n7.csv"},
    {"m": -3, "b": 15, "noise_level": 4, "filename": "data_m-3_b15_n4.csv"},
    {"m": 0, "b": 8, "noise_level": 10, "filename": "data_m0_b8_n10.csv"},
    {"m": 10, "b": -5, "noise_level": 2, "filename": "data_m10_b-5_n2.csv"},
]

# Generate, save, and plot each dataset
n = 100
for params in parameter_sets:
    X, y = generate_linear_data(n, params["m"], params["b"], params["noise_level"])
    save_data(X, y, params["filename"])
    data_vis.plot_data(X, y, title=f"m={params['m']}, b={params['b']}, noise={params['noise_level']}")