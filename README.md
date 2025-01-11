# Exploration of Machine Learning

This repository is dedicated to exploring machine learning techniques with a focus on practical implementation. As part of a university project, we worked on solving the **Fashion MNIST classification problem** using Support Vector Machines (SVMs) from multiple algorithms at our disposal. This repository contains two key implementations of SVM: one leveraging the `scikit-learn` library and another built from scratch for deeper understanding and customization.

## Repository Structure

```
Exploration-of-Machine-Learning/
│
├── SVM/
│   ├── SVM-with-sklearn/
│   │   └── main.py
│   ├── SVM-user-defined/
│   │   └── main_02.py
│
└── README.md
```

- **`SVM-with-sklearn/`**: Contains an implementation of SVM using `scikit-learn`. This is a straightforward and robust approach to achieve high accuracy.
- **`SVM-user-defined/`**: Implements SVM from scratch, showcasing techniques such as kernel computation, optimization, and memory-efficient batching.

**Note**: The dataset files (`train.npz` and `test.npz`) are not included in this repository due to size constraints. Please download them separately or generate equivalent datasets using Fashion MNIST preprocessing scripts.

---

## Implementations

### SVM with `scikit-learn` (`main.py`)
This implementation focuses on ease of use and flexibility by leveraging the `scikit-learn` library. Key features include:
- Preprocessing with `StandardScaler` to normalize data.
- Training with various kernel options (`RBF`, `POLY`, and `LINEAR`) to explore performance trade-offs.
- Achieved a maximum accuracy of **90.4%** with the RBF kernel (`C=10.0, gamma=scale`), aligning closely with benchmarks for SVM on Fashion MNIST.

**Highlights**:
- Experimented with different hyperparameters (`C`, `gamma`) to optimize accuracy.
- Explored performance trade-offs between kernel types.

### SVM from Scratch (`main_02.py`)
This custom implementation of SVM demonstrates the fundamental workings of the algorithm. It was designed with optimizations to ensure it runs efficiently on local machines. Key features include:
- **Custom RBF Kernel**: Batches computation to conserve memory and uses an optimized formula for squared distances.
- **Gradient-based Optimization**: Implements iterative updates to Lagrange multipliers (`alpha`) for margin maximization.
- **Early Stopping**: Convergence is determined based on the norm difference of `alpha` across iterations.

**Highlights**:
- Achieved convergence in fewer iterations using efficient batching and learning rate adjustments.
- Memory-efficient kernel computation for large datasets.

## Future Plans
- Extend the repository with additional algorithms like Random Forest, Logistic Regression, and Deep Learning models.
- Explore further optimizations and scalability for the custom SVM implementation.
- Add more subfolders categorizing implementations by algorithm types.

## How to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Exploration-of-Machine-Learning.git
   ```
2. Place the Fashion MNIST dataset files (`train.npz` and `test.npz`) in the repository root.
3. Run the desired SVM implementation:
   ```bash
   python SVM/SVM-with-sklearn/main.py
   ```
   or
   ```bash
   python SVM/SVM-user-defined/main_02.py
   ```

---

Feel free to suggest improvements or contribute to this repository!
