import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pandas as pd


class SVM:
    def __init__(self, C=1.0, gamma=0.1, learning_rate=0.01, max_iter=1000, batch_size=200):
        self.C = C
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size

    def rbf_kernel_batch(self, X1, X2):
        """Calculează kernel-ul RBF între X1 și X2 în blocuri pentru a economisi memorie."""
        # Optimization for Kernel:
        # Batching
        # Optimized formula for squared distances
        n1, n2 = X1.shape[0], X2.shape[0]
        K = np.zeros((n1, n2))
        for i in range(0, n1, self.batch_size):
            X1_batch = X1[i:i + self.batch_size]
            for j in range(0, n2, self.batch_size):
                X2_batch = X2[j:j + self.batch_size]
                pairwise_sq_dists = (
                    np.sum(X1_batch**2, axis=1).reshape(-1, 1) +
                    np.sum(X2_batch**2, axis=1) -
                    2 * np.dot(X1_batch, X2_batch.T)
                )
                K[i:i + self.batch_size, j:j + self.batch_size] = np.exp(-self.gamma * pairwise_sq_dists)
        return K

    def fit(self, X, y):
        """Antrenează SVM-ul folosind kernel-ul precalculat."""
        n_samples = X.shape[0]
        self.alpha = np.zeros(n_samples)
        self.b = 0
        prev_alpha = np.zeros_like(self.alpha)

        # Precalculation of kernel
        K = self.rbf_kernel_batch(X, X)

        for iteration in range(self.max_iter):
            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                X_batch, y_batch = X[start:end], y[start:end]
                K_batch = K[start:end, :]  # Kernel Batching

                for i in range(X_batch.shape[0]):
                    margin = y_batch[i] * (np.sum(self.alpha * y * K_batch[i, :]) + self.b)
                    if margin < 1:
                        self.alpha[start + i] += self.learning_rate * (1 - margin)

            # Convergence for time peroformance
            if np.linalg.norm(self.alpha - prev_alpha) < 1e-5:
                print(f"Convergență atinsă la iterația {iteration + 1}")
                break
            prev_alpha = self.alpha.copy()

        self.b -= self.learning_rate * np.sum(self.alpha * y)
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """Realizează predicții pe baza kernel-ului calculat între punctele de testare și cele de antrenament."""
        K = self.rbf_kernel_batch(X, self.X_train)
        return np.sign(np.dot(K, self.alpha * self.y_train) + self.b)


# Încărcarea datelor
train_data = np.load('train.npz')
test_data = np.load('test.npz')

x_train = train_data['x_train']
y_train = train_data['y_train']
x_test = test_data['x_test']

# Preprocesare: standardizare
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Împărțire în antrenare și validare
x_train_split, x_val_split, y_train_split, y_val_split = train_test_split(
    x_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

# Antrenarea modelului SVM
svm = SVM(C=10.0, gamma=1, learning_rate=0.01, max_iter=500, batch_size=500)
svm.fit(x_train_split, y_train_split)

# Evaluarea pe setul de validare
y_val_pred = svm.predict(x_val_split)
accuracy = accuracy_score(y_val_split, y_val_pred)
print(f"Acuratețea pe setul de validare: {accuracy * 100:.2f}%")

# Predicții pe setul de testare
y_test_pred = svm.predict(x_test)

# Crearea fișierului de trimitere
submission = pd.DataFrame({
    'Id': np.arange(len(x_test)),
    'Label': y_test_pred
})
submission.to_csv('submission.csv', index=False)
print("Fișierul submission.csv a fost creat cu succes.")
