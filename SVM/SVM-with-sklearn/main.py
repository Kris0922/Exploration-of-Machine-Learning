# This is the file that implements the SVM model
# I want to also mention that I chose to implement the SVM model first at random
# I tried different kernels and C values and gammas, see the comments for more details
# I also had the intention to implement a random forest and logistic regression model
# The thing that stopped me is this post:
# https://www.kaggle.com/code/anushkabhadra/fashion-mnist-svm-90-6-accuracy
# I saw that the best accuracy that can be achieved with SVM is 90.6% and with deep learning 95%
# As my submission was close to 90.4% I decided to stop here and not implement the other models

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Încărcă datele din fișierele .npz
train_data = np.load('train.npz')
test_data = np.load('test.npz')

x_train = train_data['x_train']
y_train = train_data['y_train']
x_test = test_data['x_test']

# 1. Preprocesarea datelor
# Standardizare pentru a aduce datele la aceeași scară
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Împrem datele de antrenament în antrenare/validare
x_train_split, x_val_split, y_train_split, y_val_split = train_test_split(x_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
##############################################################################################################
# 2. Construirea modelului SVM cu kernel RBF
# the best working (for now)[old]
# this is the second one that I tried with C = 10.0 and gamma = scale
# accuracy: 90.45%
svm_model = SVC(kernel='rbf', C=10.0, gamma='scale', decision_function_shape='ovr', random_state=42)

##############################################################################################################
#testing with auto gamma
#accuracy: = 90.45% - the same output as with scale
#svm_model = SVC(kernel='rbf', C=10.0, gamma='auto', decision_function_shape='ovr', random_state=42)
#gamma too big = overfitting, too small = underfitting
# gamma = 0.1, note C = 10.0 does not run in a reasonable time
# Aici am cautat si vazut ca 90% e maxim obtinubil cu SVM si la 95% se poate ajunge cu deep learning
#svm_model = SVC(kernel='rbf', C=1.0, gamma=0.1, decision_function_shape='ovr', random_state=42)
# gamma = 0.01
#svm_model = SVC(kernel='rbf', C=1.0, gamma=0.01, decision_function_shape='ovr', random_state=42)
##############################################################################################################
# 2. Construirea modelului SVM cu kernel POLY
# accuracy: 89.36%
#svm_model = SVC(kernel='poly', C=10.0, gamma='scale', degree=3, coef0=1.0, decision_function_shape='ovr', random_state=42)
##############################################################################################################
##### other models tested
##############################################################################################################
# accuracy: ~85.13%
#svm_model = SVC(kernel='rbf', C=0.1, gamma='scale', decision_function_shape='ovr', random_state=42)
# first step, testing with multimple C values for the RBF kernel
# this was the first ever tested model with C = 1.0
# accuracy: ~89%
# svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', decision_function_shape='ovr', random_state=42)
# accuracy: 89.87% # third one tested with C = 100.0
#svm_model = SVC(kernel='rbf', C=100.0, gamma='scale', decision_function_shape='ovr', random_state=42)

# 2. Construirea modelului SVM cu kernel linear
# the 4th test was with a linear kernel
# observations: the wrost performace is with the linear kernel,
# the data does have a linear separability, but is not strong enough to produce a good enough result
# accuracy: ~84% # tested with a linear kernel did not work well enough, will do more testing with different C's
#svm_model = SVC(kernel='linear', C=1.0, decision_function_shape='ovr', random_state=42)
# before test: also testing linear with C = 10.0 because it worked well with the RBF kernel
# takes too long to run, test later
#svm_model = SVC(kernel='linear', C=10.0, decision_function_shape='ovr', random_state=42)
##############################################################################################################


# Antrenare

svm_model.fit(x_train_split, y_train_split)

# 3. Evaluarea performanței pe setul de validare
y_val_pred = svm_model.predict(x_val_split)
accuracy = accuracy_score(y_val_split, y_val_pred)
print(f"Acurateța pe setul de validare: {accuracy * 100:.2f}%")

# 4. Predicții pe setul de testare
y_test_pred = svm_model.predict(x_test)

# 5. Crearea fișierului de trimitere
submission = pd.DataFrame({
    'Id': np.arange(len(x_test)),
    'Label': y_test_pred
})

submission.to_csv('submission.csv', index=False)
print("Fișierul submission.csv a fost creat cu succes.")
