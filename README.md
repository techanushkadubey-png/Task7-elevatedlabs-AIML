# KNN Classification – Iris Dataset
# Task 6 – AI & ML Internship

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from matplotlib.colors import ListedColormap

# -----------------------------
# 1. Load Dataset
# -----------------------------
iris = load_iris()
X = iris.data
y = iris.target

# -----------------------------
# 2. Normalize Features
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# 3. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -----------------------------
# 4. Train KNN Model
# -----------------------------
k = 5
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# -----------------------------
# 5. Evaluation
# -----------------------------
y_pred = knn.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -----------------------------
# 6. Try different values of K
# -----------------------------
k_values = range(1, 21)
accuracies = []

for K in k_values:
    model = KNeighborsClassifier(n_neighbors=K)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    accuracies.append(acc)

plt.plot(k_values, accuracies)
plt.xlabel('K value')
plt.ylabel('Accuracy')
plt.title('Accuracy vs K')
plt.grid(True)
plt.show()

# -----------------------------
# 7. Decision Boundary (2 Features Only)
# -----------------------------
X_vis = X_scaled[:, :2]  # First two features for visualization

knn_vis = KNeighborsClassifier(n_neighbors=5)
knn_vis.fit(X_vis, y)

x_min, x_max = X_vis[:, 0].min() - 1, X_vis[:, 0].max() + 1
y_min, y_max = X_vis[:, 1].min() - 1, X_vis[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

Z = knn_vis.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")

plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y, edgecolor='k')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Decision Boundary (KNN)")
plt.show()
