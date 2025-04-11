"""
20 мая 2024
Линейный и квадратичный дискриминантный анализ с использованием Sklearn
www.geeksforgeeks.org/linear-and-quadratic-discriminant-analysis-using-sklearn/
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from pspat_tst_data import *

# Generate synthetic data

# X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0,
#                            n_clusters_per_class=1, n_classes=3, random_state=42)

n_samples = 100
K = 3
# cor=['blue','green']
# cor=colors
X, y = all_create_data(npoints=n_samples, view1=True)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.75, random_state=42)

# Применение линейного дискриминантного анализа (LDA)
# Initialize and train the LDA model
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
y_pred_lda = lda.predict(X_test)

print("LDA Accuracy:", accuracy_score(y_test, y_pred_lda))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lda))
print("Classification Report:\n", classification_report(y_test, y_pred_lda))


# Применение квадратичного дискриминантного анализа (QDA)
# Initialize and train the QDA model
qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, y_train)

# Make predictions
y_pred_qda = qda.predict(X_test)

# Evaluate the model
print("QDA Accuracy:", accuracy_score(y_test, y_pred_qda))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_qda))
print("Classification Report:\n", classification_report(y_test, y_pred_qda))

# Визуализация линейного и Квадратичного Дискриминантного анализа

def plot_decision_boundaries(X, y, model, title, subplot_index, X_train, y_train, X_all, y_all):
    plt.subplot(subplot_index)
    x_min, x_max = X_all[:, 0].min() - 1, X_all[:, 0].max() + 1
    y_min, y_max = X_all[:, 1].min() - 1, X_all[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.1)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    # plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors='k', marker='s')
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker='1')
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')


# plt.figure(figsize=(10, 4))
# # Plot decision boundaries for LDA
# plot_decision_boundaries(X_test, y_test, lda, "LDA Decision Boundary", 121, X_train, y_train, X, y)
# # Plot decision boundaries for QDA
# plot_decision_boundaries(X_test, y_test, qda, "QDA Decision Boundary", 122, X_train, y_train, X, y)

plot_decision_boundaries(X_test, y_test, lda, "Линейный дискриминантный анализ", 111, X_train, y_train, X, y)
plt.grid()

plt.tight_layout()
plt.show()