# Task 2 Step 2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

original_data = pd.read_excel("Coursework/CW_Data.xlsx")
X = original_data.drop(['Index', 'Programme'], axis="columns")
y = original_data['Programme']

mean = X.mean(axis=0)
std = X.std(axis=0)
X_normalized = (X - mean) / std

tsne = TSNE(n_components=2, random_state=60, perplexity=100, learning_rate=50, n_iter=1000, early_exaggeration=70)
X_scale = tsne.fit_transform(X_normalized)

def model_train(X, y, test_size, seed, kernel_type='rbf'):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    model = SVC(kernel=kernel_type, random_state=seed)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return model, accuracy

def find_best_model(X, y, size, kernel_type='rbf'):
    best_model = None
    best_accuracy = 0
    for i in range(10):
        model, accuracy = model_train(X, y, size, i, kernel_type)
        if accuracy > best_accuracy:
            best_model, best_accuracy = model, accuracy
    return best_model, best_accuracy

def plot_decision_boundary(clf, X, y, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, num=400), np.linspace(y_min, y_max, num=400))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.4)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', s=50)
    plt.title(f'{title} - Accuracy in all Dataset: {round(accuracy_score(y, clf.predict(X)) * 100, 2)}%')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.show()

size = 0.3
best_linear_svm_model, best_linear_svm_accuracy = find_best_model(X_scale, y, size, 'linear')
best_nonlinear_svm_model, best_nonlinear_svm_accuracy = find_best_model(X_scale, y, size, 'rbf')

plot_decision_boundary(best_linear_svm_model, X_scale, y, "Best Linear SVM")
plot_decision_boundary(best_nonlinear_svm_model, X_scale, y, "Best Nonlinear SVM")
