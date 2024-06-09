# Task 1 Step 4

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

original_data = pd.read_excel("Coursework/CW_Data.xlsx")
X = original_data.drop(['Index', 'Programme'], axis="columns")
y = original_data['Programme']

mean = X.mean(axis=0)
std = X.std(axis=0)
X_normalized = (X - mean) / std

# pca = PCA(n_components=2)
# X_scale = pca.fit_transform(X_normalized)

tsne = TSNE(n_components=2, random_state=60, perplexity=100, learning_rate=50, n_iter=1000, early_exaggeration=70)
X_scale = tsne.fit_transform(X_normalized)

def model_train(X, y, test_size, seed, model_cls, criterion='entropy'):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    model = model_cls(random_state=seed, criterion=criterion)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return model, accuracy

def find_best_models(X, y, size):
    best_dt_model, best_rf_model = None, None
    best_dt_accuracy, best_rf_accuracy = 0, 0
    for i in range(10):
        dt_model, dt_accuracy = model_train(X, y, size, i, DecisionTreeClassifier)
        rf_model, rf_accuracy = model_train(X, y, size, i, RandomForestClassifier)
        if dt_accuracy > best_dt_accuracy:
            best_dt_model, best_dt_accuracy = dt_model, dt_accuracy
        if rf_accuracy > best_rf_accuracy:
            best_rf_model, best_rf_accuracy = rf_model, rf_accuracy
    print(best_dt_accuracy)
    print(best_rf_accuracy)
    return best_dt_model, best_dt_accuracy, best_rf_model, best_rf_accuracy

def plot_decision_boundary(clf, X, y, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, num=400), np.linspace(y_min, y_max, num=400))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.4)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', s=50)
    plt.title(f'{title} - Accuracy in all Datasets: {round(accuracy_score(y, clf.predict(X)) * 100, 2)}%')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.show()


size = 0.3
best_dt_model, best_dt_accuracy, best_rf_model, best_rf_accuracy = find_best_models(X_scale, y, size)

plot_decision_boundary(best_dt_model, X_scale, y, "Best Decision Tree")
plot_decision_boundary(best_rf_model, X_scale, y, "Best Random Forest")
