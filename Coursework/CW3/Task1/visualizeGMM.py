
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from itertools import permutations
from sklearn.metrics import make_scorer

def preprocess_data(data, listIndex):
    X = data.drop(listIndex, axis="columns")
    y = data['Programme']
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    X_normalized = (X - mean) / std
    return X_normalized, y

def match_labels(true_labels, pred_labels):
    unique_true_labels = np.unique(true_labels)
    unique_pred_labels = np.unique(pred_labels)
    
    if len(unique_true_labels) != len(unique_pred_labels):
        raise ValueError("Number of unique true labels must be equal to number of unique predicted labels.")
    
    best_mapping = None
    best_accuracy = 0
    best_mapped_labels = None

    for perm in permutations(unique_true_labels):
        perm_mapping = {unique_pred_labels[i]: perm[i] for i in range(len(unique_pred_labels))}
        mapped_labels = np.array([perm_mapping[label] for label in pred_labels])
        accuracy = np.sum(mapped_labels == true_labels) / len(true_labels)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_mapping = perm_mapping
            best_mapped_labels = mapped_labels

    return best_mapped_labels

def custom_accuracy(true_labels, pred_labels):
    mapped_labels = match_labels(true_labels, pred_labels)
    correct_predictions = np.sum(mapped_labels == true_labels)
    accuracy = correct_predictions / len(true_labels)
    return accuracy

np.random.seed(0)
original_data = pd.read_excel("Coursework/CW_Data.xlsx")
scorer = make_scorer(custom_accuracy, greater_is_better=True)
indexList1 = ['Index', 'Programme']
indexList2 = ['Index', 'Programme','Q4' ,'Q5' ,'Q3' ,'Q1', 'Gender']

X1, y = preprocess_data(original_data, indexList1)
X2, y = preprocess_data(original_data, indexList2)

tsne = TSNE(n_components=2, random_state=60, perplexity=100, learning_rate=50, n_iter=1000, early_exaggeration=70)
X_tsne = tsne.fit_transform(X1)

X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size=0.3, random_state=0)
gmm = GaussianMixture(n_components=4, random_state=0)
gmm.fit(X_train)
test_labels = gmm.predict(X2)
mapped_labels = match_labels(y, test_labels)

def color_label(y):
    unique_labels = np.unique(y)
    label_to_color = {label: idx for idx, label in enumerate(unique_labels)}
    colors = ['r', 'g', 'b', 'y']
    y_colors = [colors[label_to_color[label]] for label in y]
    return y_colors

plt.figure(figsize=(10, 8))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=color_label(y), marker='o', edgecolor='k', s=30, alpha=0.6)
plt.xlabel('t-SNE component 1')
plt.ylabel('t-SNE component 2')
plt.title('t-SNE visualization of True Labels')
plt.show()

plt.figure(figsize=(10, 8))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=color_label(mapped_labels), marker='o', edgecolor='k', s=30, alpha=0.6)
plt.xlabel('t-SNE component 1')
plt.ylabel('t-SNE component 2')
plt.title('t-SNE visualization of True Labels')
plt.show()

plt.figure(figsize=(10, 8))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=color_label(y), marker='o', edgecolor='k', s=30, alpha=0.6)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_tsne, mapped_labels)
h = .02 
x_min, x_max = X_tsne[:, 0].min() - 1, X_tsne[:, 0].max() + 1
y_min, y_max = X_tsne[:, 1].min() - 1, X_tsne[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.2)
plt.xlabel('t-SNE component 1')
plt.ylabel('t-SNE component 2')
plt.title('t-SNE visualization of GMM True Labels with Decision Boundaries')
plt.show()

