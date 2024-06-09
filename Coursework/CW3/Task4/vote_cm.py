
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import confusion_matrix, adjusted_rand_score, make_scorer, accuracy_score
import seaborn as sns
import numpy as np
from itertools import permutations
from scipy.stats import mode

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

    for perm in permutations(range(len(unique_pred_labels))):
        perm_mapping = {unique_pred_labels[i]: unique_true_labels[perm[i]] for i in range(len(unique_pred_labels))}
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

# Data preprocessing
indexList= ['Index', 'Programme', 'Q4', 'Q5', 'Q3', 'Q1', 'Gender']
# indexList2 = ['Index', 'Programme']
# indexList = ['Index', 'Programme', 'Q4', 'Q5', 'Q3']
X, y = preprocess_data(original_data, indexList)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
n_clusters = 4

def voting_ensemble_predict(X, kmeans, gmm, hierarchical):
    y_pred_kmeans = kmeans.predict(X)
    y_pred_gmm = gmm.predict(X)
    y_pred_hierarchical = hierarchical.fit_predict(X)
    y_pred_ensemble = np.vstack((y_pred_kmeans, y_pred_gmm, y_pred_hierarchical)).T
    y_pred_voting, _ = mode(y_pred_ensemble, axis=1)
    return y_pred_voting.flatten()


kf = KFold(n_splits=5, shuffle=True, random_state=0)
cv_accuracies = []

for train_index, val_index in kf.split(X_train):
    X_fold_train, X_fold_val = X_train.iloc[train_index], X_train.iloc[val_index]
    y_fold_train, y_fold_val = y_train.iloc[train_index], y_train.iloc[val_index]
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    gmm = GaussianMixture(n_components=n_clusters, random_state=0)
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
    
    kmeans.fit(X_fold_train)
    gmm.fit(X_fold_train)
    hierarchical.fit(X_fold_train)
    
    y_pred_voting = voting_ensemble_predict(X_fold_val, kmeans, gmm, hierarchical)
    mapped_y_pred_voting = match_labels(y_fold_val, y_pred_voting)
    
    accuracy = custom_accuracy(y_fold_val, mapped_y_pred_voting)
    cv_accuracies.append(accuracy)

print("5-Fold Cross-Validation Accuracy for Voting Ensemble: ", np.mean(cv_accuracies))

kmeans.fit(X_train)
gmm.fit(X_train)
hierarchical.fit(X_train)

y_pred_voting = voting_ensemble_predict(X_test, kmeans, gmm, hierarchical)
mapped_y_pred_voting = match_labels(y_test, y_pred_voting)
test_accuracy = custom_accuracy(y_test, mapped_y_pred_voting)

print("Test Accuracy for Voting Ensemble: ", test_accuracy)

y_pred_voting = voting_ensemble_predict(X, kmeans, gmm, hierarchical)
mapped_y_pred_voting = match_labels(y, y_pred_voting)

# Plotting confusion matrices
def plot_confusion_matrix(y_true, y_pred, title):
    conf_matrix = confusion_matrix(y_true, y_pred, labels=[1, 2, 3, 4])
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(1, 5), yticklabels=range(1, 5))
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(title)
    plt.show()

plot_confusion_matrix(y, mapped_y_pred_voting, 'Confusion Matrix for Voting Ensemble')
