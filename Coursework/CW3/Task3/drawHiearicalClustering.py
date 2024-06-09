import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import confusion_matrix, make_scorer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from itertools import permutations

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

# indexList = ['Index', 'Programme','Q4' ,'Q5' ,'Q3' ,'Q1', 'Gender']
# indexList = ['Index', 'Programme' ,'Q5' ,'Q3' ,'Q1', 'Gender']
indexList = ['Index', 'Programme','Q3' ,'Q1', 'Gender']
# indexList = ['Index', 'Programme','Q1', 'Gender']
# indexList = ['Index', 'Programme','Gender']
# indexList = ['Index', 'Programme']

X, y = preprocess_data(original_data, indexList)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

agg_clustering = AgglomerativeClustering(n_clusters=4)
agg_clustering.fit(X_train)
train_labels = agg_clustering.labels_
mapped_train_labels = match_labels(y_train, train_labels)

# Custom cross-validation for clustering
kf = KFold(n_splits=5, shuffle=True, random_state=0)
cv_scores = []

for train_index, val_index in kf.split(X_train):
    X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
    
    agg_clustering.fit(X_train_fold)
    val_labels = agg_clustering.fit_predict(X_val_fold)
    mapped_val_labels = match_labels(y_val_fold, val_labels)
    accuracy = custom_accuracy(y_val_fold, mapped_val_labels)
    cv_scores.append(accuracy)

print(f"Average Cross-Validation Accuracy: {np.mean(cv_scores)}")

agg_clustering.fit(X_train)
test_labels = agg_clustering.fit_predict(X_test)
mapped_test_labels = match_labels(y_test, test_labels)
correct_predictions = np.sum(mapped_test_labels == y_test)
test_accuracy = correct_predictions / len(y_test)
print(f"Test Accuracy: {test_accuracy}")

agg_clustering.fit(X)
all_labels = agg_clustering.labels_
mapped_all_labels = match_labels(y, all_labels)

cm = confusion_matrix(y, mapped_all_labels, labels=np.unique(y))
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
