import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, make_scorer
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import train_test_split, cross_val_score
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
scorer = make_scorer(custom_accuracy, greater_is_better=True)

# indexList = ['Index', 'Programme','Q4' ,'Q5' ,'Q3' ,'Q1', 'Gender']
indexList = ['Index', 'Programme' ,'Q5' ,'Q3' ,'Q1', 'Gender']
# indexList = ['Index', 'Programme','Q3' ,'Q1', 'Gender']
# indexList = ['Index', 'Programme','Q1', 'Gender']
# indexList = ['Index', 'Programme','Gender']
# indexList = ['Index', 'Programme']

X, y = preprocess_data(original_data, indexList)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

kmeans = KMeans(n_clusters=4, random_state=0)
cv_scores = cross_val_score(kmeans, X_train, y_train, cv=5, scoring=scorer)
print(f"Average Cross-Validation: {cv_scores.mean()}")

kmeans.fit(X_train)
test_labels = kmeans.predict(X_test)
mapped_test_labels = match_labels(y_test, test_labels)
correct_predictions = np.sum(mapped_test_labels == y_test)
test_accuracy = correct_predictions / len(y_test)
print(f"Test Accuracy: {test_accuracy}")

kmeans.fit(X_train)
all_labels = kmeans.predict(X)
mapped_all_labels = match_labels(y, all_labels)

cm = confusion_matrix(y, mapped_all_labels, labels=np.unique(y))
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
