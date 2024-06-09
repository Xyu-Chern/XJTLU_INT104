import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import numpy as np

def preprocess_data(data, listIndex):
    X = data.drop(listIndex, axis="columns")
    y = data['Programme']
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    X_normalized = (X - mean) / std
    return X_normalized, y

np.random.seed(0)
original_data = pd.read_excel("Coursework/CW_Data.xlsx")

# Data for all classifiers
X, y = preprocess_data(original_data, ['Index', 'Programme', 'Q3', 'Q1', 'Gender'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Grid Search to find best ccp_alpha for Decision Tree
param_grid = {'ccp_alpha': np.linspace(0, 0.02, 100)}
dt_grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
dt_grid_search.fit(X_train, y_train)
best_ccp_alpha = dt_grid_search.best_params_['ccp_alpha']

# Decision Tree with pruning
dt_classifier = DecisionTreeClassifier(random_state=42, ccp_alpha=best_ccp_alpha)

# SVM and Naive Bayes Classifiers
svm_classifier = SVC(kernel='rbf', random_state=42, probability=True)
nb_classifier = GaussianNB()

# Voting Classifier with hard voting
voting_clf = VotingClassifier(
    estimators=[
        ('decision_tree', dt_classifier),
        ('svm', svm_classifier),
        ('naive_bayes', nb_classifier)
    ],
    voting='hard'
)

# Cross-validation for the voting classifier
cv_scores = cross_val_score(voting_clf, X_train, y_train, cv=5)
print("Mean CV accuracy:", np.mean(cv_scores))

# Train the ensemble classifier
voting_clf.fit(X_train, y_train)

# Predictions using hard voting
y_pred_voting = voting_clf.predict(X_test)
accuracy_voting = accuracy_score(y_test, y_pred_voting)
print("Accuracy for Voting Classifier: ", accuracy_voting)

# Plotting confusion matrix
conf_matrix_voting = confusion_matrix(y_test, y_pred_voting)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_voting, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix for Voting Classifier')
plt.show()
