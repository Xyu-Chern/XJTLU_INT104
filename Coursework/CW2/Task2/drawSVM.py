# Task 2 Step 1

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
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
X, y = preprocess_data(original_data, ['Index', 'Programme'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear SVM
linear_svm = SVC(kernel='linear', random_state=42)
cv_scores_linear = cross_val_score(linear_svm, X_train, y_train, cv=5)
print("Mean CV Accuracy for Linear SVM:", cv_scores_linear.mean())
linear_svm.fit(X_train, y_train)
y_pred_linear = linear_svm.predict(X_test)
accuracy_linear = accuracy_score(y_test, y_pred_linear)
print("Accuracy for Linear SVM: ", accuracy_linear)

# Non-linear SVM
nonlinear_svm = SVC(kernel='rbf', random_state=42)  # RBF kernel
cv_scores_nonlinear = cross_val_score(nonlinear_svm, X_train, y_train, cv=5)
print("Mean CV Accuracy for Non-linear SVM:", cv_scores_nonlinear.mean())
nonlinear_svm.fit(X_train, y_train)
y_pred_nonlinear = nonlinear_svm.predict(X_test)
accuracy_nonlinear = accuracy_score(y_test, y_pred_nonlinear)
print("Accuracy for Non-linear SVM: ", accuracy_nonlinear)

# Plotting confusion matrices
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
conf_matrix_linear = confusion_matrix(y_test, y_pred_linear)
sns.heatmap(conf_matrix_linear, annot=True, fmt='d', cmap='Blues', xticklabels=linear_svm.classes_, yticklabels=linear_svm.classes_)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix for Linear SVM')

plt.subplot(1, 2, 2)
conf_matrix_nonlinear = confusion_matrix(y_test, y_pred_nonlinear)
sns.heatmap(conf_matrix_nonlinear, annot=True, fmt='d', cmap='Blues', xticklabels=nonlinear_svm.classes_, yticklabels=nonlinear_svm.classes_)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix for Non-linear SVM')

plt.tight_layout()
plt.show()



#   Feature  Average Mutual Information
# 1   Grade                    0.198751
# 2   Total                    0.187396
# 3     MCQ                    0.104358
# 5      Q2                    0.087815
# 7      Q4                    0.086718
# 8      Q5                    0.058307
# 6      Q3                    0.048980
# 4      Q1                    0.040192
# 0  Gender                    0.032749