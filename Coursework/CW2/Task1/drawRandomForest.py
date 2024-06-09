# Task 1 Step 3

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
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

rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
cv_scores = cross_val_score(rf, X_train, y_train, cv=5)
print("Mean CV Accuracy:", cv_scores.mean())

rf.fit(X_train, y_train)

conf_matrix = confusion_matrix(y_test, rf.predict(X_test))
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=rf.classes_, yticklabels=rf.classes_)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy : ", accuracy)


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