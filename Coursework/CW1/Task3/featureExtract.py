# Task 3 Step 4

import pandas as pd
import numpy as np

# Load the data
original_data = pd.read_excel("Coursework/CW_Data.xlsx")
cleaned_data = original_data.drop(['Index', 'Programme'], axis=1)
mean = cleaned_data.mean()
std = cleaned_data.std()
standardized_data = (cleaned_data - mean) / std

X = standardized_data
y = original_data['Programme'] 

def calculate_entropy(y):
    total_samples = len(y)
    _, class_counts = np.unique(y, return_counts=True)
    entropy = 0
    for count in class_counts:
        p = count / total_samples
        if p > 0:
            entropy -= p * np.log2(p)
    return entropy

def calculate_information_gain_continuous(X, y, feature):
    entropy_parent = calculate_entropy(y)
    
    sorted_indices = np.argsort(X[feature])
    sorted_y = y.iloc[sorted_indices]
    sorted_x = X[feature].iloc[sorted_indices]
    best_gain = 0

    total_samples = len(y)
    for i in range(1, total_samples):
        if sorted_x.iloc[i] == sorted_x.iloc[i - 1]:
            continue

        left_y = sorted_y.iloc[:i]
        right_y = sorted_y.iloc[i:]

        entropy_left = calculate_entropy(left_y)
        entropy_right = calculate_entropy(right_y)

        entropy_children = (len(left_y) / total_samples) * entropy_left + (len(right_y) / total_samples) * entropy_right

        gain = entropy_parent - entropy_children
        if gain > best_gain:
            best_gain = gain

    return best_gain

selected_features = []
remaining_features = list(X.columns)


while remaining_features:
    gains_with_new_feature = []
    for feature in remaining_features:
        gain_with_feature = calculate_information_gain_continuous(X, y, feature)
        gains_with_new_feature.append((gain_with_feature, feature))
    
    best_new_gain, best_new_feature = max(gains_with_new_feature, key=lambda x: x[0])

    if best_new_gain <= 0.1:
        break
    
    selected_features.append(best_new_feature)
    remaining_features.remove(best_new_feature)

print(selected_features) # remove 'Q2', 'Q5', 'Q3', 'Q1', 'Gender'

