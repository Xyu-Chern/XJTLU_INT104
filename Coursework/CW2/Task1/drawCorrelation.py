# Task 1 Step 1

import pandas as pd
from sklearn.feature_selection import mutual_info_classif
import numpy as np

original_data = pd.read_excel("Coursework/CW_Data.xlsx")
cleaned_data = original_data.drop(['Index', 'Programme'], axis=1)
mean = cleaned_data.mean()
std = cleaned_data.std()
standardized_data = (cleaned_data - mean) / std

X = standardized_data.to_numpy()
y = original_data['Programme'].to_numpy()

mutual_info_avg_df = pd.DataFrame({
    'Feature': cleaned_data.columns,
    'Average Mutual Information': np.zeros(len(cleaned_data.columns))
})

num_seeds = 10
for seed in range(num_seeds):
    np.random.seed(seed)
    mutual_info_scores = mutual_info_classif(X, y)
    mutual_info_avg_df['Average Mutual Information'] += mutual_info_scores

mutual_info_avg_df['Average Mutual Information'] /= num_seeds
mutual_info_avg_df.sort_values(by='Average Mutual Information', ascending=False, inplace=True)

print(mutual_info_avg_df)

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