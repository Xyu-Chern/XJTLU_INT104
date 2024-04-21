# Task 4 Step 5

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


original_data = pd.read_excel("Coursework/CW_Data.xlsx")
cleaned_data = original_data.drop(['Index', 'Programme'], axis=1)

plt.figure(figsize=(12, 10))
sns.heatmap(cleaned_data.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Heatmap of Original Features')
plt.show()

extract_data = original_data.drop(['Index', 'Programme','Q2', 'Q5', 'Q3', 'Q1', 'Gender'], axis=1)
plt.figure(figsize=(8, 6))
sns.heatmap(extract_data .corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Heatmap of Selected Features After MI')
plt.show()
