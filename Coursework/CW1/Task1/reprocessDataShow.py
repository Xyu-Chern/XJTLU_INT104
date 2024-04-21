# Task 1 Step 2
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np

original_data = pd.read_excel("Coursework/CW_Data.xlsx")
cleaned_data= original_data.drop(['Index','Programme'], axis="columns")

cleaned_np = cleaned_data.to_numpy()
min_np = cleaned_np.min(axis=0)
max_np= cleaned_np.max(axis=0)
normalized_data = (cleaned_np-min_np)/(max_np-min_np)

variances = np.var(normalized_data, axis=0)
sorted_indices = np.argsort(variances)[::-1]
# print(sorted_indices ) # [0 1 8 6 7 5 9 2 4 3]
sorted_normalized_data = normalized_data[:, sorted_indices]
sorted_normalized_df = pd.DataFrame(sorted_normalized_data, columns=cleaned_data.columns[sorted_indices])

# Plot the sorted boxplot
plt.figure(figsize=(10, 6))
plt.boxplot(sorted_normalized_df, labels=sorted_normalized_df.columns)
plt.title('Box Plot - Sorted by Variance from high to low')  
plt.xlabel('Variables without Index and Programme') 
plt.ylabel('Normalized Values')  
plt.xticks(rotation=20)  
plt.show()



