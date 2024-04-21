# Task 4 Step 2

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import LocallyLinearEmbedding

original_data = pd.read_excel("Coursework/CW_Data.xlsx")
cleaned_data= original_data.drop(['Index','Programme','Q2', 'Q5', 'Q3', 'Q1', 'Gender'], axis="columns")
cleaned_np = cleaned_data.to_numpy()
mean = cleaned_np.mean(axis=0)
std = cleaned_np.std(axis=0)
cleaned_np = (cleaned_np-mean)/std

lle =LocallyLinearEmbedding(n_components=2,n_neighbors=16, random_state=128)
transform_X= lle.fit_transform(cleaned_np)

colors = ['r', 'g', 'b', 'y']
for i in range(len(original_data)):
    programme = int(original_data.iloc[i]['Programme']) -1 
    plt.scatter(transform_X[i, 0], transform_X[i, 1],  c=colors[programme], marker='o', edgecolor='k', s=30, alpha=0.8)

plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.title('LLE visualization ')
plt.show()