# Task2 Step 3

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

original_data = pd.read_excel("Coursework/CW_Data.xlsx")
cleaned_data= original_data.drop(['Index','Programme','Q2', 'Q5', 'Q3', 'Q1', 'Gender'], axis="columns")
cleaned_np = cleaned_data.to_numpy()
mean = cleaned_np.mean(axis=0)
std = cleaned_np.std(axis=0)
cleaned_np = (cleaned_np-mean)/std

pca = PCA(n_components=2)
transform_X= pca.fit_transform(cleaned_np)
ratio=int(pca.explained_variance_ratio_.sum()*100)

colors = ['r', 'g', 'b', 'y']
for i in range(len(original_data)):
    programme = int(original_data.iloc[i]['Programme']) -1 
    plt.scatter(transform_X[i, 0], transform_X[i, 1],  c=colors[programme], marker='o', edgecolor='k', s=30, alpha=0.8)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title(f'PCA method in Reduce Dimension with {ratio}% explained variance ratio')
plt.show()
