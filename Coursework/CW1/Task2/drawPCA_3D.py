# Task2 Step 4

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

original_data = pd.read_excel("Coursework/CW_Data.xlsx")
cleaned_data= original_data.drop(['Index','Programme'], axis="columns")
cleaned_np = cleaned_data.to_numpy()
mean = cleaned_np.mean(axis=0)
std = cleaned_np.std(axis=0)
cleaned_np = (cleaned_np-mean)/std

pca = PCA(n_components=3)
transform_X= pca.fit_transform(cleaned_np)
ratio=int(pca.explained_variance_ratio_.sum()*100)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')  
colors = ['r', 'g', 'b', 'y']
for i in range(len(original_data)):
    programme = int(original_data.iloc[i]['Programme']) -1 
    ax.scatter(transform_X[i, 0], transform_X[i, 1], transform_X[i, 2], c=colors[programme], marker='o', edgecolor='k', s=30, alpha=0.8)
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
plt.title(f'PCA method in Reduce Dimension with {ratio}% explained variance ratio')
plt.show()
