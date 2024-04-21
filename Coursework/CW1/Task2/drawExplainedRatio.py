# Task2 Step 2

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

# basic check varriance ratio

original_data = pd.read_excel("Coursework/CW_Data.xlsx")
cleaned_data= original_data.drop(['Index','Programme'], axis="columns")
cleaned_np = cleaned_data.to_numpy()
mean = cleaned_np.mean(axis=0)
std = cleaned_np.std(axis=0)
cleaned_np = (cleaned_np-mean)/std

pca = PCA()
transform_X= pca.fit_transform(cleaned_np)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95)+1

plt.plot(cumsum, linewidth=3)
plt.title('Find appropriate pricipal components to reduce variance')  
plt.axis([0, len(cumsum)-1, 0, 1])
plt.xlabel("Dimensions from 0 to 8")
plt.ylabel("Explained Variance")
plt.plot([d-1, d-1], [0, cumsum[d-1]], "k:", linewidth=2)  
plt.plot([0, d-1], [0.95, 0.95], "k:", linewidth=2)
plt.plot(d-1, cumsum[d-1], "ko") 
plt.annotate("Elbow", xy=(7,cumsum[d-1]), xytext=(5.7, 0.75),arrowprops=dict(arrowstyle="->"), fontsize=13)
plt.grid(True)
plt.show()

