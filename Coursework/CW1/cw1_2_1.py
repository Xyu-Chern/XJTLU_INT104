import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
np.random.seed(0)

original_data = pd.read_excel("Coursework/CW_Data.xlsx")
print(original_data)
# Data clean

# check nan
cleaned_data= original_data.dropna(axis=1)
print(cleaned_data.keys()==original_data.keys())

# check gender noise 
s =0
for i in cleaned_data["Gender"]:
    if not( i ==1 or i ==2 ):
       s = s+1
if s == 0 :     
    print("No out gender")

# check programme noise 
s =0
for i in cleaned_data["Programme"]:
    if not( i ==1 or i ==2 or i ==3 or i ==4):
       s = s+1
if s == 0 :     
    print("No out Programme")
    
# check Grade noise 
s =0
for i in cleaned_data["Grade"]:
    if not(i ==2 or i ==3 ):
       s = s+1
if s == 0 :     
    print("No out Grade")

# basic check varriance ratio
cleaned_data= original_data.drop(['Index','Programme'], axis="columns")
cleaned_np = cleaned_data.to_numpy()
mean = cleaned_np.mean(axis=0)
std = cleaned_np.std(axis=0)
cleaned_np = (cleaned_np-mean)/std

# pca = PCA()
# transform_X= pca.fit_transform(cleaned_np)
# cumsum = np.cumsum(pca.explained_variance_ratio_)
# d = np.argmax(cumsum >= 0.90) + 1
# print(d)

# plt.plot(cumsum, linewidth=3)
# plt.xlabel("Dimensions")
# plt.ylabel("Explained Variance")
# plt.show()

pca = PCA(n_components = 7)
transform_X= pca.fit_transform(cleaned_np)
# print(1 - pca.explained_variance_ratio_.sum())


colors = ['r', 'g', 'b', 'y']
for m in range(7):
    for j in range(m+1,7):
        for i in range(len(original_data)):
            programme = int(original_data.iloc[i]['Programme']) -1 
            plt.scatter(transform_X[i, m], transform_X[i, j],  c=colors[programme])
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title(f'PCA method in visualization{m}+{j}')
        plt.show()