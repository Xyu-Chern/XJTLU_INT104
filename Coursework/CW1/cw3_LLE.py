import pandas as pd
import matplotlib.pyplot as plt
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

from sklearn.manifold import LocallyLinearEmbedding
for j in range(41,60):
    lle =LocallyLinearEmbedding(n_components=2,n_neighbors=j)
    transform_X= lle.fit_transform(cleaned_np)

    colors = ['r', 'g', 'b', 'y']

    for i in range(len(original_data)):
        programme = int(original_data.iloc[i]['Programme']) -1 
        plt.scatter(transform_X[i, 0], transform_X[i, 1],  c=colors[programme])
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title(f'lle method in visualization {j} neighbours')
    plt.show()