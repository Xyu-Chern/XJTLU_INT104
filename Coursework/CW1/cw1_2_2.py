import pandas as pd
import matplotlib.pyplot as plt

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

# # check dietributional noise 
# cleaned_data.hist()
# plt.show()

from sklearn.decomposition import PCA
import numpy as np
np.random.seed(0)


# basic check varriance ratio

cleaned_data= original_data.drop(['Index','Programme'], axis="columns")
# print(type(cleaned_data))
# print(cleaned_data)
cleaned_np = cleaned_data.to_numpy()
# print(cleaned_data)


# mean = cleaned_np.mean(axis=0)
# std = cleaned_np.std(axis=0)
# cleaned_np = (cleaned_np-mean)/std

# pca = PCA()
# transform_X= pca.fit_transform(cleaned_np)
# # print(transform_X.shape)
# # print(pca.explained_variance_ratio_)
# # print(1 - pca.explained_variance_ratio_.sum())

# cumsum = np.cumsum(pca.explained_variance_ratio_)
# d = np.argmax(cumsum >= 0.95) + 1
# print(d)
# plt.plot(cumsum, linewidth=3)
# plt.xlabel("Dimensions")
# plt.ylabel("Explained Variance")
# plt.show()


# DO A CIRCLE


# checkpoint = ""
# keyList=['Gender', 'Grade', 'Total', 'MCQ', 'Q1', 'Q2','Q3', 'Q4', 'Q5']
# for i in keyList:
#     cleaned_data_i = cleaned_data.drop([i], axis="columns")
#     cleaned_np_i = cleaned_data_i.to_numpy()
#     mean_i = cleaned_np_i.mean(axis=0)
#     std_i= cleaned_np_i.std(axis=0)
#     cleaned_np_i = (cleaned_np_i-mean_i)/std_i
#     pca = PCA()
#     transform_X_i= pca.fit_transform(cleaned_np_i)
#     ratio =pca.explained_variance_ratio_
#     cumsum = np.cumsum(ratio)
#     d = np.argmax(cumsum >= 0.9) + 1
#     print(d)
#     if d <=3 :
#         print(i)
#         checkpoint =" get !"
#         print(checkpoint)
# if checkpoint != " get !":
#     print("lose in 1 lost")


# checkpoint = ""
# keyList=['Gender', 'Grade', 'Total', 'MCQ', 'Q1', 'Q2','Q3', 'Q4', 'Q5']
# for i in range(9):
#     for j in range(i+1,9):
#         cleaned_data_i = cleaned_data.drop([keyList[i],keyList[j]], axis="columns")
#         cleaned_np_i = cleaned_data_i.to_numpy()
#         mean_i = cleaned_np_i.mean(axis=0)
#         std_i= cleaned_np_i.std(axis=0)
#         cleaned_np_i = (cleaned_np_i-mean_i)/std_i
#         pca = PCA()
#         transform_X_i= pca.fit_transform(cleaned_np_i)
#         ratio =pca.explained_variance_ratio_
#         cumsum = np.cumsum(ratio)
#         d = np.argmax(cumsum >= 0.9) + 1
#         print(d)
#         if d <=3 :
#             print(keyList[i],keyList[j])
#             checkpoint =" get !"
#             print(checkpoint)
# if checkpoint != " get !":
#     print("lose in 2 lost")

# checkpoint = ""
# keyList=['Gender', 'Grade', 'Total', 'MCQ', 'Q1', 'Q2','Q3', 'Q4', 'Q5']
# for i in range(9):
#     for j in range(i+1,9):
#         for k in range(j+1,9):
#             cleaned_data_i = cleaned_data.drop([keyList[i],keyList[j],keyList[k]], axis="columns")
#             cleaned_np_i = cleaned_data_i.to_numpy()
#             mean_i = cleaned_np_i.mean(axis=0)
#             std_i= cleaned_np_i.std(axis=0)
#             cleaned_np_i = (cleaned_np_i-mean_i)/std_i
#             pca = PCA()
#             transform_X_i= pca.fit_transform(cleaned_np_i)
#             ratio =pca.explained_variance_ratio_
#             cumsum = np.cumsum(ratio)
#             d = np.argmax(cumsum >= 0.9) + 1
#             print(d)
#             if d <=3 :
#                 print(keyList[i],keyList[j],keyList[k])
#                 checkpoint =" get !"
#                 print(checkpoint)
# if checkpoint != " get !":
#     print("lose in 3 lost")

# checkpoint = ""
# keyList=['Gender', 'Grade', 'Total', 'MCQ', 'Q1', 'Q2','Q3', 'Q4', 'Q5']
# for i in range(9):
#     for j in range(i+1,9):
#         for k in range(j+1,9):
#             for l in range(k+1,9):
#                 cleaned_data_i = cleaned_data.drop([keyList[i],keyList[j],keyList[k],keyList[l]], axis="columns")
#                 cleaned_np_i = cleaned_data_i.to_numpy()
#                 mean_i = cleaned_np_i.mean(axis=0)
#                 std_i= cleaned_np_i.std(axis=0)
#                 cleaned_np_i = (cleaned_np_i-mean_i)/std_i
#                 pca = PCA()
#                 transform_X_i= pca.fit_transform(cleaned_np_i)
#                 ratio =pca.explained_variance_ratio_
#                 cumsum = np.cumsum(ratio)
#                 d = np.argmax(cumsum >= 0.9) + 1
#                 print(d)
#                 if d <=3 :
#                     print(keyList[i],keyList[j],keyList[k],keyList[l])
#                     checkpoint =" get !"
#                     print(checkpoint)
# if checkpoint != " get !":
#     print("lose in 4 lost")

checkpoint = ""
keyList=['Gender', 'Grade', 'Total', 'MCQ', 'Q1', 'Q2','Q3', 'Q4', 'Q5']
for i in range(9):
    for j in range(i+1,9):
        for k in range(j+1,9):
            for l in range(k+1,9):
                for p in range(l+1,9):
                    cleaned_data_i = cleaned_data.drop([keyList[i],keyList[j],keyList[k],keyList[l],keyList[p]], axis="columns")
                    cleaned_np_i = cleaned_data_i.to_numpy()
                    mean_i = cleaned_np_i.mean(axis=0)
                    std_i= cleaned_np_i.std(axis=0)
                    cleaned_np_i = (cleaned_np_i-mean_i)/std_i
                    pca = PCA()
                    transform_X_i= pca.fit_transform(cleaned_np_i)
                    ratio =pca.explained_variance_ratio_
                    cumsum = np.cumsum(ratio)
                    d = np.argmax(cumsum >= 0.99) + 1
                    # print(d)
                    if d <=3:
                        print(keyList[i],keyList[j],keyList[k],keyList[l],keyList[p])
                        checkpoint =" get !"
if checkpoint != " get !":
    print("lose in 5 lost")


# Gender Grade Q1 Q2 Q5
    
# # 3d
# cleaned_data= original_data.drop(['Index','Programme','Gender', 'Grade', 'Q1', 'Q2', 'Q5'], axis="columns")
# cleaned_np = cleaned_data.to_numpy()
# mean = cleaned_np.mean(axis=0)
# std = cleaned_np.std(axis=0)
# cleaned_np = (cleaned_np-mean)/std

# pca = PCA(n_components=3)
# transform_X= pca.fit_transform(cleaned_np)
# print(1 - pca.explained_variance_ratio_.sum())

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')  
# colors = ['r', 'g', 'b', 'y']
# for i in range(len(original_data)):
#     programme = int(original_data.iloc[i]['Programme']) -1 
#     ax.scatter(transform_X[i, 0], transform_X[i, 1], transform_X[i, 2], c=colors[programme])
# ax.set_xlabel('Principal Component 1')
# ax.set_ylabel('Principal Component 2')
# ax.set_zlabel('Principal Component 3')
# plt.title('PCA method in visualization')
# plt.show()

checkpoint = ""
keyList=['Gender', 'Grade', 'Total', 'MCQ', 'Q1', 'Q2','Q3', 'Q4', 'Q5']
for i in range(9):
    for j in range(i+1,9):
        for k in range(j+1,9):
            for l in range(k+1,9):
                for p in range(l+1,9):
                    for q in range(p+1,9):
                        cleaned_data_i = cleaned_data.drop([keyList[i],keyList[j],keyList[k],keyList[l],keyList[p],keyList[q]], axis="columns")
                        cleaned_np_i = cleaned_data_i.to_numpy()
                        mean_i = cleaned_np_i.mean(axis=0)
                        std_i= cleaned_np_i.std(axis=0)
                        cleaned_np_i = (cleaned_np_i-mean_i)/std_i
                        pca = PCA()
                        transform_X_i= pca.fit_transform(cleaned_np_i)
                        ratio =pca.explained_variance_ratio_
                        cumsum = np.cumsum(ratio)
                        d = np.argmax(cumsum >= 0.978) + 1
                        # print(d)
                        if d <=2:
                            print(keyList[i],keyList[j],keyList[k],keyList[l],keyList[p],keyList[q])
                            checkpoint =" get !"
if checkpoint != " get !":
    print("lose in 6 lost")


# Gender Grade Q1 Q2 Q3 Q5

# 2d
cleaned_data= original_data.drop(['Index','Programme','Gender', 'Grade', 'Q1', 'Q2','Q3', 'Q5'], axis="columns")
cleaned_np = cleaned_data.to_numpy()
mean = cleaned_np.mean(axis=0)
std = cleaned_np.std(axis=0)
cleaned_np = (cleaned_np-mean)/std

pca = PCA(n_components=2)
transform_X= pca.fit_transform(cleaned_np)
print(1 - pca.explained_variance_ratio_.sum())


colors = ['r', 'g', 'b', 'y']
for i in range(len(original_data)):
    programme = int(original_data.iloc[i]['Programme']) -1 
    plt.scatter(transform_X[i, 0], transform_X[i, 1],  c=colors[programme])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA method in visualization')
plt.show()





