# Task 4 Step 3

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd

original_data = pd.read_excel("Coursework/CW_Data.xlsx")
features = original_data.drop(['Index', 'Programme','Q2', 'Q5', 'Q3', 'Q1', 'Gender'], axis=1)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

tsne = TSNE(n_components=2, random_state=60, perplexity=100, learning_rate=50, n_iter=1000, early_exaggeration=70)
features_tsne = tsne.fit_transform(features_scaled)

colors = ['r', 'g', 'b', 'y']
for i in range(len(original_data)):
    programme = int(original_data.iloc[i]['Programme']) - 1
    plt.scatter(features_tsne[i, 0], features_tsne[i, 1], c=colors[programme], marker='o', edgecolor='k', s=30, alpha=0.6)

plt.title('t-SNE visualization')
plt.xlabel('t-SNE feature 1')
plt.ylabel('t-SNE feature 2')
plt.show()