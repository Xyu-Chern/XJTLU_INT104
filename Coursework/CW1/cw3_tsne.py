from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
np.random.seed(0)

original_data = pd.read_excel("Coursework/CW_Data.xlsx")


features = original_data.drop(['Index', 'Programme'], axis=1)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(features_scaled)

# Apply t-SNE for visualization in 2D
tsne = TSNE(n_components=2, random_state=42)
features_tsne = tsne.fit_transform(features_scaled)

# Plot the clustered data in 2D space
plt.figure(figsize=(10, 6))
plt.scatter(features_tsne[:, 0], features_tsne[:, 1], c=clusters, cmap='viridis', marker='o', edgecolor='k', s=50, alpha=0.6)
plt.title('Data clustered into 4 categories (t-SNE visualization)')
plt.xlabel('t-SNE feature 1')
plt.ylabel('t-SNE feature 2')
plt.colorbar(label='Cluster ID')
plt.show()