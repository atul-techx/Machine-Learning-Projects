from sklearn.cluster import KMeans
import numpy as np

X = np.array([[1,2],[1,4],[1,0],[10,2]])

kmeans = KMeans(n_clusters=2)
kmeans.fit(X)

print(kmeans.labels_)