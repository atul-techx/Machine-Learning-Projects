from sklearn.cluster import KMeans

# Training data
x = [[20], [30], [40], [50], [60], [60]]

# Choose clusters less than or equal to number of samples
model = KMeans(n_clusters=4)  # try 2 clusters

# Fit the model
model.fit(x)

# Predict cluster for 60
print(model.predict([[60]]))