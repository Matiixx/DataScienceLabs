import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import rand_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Loading dataset
df = pd.read_csv("wine.data", header=None, names=[
                 "Class", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium",
                 "Total phenols", "Flavanoids", "Nonflavanoid phenols", "Proanthocyanins",
                 "Color intensity", "Hue", "OD280/OD315 of diluted wines", "Proline"], sep=",")

df_classes = df.iloc[:, 0]
df_without_class = df.iloc[:, 1:]

print("Average of values:")
print(df.groupby('Class').mean())
print()

# Reducing data dimensionality
print("Reducing data dimensionality...")
pca = PCA(n_components=2)
pca.fit(df_without_class)

print(f'Variance ratio: {pca.explained_variance_ratio_}')
print(f'Variance: {pca.explained_variance_}')

df_pca = pd.DataFrame(pca.transform(df_without_class))
df_pca_with_classes = df_pca
df_pca_with_classes.insert(2, 2, list(df_classes))
groups = df_pca_with_classes.groupby(2)
print()

# Visualize the reduced dataset, save to file
print("Save to file visualization of the reduced dataset...")
for name, group in groups:
  plt.plot(group[0], group[1], marker="o", linestyle="", label=name)
plt.legend()
plt.savefig("reducedData.png")
print()

# Clustering the dataset
print("Clustering the dataset...")
kmeans_clustering = KMeans(
  n_clusters=3, n_init='auto').fit_predict(df_without_class)

print(
  f'Kmeans accurate on non-reduced dataset: {rand_score(kmeans_clustering, df_classes)}')

kmeans_clustering = KMeans(
  n_clusters=3, n_init='auto').fit_predict(df_pca)

print(
  f'Kmeans accurate on reduced dataset: {rand_score(kmeans_clustering, df_classes)}')
print()

# Splitting the dataset into training and testing
print("Splitting the dataset into training and testing...")
X_train, X_test, y_train, y_test = train_test_split(
  df_without_class, df_classes, test_size=0.33, random_state=121)

# Classification

classificationAccuracy = {}

for k in range(1, 11):
  for met in ['cosine', 'euclidean', 'manhattan']:
    knn = KNeighborsClassifier(n_neighbors=k, metric=met)
    knn.fit(X_train, y_train)

    if met in classificationAccuracy.keys():
      classificationAccuracy[met].append(knn.score(X_test, y_test))
    else:
      classificationAccuracy[met] = [knn.score(X_test, y_test)]

print(pd.DataFrame(data=classificationAccuracy))
print(pd.DataFrame(data=classificationAccuracy).mean())
