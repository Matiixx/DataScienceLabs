import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from sklearn.metrics import rand_score


def plot_dendrogram(model, **kwargs):
  counts = np.zeros(model.children_.shape[0])
  n_samples = len(model.labels_)
  for i, merge in enumerate(model.children_):
    current_count = 0
    for child_idx in merge:
      if child_idx < n_samples:
        current_count += 1
      else:
        current_count += counts[child_idx - n_samples]
    counts[i] = current_count

  linkage_matrix = np.column_stack(
      [model.children_, model.distances_, counts]
  ).astype(float)

  dendrogram(linkage_matrix, **kwargs)


df = pd.read_csv("seeds_dataset.dat", header=None, names=[
                 "Area", "P", "C", "length", "width", "asym", "lenghthGrove", "grain"], sep="\s+")

#######################################################################################################
# AVERAGE LINKAGE
ac_cosine_avg = AgglomerativeClustering(
  compute_distances=True, n_clusters=3, metric='cosine', linkage="average").fit(df.iloc[:, : -1])
ac_euclidean_avg = AgglomerativeClustering(
    compute_distances=True, n_clusters=3, metric='euclidean', linkage="average").fit(df.iloc[:, : -1])
ac_manhattan_avg = AgglomerativeClustering(
    compute_distances=True, n_clusters=3, metric='manhattan', linkage="average").fit(df.iloc[:, : -1])
plot_dendrogram(ac_cosine_avg, truncate_mode="level", p=3)
plt.savefig("AgglomerativeClusteringCosineAvg.png")
plt.close()
plot_dendrogram(ac_euclidean_avg, truncate_mode="level", p=3)
plt.savefig("AgglomerativeClusteringEucludeanAvg.png")
plt.close()
plot_dendrogram(ac_manhattan_avg, truncate_mode="level", p=3)
plt.savefig("AgglomerativeClusteringManhattanAvg.png")
plt.close()
#######################################################################################################
# WARD LINKAGE -> ONLY EUCLIDEAN METRIC
ac_euclidean_ward = AgglomerativeClustering(
  compute_distances=True, n_clusters=3, metric='euclidean', linkage="ward").fit(df.iloc[:, : -1])
#######################################################################################################
# COMPLETE LINKAGE
ac_cosine_max = AgglomerativeClustering(
  compute_distances=True, n_clusters=3, metric='cosine', linkage="complete").fit(df.iloc[:, : -1])
ac_euclidean_max = AgglomerativeClustering(
    compute_distances=True, n_clusters=3, metric='euclidean', linkage="complete").fit(df.iloc[:, : -1])
ac_manhattan_max = AgglomerativeClustering(
    compute_distances=True, n_clusters=3, metric='manhattan', linkage="complete").fit(df.iloc[:, : -1])
#######################################################################################################
# SINGLE LINKAGE
ac_cosine_single = AgglomerativeClustering(
  compute_distances=True, n_clusters=3, metric='cosine', linkage="single").fit(df.iloc[:, : -1])
ac_euclidean_single = AgglomerativeClustering(
    compute_distances=True, n_clusters=3, metric='euclidean', linkage="single").fit(df.iloc[:, : -1])
ac_manhattan_single = AgglomerativeClustering(
    compute_distances=True, n_clusters=3, metric='manhattan', linkage="single").fit(df.iloc[:, : -1])
#######################################################################################################
# KMEANS ON SET WITH 7 ATTRIBUTES
kmeans_clustering = KMeans(
  n_clusters=3, n_init='auto').fit_predict(df.iloc[:, : -1])
#######################################################################################################
# KMEANS ON PCA WITH 2 DIMENSIONS
pca = PCA(n_components=2)
pca.fit(df.iloc[:, : -1])
df_pca = pd.DataFrame(pca.transform(df.iloc[:, : -1]))

kmeans_pca = KMeans(
  n_clusters=3, n_init='auto').fit_predict(df_pca)
plt.scatter(df_pca[0], df_pca[1], c=kmeans_pca)
plt.savefig("KmeansPCA_plot.png")
plt.close()
#######################################################################################################
# RAND INDEX SCORES
print(f'Kmeans: {rand_score(kmeans_clustering, df.iloc[:, -1])}')
print(f'KmeansPCA: {rand_score(kmeans_pca, df.iloc[:, -1])}')

clustering = ac_cosine_avg.fit_predict(df.iloc[:, : -1])
print(
  f'AgglomerativeClustering cosine avg: {rand_score(clustering, df.iloc[:, -1])}')
clustering = ac_euclidean_avg.fit_predict(df.iloc[:, : -1])
print(
  f'AgglomerativeClustering euclidean avg: {rand_score(clustering, df.iloc[:, -1])}')
clustering = ac_manhattan_avg.fit_predict(df.iloc[:, : -1])
print(
  f'AgglomerativeClustering manhattan avg: {rand_score(clustering, df.iloc[:, -1])}')
clustering = ac_euclidean_ward.fit_predict(df.iloc[:, : -1])
print(
  f'AgglomerativeClustering euclidean ward: {rand_score(clustering, df.iloc[:, -1])}')
clustering = ac_cosine_max.fit_predict(df.iloc[:, : -1])
print(
  f'AgglomerativeClustering cosine max: {rand_score(clustering, df.iloc[:, -1])}')
clustering = ac_euclidean_max.fit_predict(df.iloc[:, : -1])
print(
  f'AgglomerativeClustering euclidean max: {rand_score(clustering, df.iloc[:, -1])}')
clustering = ac_manhattan_max.fit_predict(df.iloc[:, : -1])
print(
  f'AgglomerativeClustering manhattan max: {rand_score(clustering, df.iloc[:, -1])}')
clustering = ac_cosine_single.fit_predict(df.iloc[:, : -1])
print(
  f'AgglomerativeClustering cosine single: {rand_score(clustering, df.iloc[:, -1])}')
clustering = ac_euclidean_single.fit_predict(df.iloc[:, : -1])
print(
  f'AgglomerativeClustering euclidean single: {rand_score(clustering, df.iloc[:, -1])}')
clustering = ac_manhattan_single.fit_predict(df.iloc[:, : -1])
print(
  f'AgglomerativeClustering manhattan single: {rand_score(clustering, df.iloc[:, -1])}')
