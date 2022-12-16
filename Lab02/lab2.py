import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn.neighbors
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

df = pd.read_csv("seeds_dataset.dat", header=None, names=[
                 "Area", "P", "C", "length", "width", "asym", "lenghthGrove", "grain"], sep="\s+")

print(df)
##############################################################
pca = PCA(n_components=2)
pca.fit(df.iloc[:, : -1])

print(f'Variance ratio: {pca.explained_variance_ratio_}')
print(f'Variance: {pca.explained_variance_}')

df_pca = pd.DataFrame(pca.transform(df.iloc[:, : -1]))

df_pca.insert(2, 2, list(df.iloc[:, -1]))


groups = df_pca.groupby(2)

fig, axs = plt.subplots(2)

for name, group in groups:
    axs[0].plot(group[0], group[1], marker="o", linestyle="", label=name)
axs[0].legend()

pca = PCA(n_components=None)
pca.fit(df.iloc[:, : -1])

print()
print(f'Variance ratio: {pca.explained_variance_ratio_}')
print(f'Variance: {pca.explained_variance_}')

##############################################################
tsne = TSNE(n_components=2)

tsne.fit(df.iloc[:, :-1])

df_tsne = pd.DataFrame(tsne.fit_transform(df.iloc[:, :-1]))

# print(df_tsne)

df_tsne.insert(2, 2, list(df.iloc[:, -1]))

groups = df_tsne.groupby(2)
for name, group in groups:
    axs[1].plot(group[0], group[1], marker="o", linestyle="", label=name)

plt.legend()
plt.show()
