import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn.neighbors

df = pd.read_csv('haberman.csv', header=None, names=[
                 'age', 'year', 'positiveAxillary', 'class'], usecols=[0, 1, 2, 3])

colors = np.where(df['class'] == 1, 'g', 'r')

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

for index, row in df.iterrows():
    x = row['year']
    y = row['age']
    z = row['positiveAxillary']
    ax.scatter(x, y, z, c=colors[index])
ax.set_xlabel('year')
ax.set_ylabel('age')
ax.set_zlabel('positiveAxillary')

plt.show()

# 3 sigma
df = df.drop(columns='class', axis=0)
mean = df.mean(axis=0)
print('Mean values:')
print(mean)
print()

std = df.std(axis=0)
print('Standard deviation values:')
print(std)
print()


p = 0
sigmaOutliers = []
for rowindex, row in df.iterrows():
    for (colindex, col) in enumerate(row):
        if (abs(col - mean[colindex]) > 3 * std[colindex]):
            p += 1
            sigmaOutliers.append(rowindex)
            break

print(f'Number of 3sigma outliers: {p}')
print(f'Outliers: {sigmaOutliers}')
print()

# Distance based outliers
nb = sklearn.neighbors.NearestNeighbors(n_neighbors=5)
nb.fit(df)
dist, ind = nb.kneighbors(df, n_neighbors=5)
df2 = pd.DataFrame(dist)
df2 = pd.DataFrame(df2.sum(axis=1), columns=['distance'])
df2 = df2.sort_values(by=['distance'], ascending=False)
distanceOutliers = df2.head(5).index.values.tolist()
print(f'Distance outliers: {sorted(distanceOutliers)}')
print()
