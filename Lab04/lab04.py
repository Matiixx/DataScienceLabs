import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

df = pd.read_csv("seeds_dataset.dat", header=None, names=[
                 "Area", "P", "C", "length", "width", "asym", "lenghthGrove", "grain"], sep="\s+")

kf = KFold(n_splits=5)
kf.get_n_splits(df)
##################################################################
# Changing number of neighbors
##################################################################
print(f'n_neighbors=5')
neigh = KNeighborsClassifier(n_neighbors=5)
for i, (train_index, test_index) in enumerate(kf.split(df)):
  print(f"[{i}]:", end=" ")
  train_data = df.iloc[train_index]
  test_data = df.iloc[test_index]
  neigh.fit(train_data.iloc[:, : -1], train_data.iloc[:, -1])
  predicted_classes = neigh.predict(test_data.iloc[:, : -1])
  print(
    f'{accuracy_score(test_data.iloc[:, -1], predicted_classes)}',
    end="; ")

print()
print()
print(f'n_neighbors=3')
neigh = KNeighborsClassifier(n_neighbors=3)
for i, (train_index, test_index) in enumerate(kf.split(df)):
  print(f"[{i}]:", end=" ")
  train_data = df.iloc[train_index]
  test_data = df.iloc[test_index]
  neigh.fit(train_data.iloc[:, : -1], train_data.iloc[:, -1])
  predicted_classes = neigh.predict(test_data.iloc[:, : -1])
  print(
    f'{accuracy_score(test_data.iloc[:, -1], predicted_classes)}',
    end="; ")

print()
print()
print(f'n_neighbors=10')
neigh = KNeighborsClassifier(n_neighbors=10)
for i, (train_index, test_index) in enumerate(kf.split(df)):
  print(f"[{i}]:", end=" ")
  train_data = df.iloc[train_index]
  test_data = df.iloc[test_index]
  neigh.fit(train_data.iloc[:, : -1], train_data.iloc[:, -1])
  predicted_classes = neigh.predict(test_data.iloc[:, : -1])
  print(
    f'{accuracy_score(test_data.iloc[:, -1], predicted_classes)}',
    end="; ")

##################################################################
# Changing metrics
##################################################################
print()
print()
print(f'n_neighbors=5 metric=cityblock')
neigh = KNeighborsClassifier(n_neighbors=5, metric="cityblock")
for i, (train_index, test_index) in enumerate(kf.split(df)):
  print(f"[{i}]:", end=" ")
  train_data = df.iloc[train_index]
  test_data = df.iloc[test_index]
  neigh.fit(train_data.iloc[:, : -1], train_data.iloc[:, -1])
  predicted_classes = neigh.predict(test_data.iloc[:, : -1])
  print(
    f'{accuracy_score(test_data.iloc[:, -1], predicted_classes)}',
    end="; ")

print()
print()
print(f'n_neighbors=5 metric=cosine')
neigh = KNeighborsClassifier(n_neighbors=5, metric="cosine")
for i, (train_index, test_index) in enumerate(kf.split(df)):
  print(f"[{i}]:", end=" ")
  train_data = df.iloc[train_index]
  test_data = df.iloc[test_index]
  neigh.fit(train_data.iloc[:, : -1], train_data.iloc[:, -1])
  predicted_classes = neigh.predict(test_data.iloc[:, : -1])
  print(
    f'{accuracy_score(test_data.iloc[:, -1], predicted_classes)}',
    end="; ")

##################################################################
# Decision tree for 3rd fold
##################################################################
print()
print()
plt.figure()
(train_index, test_index) = next(kf.split(df))
(train_index, test_index) = next(kf.split(df))
(train_index, test_index) = next(kf.split(df))
train_data = df.iloc[train_index]
test_data = df.iloc[test_index]

clf = DecisionTreeClassifier().fit(
  train_data.iloc[:, : -1],
  train_data.iloc[:, -1]
)

plot_tree(clf, filled=True)
plt.title("Decision tree trained on 3rd fold data")
plt.savefig('decisionTree.png', dpi=2000)
