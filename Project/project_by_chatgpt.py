import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

########################################################################
#  Command to chatGPT: Using python load the wine dataset to dataFrame.
########################################################################

data = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data", header=None)

########################################################################
#  Command to chatGPT: Do statistical summary of the data and discuss it.
########################################################################

print(data.describe())
print(data.info())

########################################################################
#  Command to chatGPT: Reduce data dimensionality
########################################################################

# Create a PCA object with the desired number of components
pca = PCA(n_components=3)

# Fit the PCA model to the data
pca.fit(data)

# Transform the data to the new lower-dimensional space
data_pca = pca.transform(data)

print(pca.explained_variance_ratio_)

#### ChatGPT inlcuded class attribute of the dataset while using PCA ####
#### Asked him to do this again ####
########################################################################
#  Command to chatGPT: Create a copy of the wine dataset and remove the first
#                       column with the class information, then reduce data dimensionality using PCA.
########################################################################

# create a copy of the dataset without the first column
data_copy = data.iloc[:, 1:]

# initialize PCA
pca = PCA(n_components=2)

# fit PCA on the data
pca.fit(data_copy)

# transform the data to the first two principal components
data_pca = pca.transform(data_copy)

print(pca.explained_variance_ratio_)


########################################################################
#  Command to chatGPT: Visualize the reduced dataset.
########################################################################

# create scatter plot of the reduced data
plt.scatter(data_pca[:, 0], data_pca[:, 1])
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.show()

class_col = data.iloc[:, 0]
colors = ['r', 'b', 'g']
for i in range(len(class_col)):
  plt.scatter(data_pca[i, 0], data_pca[i, 1], c=colors[int(class_col[i]) - 1])
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.show()

########################################################################
#  Command to chatGPT: Cluster the dataset (and evaluate clustering results with classification labels).
########################################################################

# initialize KMeans
kmeans = KMeans(n_clusters=3)

# fit KMeans on the reduced data
kmeans.fit(data_pca)

# predict the cluster labels
labels = kmeans.predict(data_pca)


# evaluate the clustering results using the adjusted Rand index
ari = adjusted_rand_score(class_col, labels)
print("Adjusted Rand index: ", ari)

########################################################################
#  Command to chatGPT: Split the dataset into training and testing.
########################################################################

# split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
  data.iloc[:, 1:], data.iloc[:, 0], test_size=0.2, random_state=42)

########################################################################
#  Command to chatGPT: Perform classification and evaluate its result.
########################################################################

# initialize logistic regression
clf = LogisticRegression()

# fit logistic regression on the training data
clf.fit(X_train, y_train)

# make predictions on the testing data
y_pred = clf.predict(X_test)

# calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)

#### Asked chatGPT to do classification using KNN ####
########################################################################
#  Command to chatGPT: Perform classification using KNN and evaluate its result.
########################################################################

# initialize KNN
knn = KNeighborsClassifier(n_neighbors=5)

# fit KNN on the training data
knn.fit(X_train, y_train)

# make predictions on the testing data
y_pred = knn.predict(X_test)

# calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)
