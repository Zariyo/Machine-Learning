import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns

sns.set()

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

# Read dataset to pandas dataframe
dataset = pd.read_csv(url, names=names)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# breast_cancer = load_breast_cancer()
# X = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
# X = X[['mean area', 'mean compactness']]
# y = pd.Categorical.from_codes(breast_cancer.target, breast_cancer.target_names)
# y = pd.get_dummies(y, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123456)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# sns.scatterplot(
#     x='mean area',
#     y='mean compactness',
#     hue='benign',
#     data=X_test.join(y_test, how='outer')
# )
# plt.scatter(
#     X_test['mean area'],
#     X_test['mean compactness'],
#     c=y_pred,
#     cmap='coolwarm',
#     alpha=0.7
# )
# plt.show()

print("========== 3 Neighbours ==========")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

print("========== 5 Neighbours ==========")

knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

print("========== 11 Neighbours ==========")

knn = KNeighborsClassifier(n_neighbors=11, metric='euclidean')
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))