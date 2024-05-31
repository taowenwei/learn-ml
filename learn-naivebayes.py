from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Gaussian Naive Bayes classifier
gnb = GaussianNB()

# Train the classifier on the training set
gnb.fit(X_train, y_train)

# Make predictions on the test set
y_pred = gnb.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Output the results
print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(report)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Use only the first two features for visualization
X_vis = X[:, :2]
X_train_vis, X_test_vis, y_train, y_test = train_test_split(X_vis, y, test_size=0.3, random_state=42)

# Train the Gaussian Naive Bayes classifier on the reduced dataset
gnb_vis = GaussianNB()
gnb_vis.fit(X_train_vis, y_train)

# Create a mesh grid for plotting decision boundaries
x_min, x_max = X_vis[:, 0].min() - 1, X_vis[:, 0].max() + 1
y_min, y_max = X_vis[:, 1].min() - 1, X_vis[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# Predict the class for each point in the mesh grid
Z = gnb_vis.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundaries
plt.figure(figsize=(10, 6))
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

plt.contourf(xx, yy, Z, alpha=0.8, cmap=cmap_light)
plt.scatter(X_train_vis[:, 0], X_train_vis[:, 1], c=y_train, marker='o', edgecolor='k', s=50, cmap=cmap_bold)
plt.scatter(X_test_vis[:, 0], X_test_vis[:, 1], c=y_test, marker='x', edgecolor='k', s=50, cmap=cmap_bold)
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title('Naive Bayes decision boundaries with Iris dataset')
plt.show()