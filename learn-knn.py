from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
iris = load_iris()
# Convert to a DataFrame
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
# Mapping target integers to species names
iris_df['species'] = iris.target
iris_df['species'] = iris_df['species'].map(
    {0: 'setosa', 1: 'versicolor', 2: 'virginica'})
print(iris_df.head())

# Features and target variable
X = iris_df.drop(columns='species')
y = iris_df['species']
# Split the dataset
# The choice of 42 is arbitrary; any integer can be used. 
# The number 42 is often used as a reference to "The Hitchhiker's Guide to the Galaxy" by Douglas Adams, 
# where 42 is the "answer to the ultimate question of life, the universe, and everything."
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape)
print(X_train[0: 4])

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
print(X_train[0: 4])
X_test = scaler.transform(X_test)
# print(X_test.head())

# Initialize the KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)
# Train the classifier
knn.fit(X_train, y_train)

# Make predictions
y_pred = knn.predict(X_test)
print(y_pred)
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Classification report
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Confusion matrix
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
