from sklearn import svm
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Generate synthetic dataset
X, y = make_blobs(n_samples=100, centers=2, random_state=42)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Soft Margin Linear SVM classifier
clf = svm.SVC(kernel='linear', C=1.0)

# Fit the model
clf.fit(X_train, y_train)

# Predict the labels
y_pred = clf.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))