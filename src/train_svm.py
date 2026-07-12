import joblib

from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Load datasets
X_train = joblib.load("models/X_train.pkl")
X_test = joblib.load("models/X_test.pkl")

y_train = joblib.load("models/y_train.pkl")
y_test = joblib.load("models/y_test.pkl")

print("Datasets loaded successfully!")

# Create SVM model
model = LinearSVC(random_state=42)

# Train model
model.fit(X_train, y_train)

print("\nSVM Model trained successfully!")

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:")
print(f"{accuracy:.4f}")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# Save model
joblib.dump(
    model,
    "models/svm_classifier.pkl"
)

print("\nSVM model saved successfully!")