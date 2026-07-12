import joblib

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Load training and testing datasets
X_train = joblib.load("models/X_train.pkl")
X_test = joblib.load("models/X_test.pkl")

y_train = joblib.load("models/y_train.pkl")
y_test = joblib.load("models/y_test.pkl")

print("Datasets loaded successfully!")

# Create Naive Bayes model
model = MultinomialNB()

# Train the model
model.fit(X_train, y_train)

print("\nModel trained successfully!")

# Make predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:")
print(f"{accuracy:.4f}")

# Classification Report
print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred
    )
)

# Confusion Matrix
print("\nConfusion Matrix:\n")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)

# Save model
joblib.dump(
    model,
    "models/naive_bayes.pkl"
)

print("\nNaive Bayes model saved successfully!")