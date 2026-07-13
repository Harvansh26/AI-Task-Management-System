import joblib

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Load datasets
X_train = joblib.load("models/priority_X_train.pkl")
X_test = joblib.load("models/priority_X_test.pkl")

y_train = joblib.load("models/priority_y_train.pkl")
y_test = joblib.load("models/priority_y_test.pkl")

print("Priority datasets loaded successfully!")

# Create model
model = XGBClassifier(
    n_estimators=100,
    random_state=42,
    eval_metric="mlogloss"
)

# Train
model.fit(X_train, y_train)

print("\nXGBoost model trained successfully!")

# Predict
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy : {accuracy:.4f}")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# Save model
joblib.dump(
    model,
    "models/xgboost.pkl"
)

print("\nXGBoost model saved successfully!")