import joblib
import pandas as pd

from sklearn.metrics import accuracy_score

# Load test data
X_test = joblib.load("models/X_test.pkl")
y_test = joblib.load("models/y_test.pkl")

# Load models
naive_bayes = joblib.load("models/naive_bayes.pkl")
svm = joblib.load("models/svm_classifier.pkl")

# Calculate accuracy
nb_accuracy = accuracy_score(
    y_test,
    naive_bayes.predict(X_test)
)

svm_accuracy = accuracy_score(
    y_test,
    svm.predict(X_test)
)

print(f"Naive Bayes Accuracy : {nb_accuracy:.4f}")
print(f"SVM Accuracy         : {svm_accuracy:.4f}")

# Select best model
if svm_accuracy >= nb_accuracy:
    best_model = svm
    best_model_name = "Support Vector Machine"
else:
    best_model = naive_bayes
    best_model_name = "Naive Bayes"

# Save best model
joblib.dump(
    best_model,
    "models/best_model.pkl"
)

print("\nBest Model Selected :", best_model_name)
print("Best model saved as models/best_model.pkl")

# Save summary
summary = pd.DataFrame({
    "Best Model": [best_model_name],
    "Accuracy": [max(nb_accuracy, svm_accuracy)]
})

summary.to_csv(
    "outputs/metrics/best_model_summary.csv",
    index=False
)

print("Summary saved successfully!")