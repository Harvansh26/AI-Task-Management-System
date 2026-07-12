import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Load test data
X_test = joblib.load("models/X_test.pkl")
y_test = joblib.load("models/y_test.pkl")

# Load models
naive_bayes = joblib.load("models/naive_bayes.pkl")
svm = joblib.load("models/svm_classifier.pkl")

# Predictions
nb_pred = naive_bayes.predict(X_test)
svm_pred = svm.predict(X_test)

# Evaluation table
results = pd.DataFrame({
    "Model": [
        "Naive Bayes",
        "Support Vector Machine"
    ],

    "Accuracy": [
        accuracy_score(y_test, nb_pred),
        accuracy_score(y_test, svm_pred)
    ],

    "Precision": [
        precision_score(
            y_test,
            nb_pred,
            average="weighted"
        ),

        precision_score(
            y_test,
            svm_pred,
            average="weighted"
        )
    ],

    "Recall": [
        recall_score(
            y_test,
            nb_pred,
            average="weighted"
        ),

        recall_score(
            y_test,
            svm_pred,
            average="weighted"
        )
    ],

    "F1 Score": [
        f1_score(
            y_test,
            nb_pred,
            average="weighted"
        ),

        f1_score(
            y_test,
            svm_pred,
            average="weighted"
        )
    ]
})

print("\nMODEL COMPARISON\n")
print(results)

# Save report
results.to_csv(
    "outputs/metrics/model_comparison.csv",
    index=False
)

print("\nComparison report saved successfully!")