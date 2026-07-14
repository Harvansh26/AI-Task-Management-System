import joblib
import pandas as pd

from sklearn.metrics import accuracy_score

# Load test data
X_test = joblib.load("models/priority_X_test.pkl")
y_test = joblib.load("models/priority_y_test.pkl")

# Load models
rf = joblib.load("models/random_forest.pkl")
xgb = joblib.load("models/xgboost.pkl")

# Accuracy
rf_acc = accuracy_score(y_test, rf.predict(X_test))
xgb_acc = accuracy_score(y_test, xgb.predict(X_test))

results = pd.DataFrame({
    "Model": [
        "Random Forest",
        "XGBoost"
    ],
    "Accuracy": [
        rf_acc,
        xgb_acc
    ]
})

print("\nPriority Model Comparison\n")
print(results)

results.to_csv(
    "outputs/metrics/priority_model_comparison.csv",
    index=False
)

# Save best model
if xgb_acc >= rf_acc:
    best = xgb
    best_name = "XGBoost"
else:
    best = rf
    best_name = "Random Forest"

joblib.dump(
    best,
    "models/best_priority_model.pkl"
)

print(f"\nBest Model: {best_name}")
print("Best model saved successfully!")