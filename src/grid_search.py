import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Load training data
X_train = joblib.load("models/priority_X_train.pkl")
y_train = joblib.load("models/priority_y_train.pkl")

print("Training data loaded successfully!")

# Base model
rf = RandomForestClassifier(random_state=42)

# Parameters to test
param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5]
}

# Grid Search
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("\nGrid Search Completed!")

print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nBest Accuracy:")
print(f"{grid_search.best_score_:.4f}")

# Save best model
joblib.dump(
    grid_search.best_estimator_,
    "models/random_forest_tuned.pkl"
)

print("\nTuned Random Forest model saved successfully!")