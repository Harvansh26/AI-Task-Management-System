import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("data/processed/cleaned_tasks.csv")

print("Dataset loaded successfully!")
print(df.head())

# Encode category
category_encoder = LabelEncoder()
df["category_encoded"] = category_encoder.fit_transform(df["category"])

# Encode priority
priority_encoder = LabelEncoder()
df["priority_encoded"] = priority_encoder.fit_transform(df["priority"])

# Features
X = df[
    [
        "category_encoded",
        "deadline_days",
        "workload"
    ]
]

# Target
y = df["priority_encoded"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# Save everything
joblib.dump(X_train, "models/priority_X_train.pkl")
joblib.dump(X_test, "models/priority_X_test.pkl")

joblib.dump(y_train, "models/priority_y_train.pkl")
joblib.dump(y_test, "models/priority_y_test.pkl")

joblib.dump(category_encoder, "models/category_encoder.pkl")
joblib.dump(priority_encoder, "models/priority_encoder.pkl")

print("\nPriority prediction dataset prepared successfully!")