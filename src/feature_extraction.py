import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Load the cleaned dataset
df = pd.read_csv("data/processed/cleaned_tasks.csv")

print("Dataset loaded successfully!")
print("Dataset Shape:", df.shape)

# Input feature (processed text)
X = df["processed_description"]

# Target variable (task category)
y = df["category"]

print("\nTarget Categories:")
print(y.value_counts())

# Create TF-IDF Vectorizer
vectorizer = TfidfVectorizer()

# Convert text into numerical features
X_tfidf = vectorizer.fit_transform(X)

print("\nTF-IDF Feature Matrix Shape:")
print(X_tfidf.shape)

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", X_train.shape[0])
print("Testing Samples:", X_test.shape[0])

# Save datasets
joblib.dump(X_train, "models/X_train.pkl")
joblib.dump(X_test, "models/X_test.pkl")
joblib.dump(y_train, "models/y_train.pkl")
joblib.dump(y_test, "models/y_test.pkl")

# Save vectorizer
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("\nTF-IDF Vectorizer saved successfully!")
print("Training and testing datasets saved!")