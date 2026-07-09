import os
import re
import pandas as pd
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


# Download required NLTK resources
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")


# Create processed data directory
os.makedirs("data/processed", exist_ok=True)


# Load raw dataset
df = pd.read_csv("data/raw/tasks.csv")

print("Raw dataset loaded successfully!")
print("Dataset shape:", df.shape)


# Initialize NLP tools
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))


def preprocess_text(text):
    # Convert text to lowercase
    text = str(text).lower()

    # Remove numbers and special characters
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Tokenization
    tokens = word_tokenize(text)

    # Stopword removal
    tokens = [
        word
        for word in tokens
        if word not in stop_words
    ]

    # Stemming
    stemmed_tokens = [
        stemmer.stem(word)
        for word in tokens
    ]

    # Join processed tokens
    processed_text = " ".join(stemmed_tokens)

    return processed_text


# Apply NLP preprocessing
df["processed_description"] = df["description"].apply(
    preprocess_text
)


# Check missing values
print("\nMissing values:")
print(df.isnull().sum())


# Remove duplicate rows
df = df.drop_duplicates()


# Save cleaned and preprocessed dataset
df.to_csv(
    "data/processed/cleaned_tasks.csv",
    index=False
)


print("\nNLP preprocessing completed successfully!")

print("\nOriginal vs Processed Task Descriptions:")

print(
    df[
        [
            "description",
            "processed_description"
        ]
    ].head(10)
)

print("\nCleaned dataset shape:", df.shape)

print(
    "\nProcessed dataset saved at "
    "data/processed/cleaned_tasks.csv"
)