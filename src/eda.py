import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create graph output directory if it does not exist
os.makedirs("outputs/graphs", exist_ok=True)

# Load task dataset
df = pd.read_csv("data/raw/tasks.csv")

print("Dataset loaded successfully!")

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nTask Category Distribution:")
print(df["category"].value_counts())

print("\nPriority Distribution:")
print(df["priority"].value_counts())

print("\nStatistical Summary:")
print(df.describe())

# Graph 1: Task Category Distribution
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x="category")
plt.title("Task Category Distribution")
plt.xlabel("Task Category")
plt.ylabel("Number of Tasks")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("outputs/graphs/category_distribution.png")
plt.close()

# Graph 2: Priority Distribution
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x="priority")
plt.title("Task Priority Distribution")
plt.xlabel("Priority")
plt.ylabel("Number of Tasks")
plt.tight_layout()
plt.savefig("outputs/graphs/priority_distribution.png")
plt.close()

# Graph 3: Deadline Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df["deadline_days"], bins=15, kde=True)
plt.title("Task Deadline Distribution")
plt.xlabel("Deadline Days")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("outputs/graphs/deadline_distribution.png")
plt.close()

# Graph 4: Workload Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df["workload"], bins=10)
plt.title("Task Workload Distribution")
plt.xlabel("Workload")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("outputs/graphs/workload_distribution.png")
plt.close()

print("\nEDA completed successfully!")
print("Graphs saved inside outputs/graphs/")