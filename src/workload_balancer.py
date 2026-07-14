import pandas as pd

# Load processed dataset
df = pd.read_csv("data/processed/cleaned_tasks.csv")

print("Dataset loaded successfully!")

# Current workload of each employee
workload = (
    df.groupby("assigned_user")["workload"]
      .sum()
      .reset_index()
)

workload.columns = ["Employee", "Total_Workload"]

print("\nCurrent Workload")

print(workload)

# Employee with least workload
best_employee = workload.loc[
    workload["Total_Workload"].idxmin()
]

print("\nRecommended Employee")

print(best_employee)

# Save report
workload.to_csv(
    "outputs/metrics/workload_report.csv",
    index=False
)

print("\nWorkload report saved successfully!")