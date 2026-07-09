import pandas as pd
import random

random.seed(42)

tasks = [
    ("Fix login authentication bug", "Bug Fix"),
    ("Resolve payment gateway error", "Bug Fix"),
    ("Fix application crash issue", "Bug Fix"),
    ("Debug user registration problem", "Bug Fix"),
    ("Resolve database connection error", "Bug Fix"),

    ("Develop user profile page", "Development"),
    ("Create admin dashboard", "Development"),
    ("Implement payment module", "Development"),
    ("Build notification system", "Development"),
    ("Develop search functionality", "Development"),

    ("Test login functionality", "Testing"),
    ("Perform payment module testing", "Testing"),
    ("Execute regression testing", "Testing"),
    ("Test user registration module", "Testing"),
    ("Perform API testing", "Testing"),

    ("Prepare project documentation", "Documentation"),
    ("Update API documentation", "Documentation"),
    ("Create user manual", "Documentation"),
    ("Write technical documentation", "Documentation"),
    ("Prepare software requirement document", "Documentation"),

    ("Deploy application to production", "Deployment"),
    ("Configure production server", "Deployment"),
    ("Setup CI CD pipeline", "Deployment"),
    ("Deploy backend service", "Deployment"),
    ("Configure cloud infrastructure", "Deployment")
]

users = [
    "Developer_A",
    "Developer_B",
    "Tester_A",
    "Tester_B",
    "DevOps_A"
]

data = []

for task_id in range(1, 501):
    description, category = random.choice(tasks)

    deadline_days = random.randint(1, 30)
    workload = random.randint(1, 10)

    if deadline_days <= 5:
        priority = "High"
    elif deadline_days <= 15:
        priority = "Medium"
    else:
        priority = "Low"

    assigned_user = random.choice(users)

    data.append([
        task_id,
        description,
        category,
        deadline_days,
        workload,
        priority,
        assigned_user
    ])

columns = [
    "task_id",
    "description",
    "category",
    "deadline_days",
    "workload",
    "priority",
    "assigned_user"
]

df = pd.DataFrame(data, columns=columns)

df.to_csv("data/raw/tasks.csv", index=False)

print("Dataset generated successfully!")
print("Total number of tasks:", len(df))
print("Dataset shape:", df.shape)
print("\nFirst 5 tasks:")
print(df.head())