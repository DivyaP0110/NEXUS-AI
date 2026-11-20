
import pandas as pd
import numpy as np
from pathlib import Path

# Random generator for repeatable sample data
rng = np.random.default_rng(42)

# --------------------------------
# 1. CREATE EMPLOYEE DATA
# --------------------------------

employees = [
    {"employee_id": "E01", "name": "Arun", "skills": "Python,SQL",
     "weekly_capacity_hours": 40, "available": True},

    {"employee_id": "E02", "name": "Priya", "skills": "Python,ML",
     "weekly_capacity_hours": 40, "available": True},

    {"employee_id": "E03", "name": "Rahul", "skills": "Java,SQL",
     "weekly_capacity_hours": 40, "available": True},

    {"employee_id": "E04", "name": "Divya", "skills": "ML,Python",
     "weekly_capacity_hours": 40, "available": True},

    {"employee_id": "E05", "name": "Karthik", "skills": "Java,Testing",
     "weekly_capacity_hours": 40, "available": True},

    {"employee_id": "E06", "name": "Meena", "skills": "SQL,Testing",
     "weekly_capacity_hours": 40, "available": True},

    {"employee_id": "E07", "name": "Vikram", "skills": "ML,Java",
     "weekly_capacity_hours": 40, "available": True},

    {"employee_id": "E08", "name": "Sneha", "skills": "Python,Testing",
     "weekly_capacity_hours": 40, "available": True}
]

employee_df = pd.DataFrame(employees)

# --------------------------------
# 2. CREATE TASK DATA
# --------------------------------

skills = ["Python", "SQL", "Java", "ML", "Testing"]

task_names = [
    "Develop API", "Database Design", "Model Training",
    "Software Testing", "Data Processing",
    "System Integration", "Bug Fixing", "Report Generation"
]

tasks = []
dependencies = []

for i in range(1, 61):

    task_id = f"T{i:03d}"

    required_skill = rng.choice(skills)

    # Find employees having the required skill
    matching_employees = employee_df[
        employee_df["skills"].str.contains(required_skill)
    ]

    assigned_employee = rng.choice(
        matching_employees["employee_id"].tolist()
    )

    complexity = int(rng.integers(1, 6))
    estimated_hours = int(rng.integers(2, 25))
    deadline_days = int(rng.integers(1, 15))

    # Generate task dependencies
    possible_previous = [f"T{j:03d}" for j in range(1, i)]

    dependency_count = min(
        len(possible_previous),
        int(rng.integers(0, 3))
    )

    selected_dependencies = (
        rng.choice(
            possible_previous,
            size=dependency_count,
            replace=False
        ).tolist()
        if dependency_count > 0 else []
    )

    for predecessor in selected_dependencies:
        dependencies.append({
            "predecessor_task_id": predecessor,
            "dependent_task_id": task_id
        })

    tasks.append({
        "task_id": task_id,
        "task_name": rng.choice(task_names),
        "assigned_employee_id": assigned_employee,
        "required_skill": required_skill,
        "complexity": complexity,
        "estimated_hours": estimated_hours,
        "deadline_days": deadline_days,
        "dependency_count": dependency_count
    })

task_df = pd.DataFrame(tasks)

# --------------------------------
# 3. CALCULATE EMPLOYEE WORKLOAD
# --------------------------------

workloads = task_df.groupby(
    "assigned_employee_id"
)["estimated_hours"].sum()

employee_df["current_workload_hours"] = (
    employee_df["employee_id"].map(workloads).fillna(0)
)

task_df["employee_workload_hours"] = (
    task_df["assigned_employee_id"]
    .map(workloads)
)

# --------------------------------
# 4. GENERATE DELAY LABELS
# --------------------------------

risk_score = (
    -2.5
    + 0.32 * task_df["complexity"]
    + 0.035 * task_df["estimated_hours"]
    + 0.012 * task_df["employee_workload_hours"]
    + 0.38 * task_df["dependency_count"]
    - 0.11 * task_df["deadline_days"]
)

delay_probability = 1 / (1 + np.exp(-risk_score))

task_df["is_delayed"] = (
    rng.random(len(task_df)) < delay_probability
).astype(int)

task_df["status"] = np.where(
    task_df["is_delayed"] == 1,
    "Delayed",
    "On Track"
)

# --------------------------------
# 5. SAVE ALL DATASETS
# --------------------------------

data_folder = Path(__file__).resolve().parent / "data"
data_folder.mkdir(exist_ok=True)

employee_df.to_csv(
    data_folder / "employees.csv", index=False
)

task_df.to_csv(
    data_folder / "tasks.csv", index=False
)

dependency_df = pd.DataFrame(
    dependencies,
    columns=["predecessor_task_id", "dependent_task_id"]
)

dependency_df.to_csv(
    data_folder / "dependencies.csv", index=False
)

# --------------------------------
# 6. DISPLAY RESULTS
# --------------------------------

print("\n================================")
print("       NEXUS DATA GENERATOR")
print("================================")

print("Employees created:", len(employee_df))
print("Tasks created:", len(task_df))
print("Dependencies created:", len(dependency_df))

print("Delayed tasks:", int(task_df["is_delayed"].sum()))
print("On-track tasks:", int((task_df["is_delayed"] == 0).sum()))

print("\nDATASETS GENERATED SUCCESSFULLY!")
print("Saved inside:", data_folder)
