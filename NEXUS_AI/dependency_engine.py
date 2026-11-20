
import pandas as pd
import networkx as nx

from pathlib import Path

# --------------------------------
# 1. LOAD ENTERPRISE DATA
# --------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

employees = pd.read_csv(DATA_DIR / "employees.csv")
tasks = pd.read_csv(DATA_DIR / "tasks.csv")
dependencies = pd.read_csv(DATA_DIR / "dependencies.csv")

print("\n================================")
print(" NEXUS DEPENDENCY INTELLIGENCE")
print("================================")

# --------------------------------
# 2. CREATE TASK DEPENDENCY GRAPH
# --------------------------------

G = nx.DiGraph()

# Add all tasks as graph nodes
for _, task in tasks.iterrows():
    G.add_node(task["task_id"])

# Add dependency connections
for _, dep in dependencies.iterrows():

    G.add_edge(
        dep["predecessor_task_id"],
        dep["dependent_task_id"]
    )

print("\nTotal tasks:", G.number_of_nodes())
print("Dependency connections:", G.number_of_edges())

# --------------------------------
# 3. SELECT UNAVAILABLE EMPLOYEE
# --------------------------------

unavailable_employee = "E01"

employee = employees[
    employees["employee_id"] == unavailable_employee
]

if employee.empty:
    raise ValueError("Employee not found!")

employee_name = employee.iloc[0]["name"]

print("\nDisruption detected!")
print("Unavailable employee:", employee_name)

# --------------------------------
# 4. FIND DIRECTLY AFFECTED TASKS
# --------------------------------

direct_tasks = tasks[
    tasks["assigned_employee_id"] == unavailable_employee
]["task_id"].tolist()

print("\nDirectly affected tasks:")
print(direct_tasks)

# --------------------------------
# 5. FIND CASCADING IMPACT
# --------------------------------

affected_tasks = set(direct_tasks)

for task_id in direct_tasks:

    downstream_tasks = nx.descendants(
        G,
        task_id
    )

    affected_tasks.update(downstream_tasks)

affected_tasks = sorted(affected_tasks)

print("\n================================")
print(" CASCADING IMPACT ANALYSIS")
print("================================")

print("Total affected tasks:", len(affected_tasks))

# --------------------------------
# 6. DISPLAY AFFECTED TASK DETAILS
# --------------------------------

affected_details = tasks[
    tasks["task_id"].isin(affected_tasks)
].copy()

affected_details["impact_type"] = (
    affected_details["task_id"].apply(
        lambda x: "Directly Assigned"
        if x in direct_tasks
        else "Dependency Impact"
    )
)

print("\nAffected task details:")

print(
    affected_details[
        [
            "task_id",
            "task_name",
            "assigned_employee_id",
            "impact_type",
            "is_delayed"
        ]
    ].to_string(index=False)
)

# --------------------------------
# 7. SAVE IMPACT REPORT
# --------------------------------

affected_details.to_csv(
    DATA_DIR / "impact_report.csv",
    index=False
)

print("\nImpact report saved successfully!")

# --------------------------------
# 8. DISPLAY SUMMARY
# --------------------------------

print("\n================================")
print(" NEXUS IMPACT SUMMARY")
print("================================")

print("Unavailable employee:", employee_name)
print("Direct tasks:", len(direct_tasks))
print(
    "Indirectly affected tasks:",
    len(affected_tasks) - len(direct_tasks)
)

print("\nDEPENDENCY ANALYSIS COMPLETED!")
