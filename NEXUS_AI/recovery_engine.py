
import pandas as pd
from pathlib import Path

# --------------------------------
# 1. LOAD COMPANY DATA
# --------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

employees = pd.read_csv(DATA_DIR / "employees.csv")
tasks = pd.read_csv(DATA_DIR / "tasks.csv")

print("\n================================")
print(" NEXUS INTELLIGENT RECOVERY ENGINE")
print("================================")

# --------------------------------
# 2. SELECT UNAVAILABLE EMPLOYEE
# --------------------------------

unavailable_employee = "E01"

employee = employees[
    employees["employee_id"] == unavailable_employee
]

if employee.empty:
    raise ValueError("Employee not found!")

employee_name = employee.iloc[0]["name"]

print("\nDisruption:", employee_name, "is unavailable")

# --------------------------------
# 3. FIND EMPLOYEE'S TASKS
# --------------------------------

affected_tasks = tasks[
    tasks["assigned_employee_id"] == unavailable_employee
].copy()

print("Tasks requiring reassignment:", len(affected_tasks))

# --------------------------------
# 4. CALCULATE CURRENT WORKLOAD
# --------------------------------

workloads = tasks.groupby(
    "assigned_employee_id"
)["estimated_hours"].sum().to_dict()

for employee_id in employees["employee_id"]:
    workloads.setdefault(employee_id, 0)

# Exclude the unavailable employee
available_employees = employees[
    (employees["available"] == True) &
    (employees["employee_id"] != unavailable_employee)
].copy()

# Planning capacity: 100 hours over a simulated
# multi-week planning horizon.
planning_capacity = 100

# --------------------------------
# 5. FIND THE BEST EMPLOYEE
# --------------------------------

recovery_plan = []

for _, task in affected_tasks.iterrows():

    required_skill = task["required_skill"]
    task_hours = int(task["estimated_hours"])

    candidates = []

    for _, emp in available_employees.iterrows():

        employee_id = emp["employee_id"]

        employee_skills = [
            skill.strip()
            for skill in emp["skills"].split(",")
        ]

        # Skill must match
        if required_skill not in employee_skills:
            continue

        current_load = workloads[employee_id]

        projected_load = current_load + task_hours

        # Calculate workload score
        utilization = projected_load / planning_capacity

        # Prefer employees with lower workload
        candidates.append({
            "employee_id": employee_id,
            "employee_name": emp["name"],
            "current_load": current_load,
            "projected_load": projected_load,
            "utilization": utilization
        })

    # --------------------------------
    # 6. SELECT BEST CANDIDATE
    # --------------------------------

    if candidates:

        candidates.sort(
            key=lambda x: (
                x["projected_load"],
                x["employee_id"]
            )
        )

        best = candidates[0]

        assigned_employee = best["employee_id"]
        assigned_name = best["employee_name"]

        # Update workload after reassignment
        workloads[assigned_employee] += task_hours

        recovery_status = (
            "Within planning capacity"
            if best["projected_load"] <= planning_capacity
            else "Capacity exceeded - review required"
        )

    else:

        assigned_employee = "UNASSIGNED"
        assigned_name = "No suitable employee"

        best = {
            "current_load": 0,
            "projected_load": 0,
            "utilization": 0
        }

        recovery_status = "Manual intervention required"

    recovery_plan.append({

        "task_id": task["task_id"],

        "task_name": task["task_name"],

        "required_skill": required_skill,

        "original_employee": unavailable_employee,

        "new_employee": assigned_employee,

        "new_employee_name": assigned_name,

        "estimated_hours": task_hours,

        "projected_workload": best["projected_load"],

        "utilization": round(
            best["utilization"] * 100, 2
        ),

        "recovery_status": recovery_status
    })


# --------------------------------
# 7. CREATE RECOVERY REPORT
# --------------------------------

recovery_df = pd.DataFrame(recovery_plan)

recovery_df.to_csv(
    DATA_DIR / "recovery_plan.csv",
    index=False
)

# --------------------------------
# 8. DISPLAY RECOVERY RESULTS
# --------------------------------

print("\n================================")
print(" NEXUS RECOVERY RECOMMENDATIONS")
print("================================")

if len(recovery_df) > 0:

    print(
        recovery_df.to_string(index=False)
    )

else:

    print("No tasks require reassignment.")

# --------------------------------
# 9. RECOVERY SUMMARY
# --------------------------------

assigned_count = (
    recovery_df["new_employee"] != "UNASSIGNED"
).sum()

unassigned_count = (
    (recovery_df["new_employee"] == "UNASSIGNED") |
    (recovery_df["recovery_status"] != "Within planning capacity")
).sum()

print("\n================================")
print(" RECOVERY SUMMARY")
print("================================")

print("Total affected tasks:", len(recovery_df))

print("Successfully reassigned:", assigned_count)

print("Tasks requiring manual review:", unassigned_count)

print("\nRecovery report saved successfully!")

print("\nNEXUS RECOVERY ENGINE COMPLETED!")
