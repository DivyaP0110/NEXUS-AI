
import sqlite3
import pandas as pd
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

DB_PATH = DATA_DIR / "nexus.db"

# Connect to SQLite
connection = sqlite3.connect(DB_PATH)

print("\n==============================")
print(" NEXUS DATABASE MANAGEMENT")
print("==============================")

# CSV files to store
datasets = {
    "employees": "employees.csv",
    "tasks": "tasks.csv",
    "dependencies": "dependencies.csv",
    "recovery_plan": "recovery_plan.csv",
    "impact_report": "impact_report.csv"
}

# Create database tables
for table_name, filename in datasets.items():

    file_path = DATA_DIR / filename

    if file_path.exists():

        df = pd.read_csv(file_path)

        df.to_sql(
            table_name,
            connection,
            if_exists="replace",
            index=False
        )

        print(
            table_name,
            "table created:",
            len(df),
            "records"
        )

    else:
        print("File not found:", filename)

# SQL query: Find delayed tasks
print("\nDELAYED TASKS")

result = pd.read_sql_query(
    """
    SELECT task_id, task_name, status
    FROM tasks
    WHERE status = 'Delayed'
    """,
    connection
)

print(result.head(10).to_string(index=False))

# Close connection
connection.close()

print("\nNEXUS DATABASE COMPLETED!")
print("Database location:", DB_PATH)
