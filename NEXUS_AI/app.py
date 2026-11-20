
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import joblib
import torch
import torch.nn as nn
import sqlite3
import json
import random
import math
import html
from pathlib import Path

# =========================================================
# NEXUS - ENTERPRISE DISRUPTION & RECOVERY INTELLIGENCE
# =========================================================

st.set_page_config(
    page_title="NEXUS | Enterprise Intelligence",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "nexus.db"

FEATURES = [
    "complexity",
    "estimated_hours",
    "deadline_days",
    "dependency_count",
    "employee_workload_hours"
]

# =========================================================
# SESSION STATE
# =========================================================

defaults = {
    "logged_in": False,
    "user_name": "",
    "user_role": "",
    "celebrate": False,
    "current_page": "Overview",
    "simulation_result": None,
    "recovery_result": None
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# =========================================================
# DEMO LOGIN
# =========================================================

DEMO_USERS = {
    "admin": {
        "password": "nexus123",
        "name": "Administrator",
        "role": "Admin"
    },
    "manager": {
        "password": "manager123",
        "name": "Project Manager",
        "role": "Manager"
    },
    "employee": {
        "password": "employee123",
        "name": "Employee User",
        "role": "Employee"
    },
    "guest": {
        "password": "guest123",
        "name": "Guest User",
        "role": "Guest"
    }
}

# =========================================================
# SQLITE DATABASE
# =========================================================

def get_connection():
    DATA_DIR.mkdir(exist_ok=True)
    return sqlite3.connect(DB_PATH)


def initialize_database():

    datasets = {
        "employees": "employees.csv",
        "tasks": "tasks.csv",
        "dependencies": "dependencies.csv",
        "recovery_plan": "recovery_plan.csv",
        "impact_report": "impact_report.csv"
    }

    connection = get_connection()

    try:
        existing_tables = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table'",
            connection
        )["name"].tolist()

        for table, filename in datasets.items():

            if table not in existing_tables:

                csv_path = DATA_DIR / filename

                if csv_path.exists():

                    df = pd.read_csv(csv_path)

                    df.to_sql(
                        table,
                        connection,
                        if_exists="replace",
                        index=False
                    )

    finally:
        connection.close()


def read_database_table(table_name):

    allowed_tables = [
        "employees",
        "tasks",
        "dependencies",
        "recovery_plan",
        "impact_report"
    ]

    if table_name not in allowed_tables:
        raise ValueError("Invalid database table")

    connection = get_connection()

    try:
        return pd.read_sql_query(
            f'SELECT * FROM "{table_name}"',
            connection
        )
    finally:
        connection.close()


def save_recovery_plan(df):

    connection = get_connection()

    try:
        df.to_sql(
            "recovery_plan",
            connection,
            if_exists="replace",
            index=False
        )
    finally:
        connection.close()


initialize_database()

# =========================================================
# LOAD PROJECT DATA FROM SQLITE
# =========================================================

@st.cache_data
def load_data():

    employee_df = read_database_table("employees")
    task_df = read_database_table("tasks")
    dependency_df = read_database_table("dependencies")

    employee_df["available"] = (
        employee_df["available"]
        .astype(str).str.lower()
        .isin(["true", "1", "yes"])
    )

    task_df["is_delayed"] = (
        task_df["is_delayed"]
        .astype(str).str.lower()
        .isin(["true", "1", "yes"])
    )

    return employee_df, task_df, dependency_df


try:
    employees_raw, tasks_raw, dependencies = load_data()

except Exception as error:
    st.error(f"Database loading error: {error}")
    st.stop()

employees = employees_raw.copy()
tasks = tasks_raw.copy()

employees["Employee"] = employees["name"]
employees["Skill"] = employees["skills"]

employees["Workload"] = (
    employees["current_workload_hours"] /
    employees["weekly_capacity_hours"].replace(0, 1) * 100
).round(1)

employees["Availability"] = employees["available"].map({
    True: "Available",
    False: "Unavailable"
})

tasks["Task"] = tasks["task_name"]
tasks["Task ID"] = tasks["task_id"]
tasks["Required Skill"] = tasks["required_skill"]

# =========================================================
# DASHBOARD CALCULATIONS
# =========================================================

total_employees = len(employees)
total_tasks = len(tasks)

delayed_tasks = int(tasks["is_delayed"].sum())
on_track_tasks = total_tasks - delayed_tasks

delay_percentage = (
    delayed_tasks / total_tasks * 100
    if total_tasks else 0
)

on_track_percentage = round(
    on_track_tasks / total_tasks * 100, 1
)

if delay_percentage >= 60:
    overall_risk = "High"
elif delay_percentage >= 40:
    overall_risk = "Medium"
else:
    overall_risk = "Low"

# =========================================================
# RANDOM FOREST MODEL
# =========================================================

@st.cache_resource
def load_ml_model():

    path = DATA_DIR / "nexus_delay_model.joblib"

    if path.exists():
        return joblib.load(path)

    return None


try:
    ml_model = load_ml_model()
except Exception:
    ml_model = None

# =========================================================
# PYTORCH MODEL
# =========================================================

class NexusNetwork(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


@st.cache_resource
def load_pytorch_model():

    model_path = DATA_DIR / "nexus_pytorch_model.pth"
    scaler_path = DATA_DIR / "nexus_scaler.joblib"

    if not model_path.exists() or not scaler_path.exists():
        return None, None

    model = NexusNetwork()

    state = torch.load(
        model_path,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(state)
    model.eval()

    scaler = joblib.load(scaler_path)

    return model, scaler


try:
    torch_model, torch_scaler = load_pytorch_model()
except Exception:
    torch_model, torch_scaler = None, None

# =========================================================
# RISK PREDICTION
# =========================================================

def predict_risk(task_row):

    values = task_row[FEATURES].to_frame().T

    ml_probability = None
    torch_probability = None

    if ml_model is not None:

        try:
            probabilities = ml_model.predict_proba(values)[0]
            classes = list(ml_model.classes_)

            if 1 in classes:
                ml_probability = (
                    float(probabilities[classes.index(1)]) * 100
                )

        except Exception:
            pass

    if torch_model is not None and torch_scaler is not None:

        try:
            scaled = torch_scaler.transform(
                values.values.astype(np.float32)
            )

            tensor = torch.tensor(
                scaled,
                dtype=torch.float32
            )

            with torch.no_grad():
                torch_probability = (
                    torch_model(tensor).item() * 100
                )

        except Exception:
            pass

    available = [
        score for score in [ml_probability, torch_probability]
        if score is not None
    ]

    if available:
        combined = float(np.mean(available))
        method = (
            "Random Forest + PyTorch"
            if len(available) == 2
            else "Available trained model"
        )
    else:
        combined = (
            100.0 if bool(task_row["is_delayed"]) else 0.0
        )
        method = "Historical status (not an ML prediction)"

    return ml_probability, torch_probability, combined, method


risk_records = []

for _, task in tasks.iterrows():

    ml_score, torch_score, combined, method = predict_risk(task)

    risk_records.append({
        "Task ID": task["task_id"],
        "Task": task["task_name"],
        "Required Skill": task["required_skill"],
        "Risk Score": round(combined, 1),
        "ML Probability": (
            round(ml_score, 1) if ml_score is not None else None
        ),
        "PyTorch Probability": (
            round(torch_score, 1) if torch_score is not None else None
        ),
        "Prediction Method": method,
        "Actual Status": task["status"]
    })

risk_predictions = pd.DataFrame(risk_records)

risk_by_skill = (
    risk_predictions.groupby("Required Skill")
    .agg(
        **{
            "Risk Score": ("Risk Score", "mean"),
            "Total Tasks": ("Task ID", "count")
        }
    )
    .reset_index()
)

risk_by_skill["Risk Score"] = (
    risk_by_skill["Risk Score"].round(1)
)

risk_by_skill = risk_by_skill.rename(
    columns={"Required Skill": "Department"}
)

# =========================================================
# LOAD EVALUATION RESULTS
# =========================================================

def load_json_file(filename):

    path = DATA_DIR / filename

    if path.exists():

        try:
            with open(path, "r") as file:
                return json.load(file)
        except Exception:
            return None

    return None


rf_metrics = load_json_file("ml_metrics.json")
torch_metrics = load_json_file("pytorch_metrics.json")


def load_confusion_matrix(filename):

    path = DATA_DIR / filename

    if path.exists():

        try:
            df = pd.read_csv(path, index_col=0)
            return df.values
        except Exception:
            return None

    return None


rf_matrix = load_confusion_matrix(
    "ml_confusion_matrix.csv"
)

torch_matrix = load_confusion_matrix(
    "pytorch_confusion_matrix.csv"
)

# =========================================================
# DEPENDENCY ENGINE
# =========================================================

def build_dependency_graph():

    graph = nx.DiGraph()

    graph.add_nodes_from(
        tasks["task_id"].astype(str)
    )

    for _, edge in dependencies.iterrows():

        graph.add_edge(
            str(edge["predecessor_task_id"]),
            str(edge["dependent_task_id"])
        )

    return graph


dependency_graph = build_dependency_graph()


def get_affected_tasks(task_id):

    task_id = str(task_id)

    if task_id not in dependency_graph:
        return []

    return list(
        nx.descendants(dependency_graph, task_id)
    )

# =========================================================
# RECOVERY ENGINE
# =========================================================

def generate_recovery_plan(unavailable_employee):

    affected = tasks[
        tasks["assigned_employee_id"].astype(str)
        == str(unavailable_employee)
    ].copy()

    workloads = employees.set_index(
        "employee_id"
    )["current_workload_hours"].to_dict()

    capacities = employees.set_index(
        "employee_id"
    )["weekly_capacity_hours"].to_dict()

    candidates_df = employees[
        (employees["available"] == True) &
        (employees["employee_id"].astype(str) != str(unavailable_employee))
    ].copy()

    plan = []

    for _, task in affected.iterrows():

        required_skill = str(task["required_skill"])
        hours = float(task["estimated_hours"])

        candidates = []

        for _, employee in candidates_df.iterrows():

            skills = [
                skill.strip()
                for skill in str(employee["skills"]).split(",")
            ]

            if required_skill not in skills:
                continue

            employee_id = employee["employee_id"]

            projected = (
                float(workloads[employee_id]) + hours
            )

            capacity = float(capacities[employee_id])

            utilization = (
                projected / capacity * 100
                if capacity > 0 else 100
            )

            candidates.append({
                "Employee ID": employee_id,
                "Employee Name": employee["name"],
                "Projected Workload Hours": projected,
                "Projected Utilization": round(utilization, 1),
                "Capacity": capacity
            })

        candidates.sort(
            key=lambda item: (
                item["Projected Utilization"],
                str(item["Employee ID"])
            )
        )

        if candidates:

            best = candidates[0]

            workloads[best["Employee ID"]] = (
                best["Projected Workload Hours"]
            )

            plan.append({
                "Task ID": task["task_id"],
                "Task": task["task_name"],
                "Required Skill": required_skill,
                "Original Employee": unavailable_employee,
                "Recommended Employee": best["Employee Name"],
                "Employee ID": best["Employee ID"],
                "Projected Workload Hours": best["Projected Workload Hours"],
                "Projected Utilization": best["Projected Utilization"],
                "Recovery Status": (
                    "Within capacity"
                    if best["Projected Workload Hours"] <= best["Capacity"]
                    else "Capacity exceeded - review required"
                )
            })

        else:

            plan.append({
                "Task ID": task["task_id"],
                "Task": task["task_name"],
                "Required Skill": required_skill,
                "Original Employee": unavailable_employee,
                "Recommended Employee": "No suitable employee",
                "Employee ID": "UNASSIGNED",
                "Projected Workload Hours": 0,
                "Projected Utilization": 0,
                "Recovery Status": "Manual intervention required"
            })

    return pd.DataFrame(plan)

# =========================================================
# HTML RENDERING
# =========================================================

def render(content):
    st.html(content.strip())

# =========================================================
# GLOBAL CSS
# =========================================================

render("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800;900&display=swap');

html, body, .stApp {
    font-family: 'Poppins', sans-serif !important;
}

.stApp {
    background:
        radial-gradient(circle at 80% 10%, #30204b 0%, transparent 35%),
        linear-gradient(135deg, #090617, #17112d, #101827);
    color: white;
}

.block-container {
    max-width: 1600px !important;
    padding-top: 2rem !important;
    padding-bottom: 3rem !important;
}

h1 {
    font-size: 45px !important;
    font-weight: 900 !important;
    color: #ffe1f7 !important;
}

h2 {
    font-size: 32px !important;
    font-weight: 800 !important;
    color: #f9d9ff !important;
}

h3 {
    font-size: 25px !important;
    color: #ffc5eb !important;
}

p, label, .stMarkdown li {
    font-size: 17px !important;
    line-height: 1.7 !important;
}

.nexus-logo {
    font-size: clamp(55px, 6vw, 95px);
    font-weight: 900;
    letter-spacing: 5px;
    white-space: nowrap;
    color: #f8baff;
    text-shadow: 0 0 30px #d66bff88;
}

.sidebar-logo {
    font-size: 38px;
    font-weight: 900;
    white-space: nowrap;
    color: #ffc0f0;
}

.nexus-subtitle {
    color: #f3d9fa;
    font-size: 21px;
    line-height: 1.7;
}

.nexus-feature {
    font-size: 19px;
    color: #f7d6fa;
    padding: 9px 0;
}

section[data-testid="stSidebar"] {
    background: #100b20;
    border-right: 2px solid #754c91;
    min-width: 270px !important;
}

section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p {
    font-size: 17px !important;
}

.stButton button {
    background: linear-gradient(90deg, #a855f7, #ec4899);
    color: white !important;
    border: none !important;
    border-radius: 15px !important;
    min-height: 52px !important;
    font-size: 18px !important;
    font-weight: 700 !important;
    transition: .25s ease;
}

.stButton button:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 28px #ec489988;
}

.stTextInput input {
    min-height: 52px !important;
    font-size: 18px !important;
    border-radius: 12px !important;
}

div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #241738, #38234c);
    padding: 22px !important;
    border-radius: 20px;
    border: 1px solid #9864b2;
    box-shadow: 0 8px 25px #00000040;
}

div[data-testid="stMetricValue"] {
    font-size: 36px !important;
    color: #ffb9e3 !important;
}

div[data-testid="stMetricLabel"] {
    font-size: 17px !important;
    color: #f5dfff !important;
}

.nexus-card {
    background: linear-gradient(135deg, #241738, #38234c);
    border: 1px solid #9864b2;
    border-radius: 22px;
    padding: 28px;
    margin: 16px 0;
    box-shadow: 0 8px 25px #00000040;
}

.nexus-card p {
    font-size: 19px;
    color: #f3e6fc;
}

.nexus-hero {
    background: linear-gradient(120deg, #49245f, #842e75, #263c69);
    border: 1px solid #c27fc2;
    border-radius: 25px;
    padding: 30px;
    margin-bottom: 25px;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #754c91;
    border-radius: 12px;
}

.stProgress > div > div {
    background-color: #ec4899;
}
</style>
""")

# =========================================================
# FIREWORKS
# =========================================================

def show_fireworks(message="WELCOME TO NEXUS ✨"):

    colors = [
        "#ff4fc3", "#ffd166", "#c084fc",
        "#53e5ff", "#ffffff", "#ff8fab"
    ]

    particles = []

    centers = [
        (12, 25), (30, 55), (50, 22),
        (70, 60), (88, 28), (20, 80),
        (80, 82), (50, 75)
    ]

    for burst, (x, y) in enumerate(centers):

        for i in range(25):

            angle = 2 * math.pi * i / 25
            distance = random.randint(70, 200)

            dx = math.cos(angle) * distance
            dy = math.sin(angle) * distance

            color = colors[i % len(colors)]

            particles.append(
                '<span class="nexus-particle" style="'
                f'--x:{x}%;--y:{y}%;'
                f'--dx:{dx:.0f}px;--dy:{dy:.0f}px;'
                f'--color:{color};--delay:{burst * 0.12:.2f}s;'
                '"></span>'
            )

    stars = ""

    for _ in range(30):

        left = random.randint(2, 96)
        top = random.randint(2, 90)

        stars += (
            f'<div class="nexus-star" '
            f'style="left:{left}%;top:{top}%">✦</div>'
        )

    render(f"""
    <style>
    #nexus-celebration {{
        position:fixed;
        inset:0;
        z-index:999999;
        pointer-events:none;
        overflow:hidden;
        animation:hideNexus 4.5s forwards;
    }}

    .nexus-particle {{
        position:absolute;
        left:var(--x);
        top:var(--y);
        width:9px;
        height:9px;
        border-radius:50%;
        background:var(--color);
        box-shadow:0 0 15px var(--color);
        animation:explode 1.7s ease-out var(--delay) forwards;
    }}

    @keyframes explode {{
        0% {{transform:translate(0,0) scale(.3);opacity:0;}}
        15% {{opacity:1;}}
        100% {{transform:translate(var(--dx),var(--dy)) scale(.1);opacity:0;}}
    }}

    .nexus-star {{
        position:absolute;
        color:white;
        font-size:30px;
        text-shadow:0 0 15px #ff8bd5;
        animation:starFloat 2.5s ease-out forwards;
    }}

    @keyframes starFloat {{
        0% {{transform:translateY(0) scale(.2);opacity:0;}}
        20% {{opacity:1;}}
        100% {{transform:translateY(200px) scale(1.5);opacity:0;}}
    }}

    .nexus-welcome {{
        position:absolute;
        top:48%;
        left:50%;
        transform:translate(-50%,-50%);
        text-align:center;
        font-size:clamp(28px,5vw,65px);
        font-weight:900;
        white-space:nowrap;
        color:#ffe5fa;
        text-shadow:0 0 25px #f472b6;
        animation:welcome 3.5s forwards;
    }}

    @keyframes welcome {{
        0% {{opacity:0;transform:translate(-50%,-50%) scale(.5);}}
        25%,65% {{opacity:1;transform:translate(-50%,-50%) scale(1);}}
        100% {{opacity:0;transform:translate(-50%,-50%) scale(1.2);}}
    }}

    @keyframes hideNexus {{
        0%,85% {{opacity:1;}}
        100% {{opacity:0;visibility:hidden;}}
    }}
    </style>

    <div id="nexus-celebration">
        {"".join(particles)}
        {stars}
        <div class="nexus-welcome">{html.escape(message)}</div>
    </div>
    """)

# =========================================================
# LOGIN PAGE
# =========================================================

if not st.session_state.logged_in:

    left, right = st.columns([1.2, 1], gap="large")

    with left:

        render("""
        <div style="padding:35px 10px;">
            <div class="nexus-logo">✦ NEXUS</div>
            <p class="nexus-subtitle">
                ENTERPRISE INTELLIGENCE SYSTEM
            </p>
            <h2>
                Enterprise Disruption &amp;
                Recovery Intelligence
            </h2>
            <p class="nexus-subtitle">
                Predict disruptions. Understand impact.
                Build smarter recovery strategies.
            </p>
            <br>
            <div class="nexus-feature">
                ✧ AI-Powered Risk Intelligence
            </div>
            <div class="nexus-feature">
                ✧ Intelligent Recovery Recommendations
            </div>
            <div class="nexus-feature">
                ✧ Enterprise Dependency Analysis
            </div>
            <div class="nexus-feature">
                ✧ Business Continuity Intelligence
            </div>
        </div>
        """)

    with right:

        render("""
        <div class="nexus-card">
            <h2>Welcome Back ✨</h2>
            <p>Sign in to access your NEXUS intelligence dashboard.</p>
        </div>
        """)

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("LOGIN →", use_container_width=True):

            if (
                username in DEMO_USERS and
                password == DEMO_USERS[username]["password"]
            ):

                st.session_state.logged_in = True
                st.session_state.user_name = DEMO_USERS[username]["name"]
                st.session_state.user_role = DEMO_USERS[username]["role"]
                st.session_state.celebrate = True

                st.rerun()

            else:
                st.error("Incorrect username or password.")

        st.markdown("### Demo Login Accounts")

        st.info("""
        Admin: admin / nexus123

        Manager: manager / manager123

        Employee: employee / employee123

        Guest: guest / guest123
        """)

    st.stop()

# =========================================================
# LOGIN CELEBRATION
# =========================================================

if st.session_state.celebrate:

    show_fireworks()

    st.session_state.celebrate = False

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    render("""
    <div style="padding:15px 0;">
        <div class="sidebar-logo">✦ NEXUS</div>
        <p style="font-size:14px;color:#d5b6ea;">
            RECOVERY INTELLIGENCE
        </p>
    </div>
    """)

    st.divider()

    st.write("Welcome,", st.session_state.user_name)
    st.write("Role:", st.session_state.user_role)

    page = st.radio(
        "MAIN MENU",
        [
            "Overview",
            "Risk Intelligence",
            "Disruption Simulator",
            "Recovery Center",
            "AI Model Performance"
        ],
        key="current_page"
    )

    st.divider()

    st.success("System Online")

    if st.button("LOGOUT ↗", use_container_width=True):

        st.session_state.logged_in = False
        st.session_state.user_name = ""
        st.session_state.user_role = ""
        st.session_state.celebrate = True

        st.rerun()

# =========================================================
# PAGE 1 - OVERVIEW
# =========================================================

if page == "Overview":

    render("""
    <h1>Welcome to NEXUS ✨</h1>

    <div class="nexus-hero">
        <h2>Enterprise Intelligence Dashboard</h2>
        <p>
            Monitor operational health, identify risks
            and prepare intelligent recovery strategies.
        </p>
    </div>
    """)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Employees", total_employees)
    c2.metric("Total Tasks", total_tasks)
    c3.metric("Risk Level", overall_risk)
    c4.metric("Tasks On Track", f"{on_track_percentage}%")

    st.markdown("## Operational Risk Overview")

    fig = px.bar(
        risk_by_skill,
        x="Department",
        y="Risk Score",
        color="Risk Score",
        color_continuous_scale=["#a78bfa", "#ec4899"],
        title="Average Predicted Risk by Required Skill"
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=17, color="#eee0ff"),
        title_font_size=24,
        height=450
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("## Task Status")

    status_counts = tasks["status"].value_counts().reset_index()
    status_counts.columns = ["Status", "Count"]

    fig2 = px.pie(
        status_counts,
        names="Status",
        values="Count",
        hole=0.55,
        title="Current Task Distribution",
        color_discrete_sequence=[
            "#ec4899", "#a855f7", "#53e5ff", "#ffd166"
        ]
    )

    fig2.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=17, color="#eee0ff"),
        height=400
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("## Recent Task Records")

    st.dataframe(
        tasks[
            ["task_id", "task_name", "assigned_employee_id",
             "required_skill", "status"]
        ],
        use_container_width=True,
        hide_index=True
    )

# =========================================================
# PAGE 2 - RISK INTELLIGENCE
# =========================================================

elif page == "Risk Intelligence":

    st.title("Risk Intelligence 🧠")

    st.write(
        "Analyze task delay risk using your trained prediction models."
    )

    a, b = st.columns(2)

    a.metric(
        "Random Forest Model",
        "Loaded" if ml_model is not None else "Unavailable"
    )

    b.metric(
        "PyTorch Model",
        "Loaded" if torch_model is not None else "Unavailable"
    )

    st.markdown("## Task Risk Predictions")

    st.dataframe(
        risk_predictions,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("## Predicted Risk Distribution")

    fig = px.bar(
        risk_predictions,
        x="Task",
        y="Risk Score",
        color="Risk Score",
        color_continuous_scale=["#53e5ff", "#a855f7", "#ec4899"],
        title="Predicted Delay Probability by Task"
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=15, color="#eee0ff"),
        height=500,
        xaxis_tickangle=-35
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("## Individual Task Risk Analysis")

    selected_task_id = st.selectbox(
        "Choose a task",
        risk_predictions["Task ID"].tolist()
    )

    selected = risk_predictions[
        risk_predictions["Task ID"] == selected_task_id
    ].iloc[0]

    st.metric(
        "Predicted Delay Probability",
        f"{selected['Risk Score']}%"
    )

    st.progress(
        float(selected["Risk Score"]) / 100
    )

    st.write(
        "Prediction method:",
        selected["Prediction Method"]
    )

    if selected["Risk Score"] >= 70:
        st.error("High risk: Immediate attention recommended.")
    elif selected["Risk Score"] >= 40:
        st.warning("Medium risk: Monitor this task.")
    else:
        st.success("Low risk: Continue normal monitoring.")

    st.download_button(
        "DOWNLOAD RISK REPORT",
        data=risk_predictions.to_csv(index=False),
        file_name="nexus_risk_predictions.csv",
        mime="text/csv"
    )

# =========================================================
# PAGE 3 - DISRUPTION SIMULATOR
# =========================================================

elif page == "Disruption Simulator":

    st.title("Disruption Simulator ⚡")

    st.write(
        "Simulate employee unavailability and identify "
        "the tasks affected by operational disruptions."
    )

    unavailable_employee = st.selectbox(
        "Select unavailable employee",
        employees["employee_id"].astype(str).tolist(),
        format_func=lambda x: (
            x + " - " +
            str(
                employees.loc[
                    employees["employee_id"].astype(str) == x,
                    "name"
                ].iloc[0]
            )
        )
    )

    severity = st.slider(
        "Disruption Severity",
        0,
        100,
        70
    )

    if st.button(
        "RUN DISRUPTION SIMULATION ⚡",
        use_container_width=True
    ):

        affected = tasks[
            tasks["assigned_employee_id"].astype(str)
            == unavailable_employee
        ].copy()

        affected_ids = set(
            affected["task_id"].astype(str)
        )

        cascading = set()

        for task_id in affected_ids:
            cascading.update(get_affected_tasks(task_id))

        cascading.difference_update(affected_ids)

        st.session_state.simulation_result = {
            "employee": unavailable_employee,
            "severity": severity,
            "affected": affected,
            "cascading": sorted(cascading)
        }

    result = st.session_state.simulation_result

    if result is not None:

        if result["employee"] == unavailable_employee:

            affected = result["affected"]
            cascading = result["cascading"]

            st.success("Disruption simulation completed.")

            a, b, c = st.columns(3)

            a.metric("Severity", f"{result['severity']}/100")
            b.metric("Directly Affected Tasks", len(affected))
            c.metric("Cascading Tasks", len(cascading))

            st.markdown("## Directly Affected Tasks")

            if affected.empty:
                st.info("No tasks are assigned to this employee.")
            else:
                st.dataframe(
                    affected,
                    use_container_width=True,
                    hide_index=True
                )

            st.markdown("## Cascading Impact")

            if cascading:

                cascading_df = tasks[
                    tasks["task_id"].astype(str).isin(cascading)
                ]

                st.warning(
                    f"{len(cascading)} additional dependent task(s) "
                    "may be affected."
                )

                st.dataframe(
                    cascading_df,
                    use_container_width=True,
                    hide_index=True
                )

            else:
                st.success(
                    "No downstream dependent tasks were found."
                )

    st.divider()

    st.markdown("## Actual Task Dependency Network")

    graph = dependency_graph

    if graph.number_of_nodes() > 0:

        positions = nx.spring_layout(
            graph,
            seed=42,
            k=0.8
        )

        edge_x, edge_y = [], []

        for source, target in graph.edges():

            x0, y0 = positions[source]
            x1, y1 = positions[target]

            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(color="#a78bfa", width=2),
            hoverinfo="none"
        )

        node_trace = go.Scatter(
            x=[positions[n][0] for n in graph.nodes()],
            y=[positions[n][1] for n in graph.nodes()],
            mode="markers+text",
            text=list(graph.nodes()),
            textposition="bottom center",
            textfont=dict(size=12, color="white"),
            marker=dict(
                size=20,
                color="#ec4899",
                line=dict(color="white", width=1)
            ),
            hovertext=list(graph.nodes()),
            hoverinfo="text"
        )

        fig = go.Figure(
            data=[edge_trace, node_trace]
        )

        fig.update_layout(
            height=550,
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#eee0ff"),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("No dependency relationships found.")

# =========================================================
# PAGE 4 - RECOVERY CENTER
# =========================================================

elif page == "Recovery Center":

    st.title("Recovery Center 🛡️")

    st.write(
        "Recommend suitable replacement employees based on "
        "skills, availability and projected workload."
    )

    st.markdown("## Employee Availability")

    display_employees = employees[
        [
            "employee_id",
            "Employee",
            "Skill",
            "weekly_capacity_hours",
            "current_workload_hours",
            "Workload",
            "Availability"
        ]
    ].copy()

    st.dataframe(
        display_employees,
        use_container_width=True,
        hide_index=True
    )

    unavailable_employee = st.selectbox(
        "Employee who is unavailable",
        employees["employee_id"].astype(str).tolist(),
        key="recovery_employee"
    )

    if st.button(
        "GENERATE RECOVERY PLAN ✨",
        use_container_width=True
    ):

        recovery_df = generate_recovery_plan(
            unavailable_employee
        )

        st.session_state.recovery_result = recovery_df

        # Save generated recovery plan in SQLite
        save_recovery_plan(recovery_df)

        st.success("Recovery plan saved to SQLite database.")

    recovery_df = st.session_state.recovery_result

    if recovery_df is not None:

        st.markdown("## Recovery Recommendations")

        if recovery_df.empty:

            st.success(
                "No tasks are assigned to this employee. "
                "No reassignment is required."
            )

        else:

            assigned_count = (
                recovery_df["Employee ID"] != "UNASSIGNED"
            ).sum()

            manual_count = len(recovery_df) - assigned_count

            a, b = st.columns(2)

            a.metric("Tasks Assigned", int(assigned_count))
            b.metric("Manual Review Required", int(manual_count))

            st.dataframe(
                recovery_df,
                use_container_width=True,
                hide_index=True
            )

            if manual_count:
                st.warning(
                    "Some tasks need manual intervention. "
                    "Check skill matching and employee capacity."
                )
            else:
                st.success(
                    "Suitable replacement employees were found "
                    "for all affected tasks."
                )

            st.download_button(
                "DOWNLOAD RECOVERY REPORT",
                data=recovery_df.to_csv(index=False),
                file_name="nexus_recovery_report.csv",
                mime="text/csv"
            )

    st.divider()

    st.markdown("## Current Task Status")

    st.dataframe(
        tasks[
            [
                "task_id",
                "task_name",
                "assigned_employee_id",
                "required_skill",
                "estimated_hours",
                "status"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )

# =========================================================
# PAGE 5 - AI MODEL PERFORMANCE
# =========================================================

elif page == "AI Model Performance":

    st.title("AI Model Performance 📊")

    st.write(
        "Compare Random Forest and PyTorch using saved test-set metrics."
    )

    if rf_metrics is None or torch_metrics is None:

        st.warning(
            "Evaluation files are missing. Run both model training scripts."
        )

        st.code(
            "python ml_engine.py\n"
            "python pytorch_model.py"
        )

    else:

        st.markdown("## Model Comparison")

        c1, c2 = st.columns(2)

        c1.metric(
            "Random Forest Accuracy",
            f"{rf_metrics.get('accuracy', 0)}%"
        )

        c2.metric(
            "PyTorch Accuracy",
            f"{torch_metrics.get('accuracy', 0)}%"
        )

        comparison_df = pd.DataFrame([
            {
                "Metric": "Accuracy",
                "Random Forest": rf_metrics.get("accuracy", 0),
                "PyTorch": torch_metrics.get("accuracy", 0)
            },
            {
                "Metric": "Precision",
                "Random Forest": rf_metrics.get("precision", 0),
                "PyTorch": torch_metrics.get("precision", 0)
            },
            {
                "Metric": "Recall",
                "Random Forest": rf_metrics.get("recall", 0),
                "PyTorch": torch_metrics.get("recall", 0)
            },
            {
                "Metric": "F1 Score",
                "Random Forest": rf_metrics.get("f1_score", 0),
                "PyTorch": torch_metrics.get("f1_score", 0)
            }
        ])

        st.markdown("## Performance Metrics")

        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True
        )

        chart_df = comparison_df.melt(
            id_vars="Metric",
            var_name="Model",
            value_name="Score"
        )

        fig = px.bar(
            chart_df,
            x="Metric",
            y="Score",
            color="Model",
            barmode="group",
            text="Score",
            color_discrete_sequence=["#ec4899", "#a78bfa"],
            title="Random Forest vs PyTorch"
        )

        fig.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside"
        )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(size=16, color="#eee0ff"),
            height=450,
            yaxis=dict(range=[0, 110])
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("## Training and Testing Records")

        record_df = pd.DataFrame([
            {
                "Model": "Random Forest",
                "Training Records": rf_metrics.get(
                    "training_records", 0
                ),
                "Testing Records": rf_metrics.get(
                    "testing_records", 0
                )
            },
            {
                "Model": "PyTorch",
                "Training Records": torch_metrics.get(
                    "training_records", 0
                ),
                "Testing Records": torch_metrics.get(
                    "testing_records", 0
                )
            }
        ])

        st.dataframe(
            record_df,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("## Confusion Matrices")

        cm1, cm2 = st.columns(2)

        labels = ["On Track", "Delayed"]

        if rf_matrix is not None:

            with cm1:

                st.markdown("### Random Forest")

                fig_rf = px.imshow(
                    rf_matrix,
                    x=labels,
                    y=labels,
                    text_auto=True,
                    color_continuous_scale="Purples",
                    labels={
                        "x": "Predicted",
                        "y": "Actual",
                        "color": "Count"
                    }
                )

                fig_rf.update_layout(
                    height=380,
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#eee0ff")
                )

                st.plotly_chart(
                    fig_rf,
                    use_container_width=True
                )

        else:
            with cm1:
                st.warning("Random Forest confusion matrix not found.")

        if torch_matrix is not None:

            with cm2:

                st.markdown("### PyTorch")

                fig_torch = px.imshow(
                    torch_matrix,
                    x=labels,
                    y=labels,
                    text_auto=True,
                    color_continuous_scale="Purples",
                    labels={
                        "x": "Predicted",
                        "y": "Actual",
                        "color": "Count"
                    }
                )

                fig_torch.update_layout(
                    height=380,
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#eee0ff")
                )

                st.plotly_chart(
                    fig_torch,
                    use_container_width=True
                )

        else:
            with cm2:
                st.warning("PyTorch confusion matrix not found.")

        st.markdown("## Feature Importance")

        importance_path = DATA_DIR / "ml_feature_importance.csv"

        if importance_path.exists():

            importance_df = pd.read_csv(importance_path)

            fig_importance = px.bar(
                importance_df,
                x="Importance",
                y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale=["#a78bfa", "#ec4899"],
                title="Random Forest Feature Importance"
            )

            fig_importance.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(size=16, color="#eee0ff"),
                height=420
            )

            st.plotly_chart(
                fig_importance,
                use_container_width=True
            )

        else:
            st.info("Feature importance file not found.")

        st.markdown("## Download Evaluation Reports")

        report1, report2 = st.columns(2)

        with report1:

            st.download_button(
                "DOWNLOAD RANDOM FOREST METRICS",
                data=json.dumps(rf_metrics, indent=4),
                file_name="random_forest_metrics.json",
                mime="application/json"
            )

        with report2:

            st.download_button(
                "DOWNLOAD PYTORCH METRICS",
                data=json.dumps(torch_metrics, indent=4),
                file_name="pytorch_metrics.json",
                mime="application/json"
            )

        st.info(
            "These metrics are calculated on the held-out test split. "
            "The dataset is small, so the results are preliminary."
        )

# =========================================================
# DATABASE STATUS
# =========================================================

with st.sidebar:

    st.divider()

    st.markdown("### Database Status")

    try:

        db_tasks = read_database_table("tasks")
        db_employees = read_database_table("employees")

        st.success("SQLite Connected")

        st.caption(f"Tasks stored: {len(db_tasks)}")
        st.caption(f"Employees stored: {len(db_employees)}")

    except Exception as error:
        st.error(f"Database error: {error}")

# =========================================================
# FOOTER
# =========================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center;color:#c9a9df;padding:15px;">
        ✦ NEXUS — Enterprise Disruption & Recovery Intelligence ✦
        <br>
        AI-Powered Operational Risk & Business Continuity System
    </div>
    """,
    unsafe_allow_html=True
)
