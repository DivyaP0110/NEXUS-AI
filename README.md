# NEXUS – Enterprise Disruption & Recovery Intelligence System

## Project Overview

NEXUS is an AI-powered operational intelligence system designed to predict task delays, identify operational disruptions, analyze task dependencies, and recommend recovery strategies.

The system helps organizations understand potential risks and make better decisions to maintain business continuity.

## Key Features

- **Overview Dashboard:** Displays employee details, task status, and operational risk summaries.
- **Risk Intelligence:** Uses Random Forest and PyTorch models to predict task delay risks.
- **Disruption Simulator:** Simulates employee unavailability and displays the task dependency network.
- **Recovery Center:** Recommends alternative employees for affected tasks based on skills and workload.
- **AI Model Performance:** Compares model performance using accuracy, precision, recall, F1-score, and confusion matrices.
- **Database Management:** Uses SQLite to store and retrieve project data.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Web application and dashboard |
| Pandas & NumPy | Data processing |
| Scikit-learn | Random Forest machine learning |
| PyTorch | Neural network model |
| NetworkX | Task dependency analysis |
| Plotly | Interactive visualizations |
| SQLite | Database management |
| Joblib | Model storage |

## Project Structure

```text
NEXUS-AI/
│
├── NEXUS_AI/
│   ├── app.py
│   ├── data_generator.py
│   ├── database.py
│   ├── dependency_engine.py
│   ├── ml_engine.py
│   ├── pytorch_model.py
│   ├── recovery_engine.py
│   └── data/
│
└── requirements.txt
```

## How to Run the Project

1. Clone this repository.
2. Install the required Python libraries.
3. Open the project folder.
4. Run the application:

```bash
python -m streamlit run NEXUS_AI/app.py
```

## Machine Learning Models

NEXUS uses two models:

- Random Forest Classifier
- PyTorch Neural Network

Their performance is compared using standard classification metrics. The current results are based on the project's sample dataset and should be treated as preliminary.

## Live Application

[Open NEXUS Dashboard](https://nexus-recovery-intelligence.streamlit.app/)

## Project Purpose

To improve operational risk awareness, identify the impact of disruptions, and support faster recovery planning through AI and data-driven decision-making.

---

**NEXUS – Predict Disruptions. Understand Impact. Build Smarter Recovery Strategies.**
