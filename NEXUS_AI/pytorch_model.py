
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import json

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# =====================================================
# 1. SETUP
# =====================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

torch.manual_seed(42)
np.random.seed(42)

print("\n================================")
print(" NEXUS PYTORCH DEEP LEARNING")
print("================================")

# =====================================================
# 2. LOAD DATA
# =====================================================

df = pd.read_csv(DATA_DIR / "tasks.csv")

features = [
    "complexity",
    "estimated_hours",
    "deadline_days",
    "dependency_count",
    "employee_workload_hours"
]

X = df[features].values.astype(np.float32)
y = df["is_delayed"].astype(int).values

print("Dataset loaded:", len(df), "records")

# =====================================================
# 3. SPLIT DATA
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# =====================================================
# 4. NORMALIZE DATA
# =====================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train_tensor = torch.tensor(
    y_train.reshape(-1, 1),
    dtype=torch.float32
)

# =====================================================
# 5. BUILD NEURAL NETWORK
# =====================================================

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


model = NexusNetwork()

# =====================================================
# 6. LOSS AND OPTIMIZER
# =====================================================

criterion = nn.BCELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)

# =====================================================
# 7. TRAIN MODEL
# =====================================================

print("\nTraining PyTorch neural network...")

for epoch in range(100):

    model.train()

    outputs = model(X_train)

    loss = criterion(outputs, y_train_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:

        print(
            f"Epoch [{epoch + 1}/100] "
            f"Loss: {loss.item():.4f}"
        )

print("\nTraining completed!")

# =====================================================
# 8. EVALUATE MODEL
# =====================================================

model.eval()

with torch.no_grad():

    test_outputs = model(X_test)

    probabilities = test_outputs.numpy().flatten()

    predictions = (
        probabilities >= 0.5
    ).astype(int)

accuracy = accuracy_score(y_test, predictions)

precision = precision_score(
    y_test, predictions, zero_division=0
)

recall = recall_score(
    y_test, predictions, zero_division=0
)

f1 = f1_score(
    y_test, predictions, zero_division=0
)

matrix = confusion_matrix(
    y_test,
    predictions,
    labels=[0, 1]
)

print("\n================================")
print(" PYTORCH MODEL EVALUATION")
print("================================")

print("Accuracy:", round(accuracy * 100, 2), "%")
print("Precision:", round(precision * 100, 2), "%")
print("Recall:", round(recall * 100, 2), "%")
print("F1 Score:", round(f1 * 100, 2), "%")

print("\nClassification Report:")

print(classification_report(
    y_test,
    predictions,
    labels=[0, 1],
    target_names=["On Track", "Delayed"],
    zero_division=0
))

print("\nConfusion Matrix:")
print(matrix)

# =====================================================
# 9. SAVE MODEL AND SCALER
# =====================================================

torch.save(
    model.state_dict(),
    DATA_DIR / "nexus_pytorch_model.pth"
)

joblib.dump(
    scaler,
    DATA_DIR / "nexus_scaler.joblib"
)

# =====================================================
# 10. SAVE EVALUATION RESULTS
# =====================================================

evaluation = {
    "model": "PyTorch Neural Network",
    "accuracy": round(float(accuracy) * 100, 2),
    "precision": round(float(precision) * 100, 2),
    "recall": round(float(recall) * 100, 2),
    "f1_score": round(float(f1) * 100, 2),
    "training_records": int(len(X_train)),
    "testing_records": int(len(X_test))
}

with open(DATA_DIR / "pytorch_metrics.json", "w") as file:
    json.dump(evaluation, file, indent=4)

pd.DataFrame(
    matrix,
    index=["Actual On Track", "Actual Delayed"],
    columns=["Predicted On Track", "Predicted Delayed"]
).to_csv(DATA_DIR / "pytorch_confusion_matrix.csv")

print("\nModel and evaluation results saved!")

# =====================================================
# 11. TEST NEW TASK
# =====================================================

new_task = np.array([[
    5, 20, 2, 3, 65
]], dtype=np.float32)

new_task_scaled = scaler.transform(new_task)

new_task_tensor = torch.tensor(
    new_task_scaled,
    dtype=torch.float32
)

model.eval()

with torch.no_grad():
    probability = model(new_task_tensor).item()

print("\n================================")
print(" PYTORCH RISK PREDICTION")
print("================================")

print("Delay Probability:", round(probability * 100, 2), "%")

print(
    "Predicted Status:",
    "DELAY RISK" if probability >= 0.5 else "ON TRACK"
)

print("\nNEXUS DEEP LEARNING COMPLETED!")
