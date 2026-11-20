
import pandas as pd
import numpy as np
import joblib
import json

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# =====================================================
# 1. LOAD DATA
# =====================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

df = pd.read_csv(DATA_DIR / "tasks.csv")

print("\n================================")
print(" NEXUS RANDOM FOREST AI ENGINE")
print("================================")

print("Dataset loaded successfully!")
print("Total records:", len(df))

# =====================================================
# 2. SELECT FEATURES AND TARGET
# =====================================================

features = [
    "complexity",
    "estimated_hours",
    "deadline_days",
    "dependency_count",
    "employee_workload_hours"
]

X = df[features].apply(pd.to_numeric, errors="coerce")
y = df["is_delayed"].astype(int)

# Remove invalid records
valid_rows = X.notna().all(axis=1) & y.notna()

X = X.loc[valid_rows]
y = y.loc[valid_rows]

print("\nSelected Features:")
print(features)

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

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# =====================================================
# 4. CREATE MODEL
# =====================================================

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42,
    class_weight="balanced"
)

# =====================================================
# 5. TRAIN MODEL
# =====================================================

print("\nTraining Random Forest model...")

model.fit(X_train, y_train)

print("Training completed!")

# =====================================================
# 6. PREDICTION
# =====================================================

predictions = model.predict(X_test)

# =====================================================
# 7. MODEL EVALUATION
# =====================================================

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

report = classification_report(
    y_test,
    predictions,
    labels=[0, 1],
    target_names=["On Track", "Delayed"],
    output_dict=True,
    zero_division=0
)

print("\n================================")
print(" RANDOM FOREST MODEL EVALUATION")
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
# 8. FEATURE IMPORTANCE
# =====================================================

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nImportant Risk Factors:")
print(importance.to_string(index=False))

# =====================================================
# 9. SAVE MODEL
# =====================================================

model_path = DATA_DIR / "nexus_delay_model.joblib"

joblib.dump(model, model_path)

# =====================================================
# 10. SAVE EVALUATION RESULTS
# =====================================================

evaluation = {
    "model": "Random Forest",
    "accuracy": round(float(accuracy) * 100, 2),
    "precision": round(float(precision) * 100, 2),
    "recall": round(float(recall) * 100, 2),
    "f1_score": round(float(f1) * 100, 2),
    "training_records": int(len(X_train)),
    "testing_records": int(len(X_test))
}

with open(DATA_DIR / "ml_metrics.json", "w") as file:
    json.dump(evaluation, file, indent=4)

pd.DataFrame(
    matrix,
    index=["Actual On Track", "Actual Delayed"],
    columns=["Predicted On Track", "Predicted Delayed"]
).to_csv(DATA_DIR / "ml_confusion_matrix.csv")

importance.to_csv(
    DATA_DIR / "ml_feature_importance.csv",
    index=False
)

print("\nModel and evaluation results saved!")

# =====================================================
# 11. TEST A NEW TASK
# =====================================================

new_task = pd.DataFrame([{
    "complexity": 5,
    "estimated_hours": 20,
    "deadline_days": 2,
    "dependency_count": 3,
    "employee_workload_hours": 65
}])

prediction = model.predict(new_task)[0]

probabilities = model.predict_proba(new_task)[0]

delayed_index = list(model.classes_).index(1)

delay_probability = probabilities[delayed_index] * 100

print("\n================================")
print(" NEW TASK RISK PREDICTION")
print("================================")

print(
    "Predicted Status:",
    "DELAY RISK" if prediction == 1 else "ON TRACK"
)

print(
    "Delay Probability:",
    round(delay_probability, 2),
    "%"
)

print("\nNEXUS AI ENGINE COMPLETED!")
