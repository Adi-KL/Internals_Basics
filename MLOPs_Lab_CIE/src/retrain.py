import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import json
import os
import joblib
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

mlflow.set_tracking_uri("file:./mlruns")

# Load original test set (same split as Task 1)
df_original = pd.read_csv("data/training_data.csv")
X_orig = df_original.drop("latency_ms", axis=1)
y_orig = df_original["latency_ms"]
X_train_orig, X_test, y_train_orig, y_test = train_test_split(
    X_orig, y_orig, test_size=0.2, random_state=42
)

# Champion model performance on test set
champion = joblib.load("models/best_model.pkl")
champion_preds = champion.predict(X_test)
champion_mae = mean_absolute_error(y_test, champion_preds)

# Combine data
df_new = pd.read_csv("data/new_data.csv")
df_combined = pd.concat([df_original, df_new], ignore_index=True)

X_combined = df_combined.drop("latency_ms", axis=1)
y_combined = df_combined["latency_ms"]
X_train_combined, _, y_train_combined, _ = train_test_split(
    X_combined, y_combined, test_size=0.2, random_state=42
)

# Retrain same model type (Ridge won Task 1)
retrained = Ridge(alpha=1.0)
retrained.fit(X_train_combined, y_train_combined)
retrained_preds = retrained.predict(X_test)
retrained_mae = mean_absolute_error(y_test, retrained_preds)

improvement = champion_mae - retrained_mae
threshold = 0.3
action = "promoted" if improvement >= threshold else "kept_champion"

if action == "promoted":
    joblib.dump(retrained, "models/best_model.pkl")

output = {
    "original_data_rows": len(df_original),
    "new_data_rows": len(df_new),
    "combined_data_rows": len(df_combined),
    "champion_mae": round(champion_mae, 4),
    "retrained_mae": round(retrained_mae, 4),
    "improvement": round(improvement, 4),
    "min_improvement_threshold": 0.3,
    "action": action,
    "comparison_metric": "mae"
}

with open("results/step4_s8.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"Champion MAE: {champion_mae:.4f}")
print(f"Retrained MAE: {retrained_mae:.4f}")
print(f"Improvement: {improvement:.4f}")
print(f"Action: {action}")
print("results/step4_s8.json saved")