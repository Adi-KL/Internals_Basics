import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import json
import os
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)
mlflow.set_tracking_uri("file:./mlruns")
df = pd.read_csv("data/training_data.csv")
X = df.drop("latency_ms", axis=1)
y = df["latency_ms"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.set_experiment("gamelag-latency-ms")

results = []

# Ridge
with mlflow.start_run(run_name="Ridge") as run:
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    mlflow.log_param("alpha", 1.0)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.set_tag("team", "ml_engineering")
    mlflow.sklearn.log_model(model, "model")

    ridge_run_id = run.info.run_id
    results.append({"name": "Ridge", "mae": round(mae,4), "rmse": round(rmse,4), "r2": round(r2,4), "run_id": ridge_run_id})
    print(f"Ridge — MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")

# GradientBoosting
with mlflow.start_run(run_name="GradientBoosting") as run:
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.set_tag("team", "ml_engineering")
    mlflow.sklearn.log_model(model, "model")

    gb_run_id = run.info.run_id
    results.append({"name": "GradientBoosting", "mae": round(mae,4), "rmse": round(rmse,4), "r2": round(r2,4), "run_id": gb_run_id})
    print(f"GradientBoosting — MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")

# Pick best by RMSE
best = min(results, key=lambda x: x["rmse"])

# Save best model
import joblib
if best["name"] == "Ridge":
    best_model = Ridge(alpha=1.0)
else:
    best_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
best_model.fit(X_train, y_train)
joblib.dump(best_model, "models/best_model.pkl")
joblib.dump((X_test, y_test), "models/test_data.pkl")
print(f"\nBest model: {best['name']} with RMSE: {best['rmse']}")

# Save JSON
output = {
    "experiment_name": "gamelag-latency-ms",
    "models": [
        {"name": r["name"], "mae": r["mae"], "rmse": r["rmse"], "r2": r["r2"]}
        for r in results
    ],
    "best_model": best["name"],
    "best_metric_name": "rmse",
    "best_metric_value": best["rmse"]
}

with open("results/step1_s1.json", "w") as f:
    json.dump(output, f, indent=2)

print("results/step1_s1.json saved")