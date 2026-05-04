import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import json
import os
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error

mlflow.set_tracking_uri("file:./mlruns")

df = pd.read_csv("data/training_data.csv")
X = df.drop("latency_ms", axis=1)
y = df["latency_ms"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

param_grid = {
    "n_estimators": [50, 150],
    "learning_rate": [0.05, 0.1, 0.2],
    "max_depth": [3, 5, 10]
}

mlflow.set_experiment("gamelag-latency-ms")

with mlflow.start_run(run_name="tuning-gamelag") as parent_run:
    search = RandomizedSearchCV(
        GradientBoostingRegressor(random_state=42),
        param_distributions=param_grid,
        n_iter=18,
        cv=3,
        scoring="neg_mean_absolute_error",
        random_state=42,
        n_jobs=-1
    )
    search.fit(X_train, y_train)

    # Log each trial as nested run
    for i in range(len(search.cv_results_["params"])):
        with mlflow.start_run(nested=True, run_name=f"trial_{i}"):
            mlflow.log_params(search.cv_results_["params"][i])
            mlflow.log_metric("cv_mae", -search.cv_results_["mean_test_score"][i])

    # Best model metrics on test set
    best_model = search.best_estimator_
    preds = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    best_cv_mae = -search.best_score_

    mlflow.log_params(search.best_params_)
    mlflow.log_metric("best_mae", mae)
    mlflow.log_metric("best_cv_mae", best_cv_mae)
    mlflow.sklearn.log_model(best_model, "model")

    joblib.dump(best_model, "models/best_model.pkl")

output = {
    "search_type": "random",
    "n_folds": 3,
    "total_trials": len(search.cv_results_["params"]),
    "best_params": search.best_params_,
    "best_mae": round(mae, 4),
    "best_cv_mae": round(best_cv_mae, 4),
    "parent_run_name": "tuning-gamelag"
}

with open("results/step2_s2.json", "w") as f:
    json.dump(output, f, indent=2)

print("Best params:", search.best_params_)
print(f"Best MAE: {mae:.4f}, Best CV MAE: {best_cv_mae:.4f}")
print("results/step2_s2.json saved")