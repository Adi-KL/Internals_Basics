import argparse
import joblib
import numpy as np
import os

model = joblib.load("models/best_model.pkl")

parser = argparse.ArgumentParser()
parser.add_argument("--server_region", type=float, required=True)
parser.add_argument("--concurrent_players", type=float, required=True)
parser.add_argument("--packet_size_kb", type=float, required=True)
parser.add_argument("--is_ranked_match", type=float, required=True)
args = parser.parse_args()

features = np.array([[args.server_region, args.concurrent_players, 
                       args.packet_size_kb, args.is_ranked_match]])
prediction = model.predict(features)[0]
print(f"Predicted latency_ms: {round(prediction, 4)}")