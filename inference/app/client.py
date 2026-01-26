from datetime import datetime

import pandas as pd
import requests
import json

# Base URL of your FastAPI service
BASE_URL = "http://localhost:8000"

# Example 1: Health check
response = requests.get(f"{BASE_URL}/health")
print("Health Check:", response.json())

# Example 2: Get model info
response = requests.get(f"{BASE_URL}/model/info")
print("\nModel Info:", response.json())


df = pd.read_csv("lgb_feature_data.csv")

df_pred = df[["Asset_ID"]].copy()
# Drop ID column
df = df.drop(columns=["Asset_ID"])


t1=datetime.now()
payload = {
    "data": df.to_dict(orient="records")  # ✅ list of rows
}

response = requests.post(
    f"{BASE_URL}/predict/batch",
    json=payload
)
t2=datetime.now()
print(t2-t1)


result = response.json()

# ---------------------------
# Extract predictions
# ---------------------------
preds = result["predictions"]

assert len(preds) == len(df_pred)

df_pred["prediction"] = preds

print(df_pred.head())

