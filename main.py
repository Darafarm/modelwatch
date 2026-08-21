from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pymongo import MongoClient
import pandas as pd
import numpy as np
import joblib
import statsmodels.api as sm

app = FastAPI()

# --- MongoDB connection (created once, at startup) ---
mongo_client = MongoClient("mongodb://localhost:27017/")
db = mongo_client["modelwatch_db"]
orders_collection = db["synthetic_orders"]

# --- Load both trained models (also once, at startup) ---
model_pipeline = joblib.load("price_model.pkl")
sm_model = joblib.load("price_model_intervals.pkl")


class PredictRequest(BaseModel):
    order_id: int


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictRequest):
    order = orders_collection.find_one({"order_id": request.order_id})
    if order is None:
        raise HTTPException(status_code=404, detail="order not found")

    X_new = pd.DataFrame([{
        "product": order["product"],
        "quantity": order["quantity"],
        "customer_type": order["customer_type"],
        "region": order["region"]
    }])

    log_prediction = model_pipeline.predict(X_new)[0]
    prediction = float(np.exp(log_prediction))

    X_encoded = model_pipeline.named_steps["preprocessor"].transform(X_new)
    X_sm = sm.add_constant(X_encoded, has_constant="add")
    interval = sm_model.get_prediction(X_sm).summary_frame(alpha=0.10)

    lower_bound = float(np.exp(interval["obs_ci_lower"].iloc[0]))
    upper_bound = float(np.exp(interval["obs_ci_upper"].iloc[0]))

    return {
        "order_id": order["order_id"],
        "product": order["product"],
        "quantity": order["quantity"],
        "prediction": round(prediction, 2),
        "lower_bound": round(lower_bound, 2),
        "upper_bound": round(upper_bound, 2)
    }