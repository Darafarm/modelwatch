from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pymongo import MongoClient
import pandas as pd
import numpy as np
import joblib
import statsmodels.api as sm
import os
import logging
import json
import redis
from prometheus_client import Counter, Histogram, make_asgi_app

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

REQUEST_COUNT = Counter(
    "predict_requests_total",
    "Total number of prediction requests received"
)

INFERENCE_LATENCY = Histogram(
    "predict_inference_duration_seconds",
    "Time spent running the model inference"
)

FEATURE_RETRIEVAL_LATENCY = Histogram(
    "predict_feature_retrieval_duration_seconds",
    "Time spent retrieving order features from MongoDB"
)

FAILURE_COUNT = Counter(
    "predict_failures_total",
    "Total number of failed prediction requests",
    ["failure_type"]
)

# --- MongoDB connection (created once, at startup) ---
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
mongo_client = MongoClient(MONGO_URI)
db = mongo_client["modelwatch_db"]
orders_collection = db["synthetic_orders"]

# --- Redis connection (created once, at startup) ---
REDIS_URI = os.environ.get("REDIS_URI", "redis://localhost:6379/")
redis_client = redis.from_url(REDIS_URI)
CACHE_TTL_SECONDS = 60

# --- Load both trained models (also once, at startup) ---
model_pipeline = joblib.load("price_model.pkl")
sm_model = joblib.load("price_model_intervals.pkl")


class PredictRequest(BaseModel):
    order_id: int


@app.get("/health")
def health():
    return {"status": "ok"}


metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


@app.post("/predict")
def predict(request: PredictRequest):
    REQUEST_COUNT.inc()

    try:
        with FEATURE_RETRIEVAL_LATENCY.time():
            cache_key = f"order:{request.order_id}"
            cached = redis_client.get(cache_key)

            if cached is not None:
                order = json.loads(cached)
            else:
                order = orders_collection.find_one({"order_id": request.order_id})
                if order is not None:
                    order.pop("_id", None)
                    redis_client.setex(cache_key, CACHE_TTL_SECONDS, json.dumps(order))

        if order is None:
            FAILURE_COUNT.labels(failure_type="not_found").inc()
            raise HTTPException(status_code=404, detail="order not found")

        X_new = pd.DataFrame([{
            "product": order["product"],
            "quantity": order["quantity"],
            "customer_type": order["customer_type"],
            "region": order["region"]
        }])

        with INFERENCE_LATENCY.time():
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

    except HTTPException:
        raise
    except Exception:
        logger.exception("Unexpected error in /predict")
        FAILURE_COUNT.labels(failure_type="internal_error").inc()
        raise HTTPException(status_code=500, detail="internal server error")