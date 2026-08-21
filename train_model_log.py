import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import statsmodels.api as sm
import joblib

# 1. Load the synthetic dataset
df = pd.read_json("synthetic_orders.json")

X = df[["product", "quantity", "customer_type", "region"]]
y = df["price"]

# 2. Log-transform the target BEFORE splitting
y_log = np.log(y)

# 3. Split into training and test sets
X_train, X_test, y_train_log, y_test_log = train_test_split(X, y_log, test_size=0.2, random_state=42)

# 4. Define preprocessing
categorical_features = ["product", "customer_type", "region"]
numeric_features = ["quantity"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ("num", "passthrough", numeric_features)
    ]
)

# 5. Combine preprocessing + model into a single pipeline
model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# 6. Train on the LOG of price
model_pipeline.fit(X_train, y_train_log)
print("Training complete")

# 7. Evaluate on real-dollar scale
y_pred_log = model_pipeline.predict(X_test)
y_pred = np.exp(y_pred_log)
y_test_real = np.exp(y_test_log)

mae = mean_absolute_error(y_test_real, y_pred)
r2 = r2_score(y_test_real, y_pred)
print("Mean Absolute Error: $", round(mae, 2))
print("R² score:", round(r2, 4))

# 8. Compute a genuine 90% prediction interval, on the log scale, then convert back
X_encoded_train = model_pipeline.named_steps["preprocessor"].transform(X_train)
X_encoded_test = model_pipeline.named_steps["preprocessor"].transform(X_test)

X_sm_train = sm.add_constant(X_encoded_train)
sm_model = sm.OLS(y_train_log, X_sm_train).fit()

X_sm_test = sm.add_constant(X_encoded_test, has_constant="add")
prediction_result = sm_model.get_prediction(X_sm_test)
interval_summary_log = prediction_result.summary_frame(alpha=0.10)

interval_summary_real = pd.DataFrame({
    "prediction": np.exp(interval_summary_log["mean"]),
    "lower_bound": np.exp(interval_summary_log["obs_ci_lower"]),
    "upper_bound": np.exp(interval_summary_log["obs_ci_upper"])
})

print("\nSample prediction intervals (first 5 test rows, real dollars):")
print(interval_summary_real.head())
print("\nMinimum lower_bound across all test rows:", interval_summary_real["lower_bound"].min())

# 9. Save both models to disk
joblib.dump(model_pipeline, "price_model.pkl")
joblib.dump(sm_model, "price_model_intervals.pkl")
print("\nModels saved to price_model.pkl and price_model_intervals.pkl")