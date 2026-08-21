import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import statsmodels.api as sm
import joblib

# 1. Load the synthetic dataset (real MongoGuard data had no genuine price signal)
df = pd.read_json("synthetic_orders.json")

X = df[["product", "quantity", "customer_type", "region"]]
y = df["price"]

# 2. Split into training and test sets, so evaluation is honest
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Define preprocessing: one-hot encode categoricals, pass quantity through unchanged
categorical_features = ["product", "customer_type", "region"]
numeric_features = ["quantity"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ("num", "passthrough", numeric_features)
    ]
)
# 4. Combine preprocessing + model into a single pipeline
model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# 5. Train
model_pipeline.fit(X_train, y_train)
print("Training complete")

# 6. Evaluate honestly, on data the model never saw during training
y_pred = model_pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Absolute Error: $", round(mae, 2))
print("R² score:", round(r2, 4))

# 7. Compute a genuine 90% prediction interval using statsmodels
X_encoded_train = model_pipeline.named_steps["preprocessor"].transform(X_train)
X_encoded_test = model_pipeline.named_steps["preprocessor"].transform(X_test)

X_sm_train = sm.add_constant(X_encoded_train)
sm_model = sm.OLS(y_train, X_sm_train).fit()

X_sm_test = sm.add_constant(X_encoded_test, has_constant="add")
prediction_result = sm_model.get_prediction(X_sm_test)
interval_summary = prediction_result.summary_frame(alpha=0.10)

print("\nSample prediction intervals (first 5 test rows):")
print(interval_summary[["mean", "obs_ci_lower", "obs_ci_upper"]].head())

# 8. Save both the scikit-learn pipeline and the statsmodels model to disk
joblib.dump(model_pipeline, "price_model.pkl")
joblib.dump(sm_model, "price_model_intervals.pkl")
print("\nModels saved to price_model.pkl and price_model_intervals.pkl")