import statsmodels.api as sm
import pandas as pd

X_encoded = model_pipeline.named_steps["preprocessor"].transform(X_train)
X_encoded_test = model_pipeline.named_steps["preprocessor"].transform(X_test)

X_sm = sm.add_constant(X_encoded)
sm_model = sm.OLS(y_train, X_sm).fit()

X_sm_test = sm.add_constant(X_encoded_test, has_constant="add")
prediction_result = sm_model.get_prediction(X_sm_test)
interval_summary = prediction_result.summary_frame(alpha=0.10)

print(interval_summary[["mean", "obs_ci_lower", "obs_ci_upper"]].head())