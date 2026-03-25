import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error
import time
from sklearn.dummy import DummyRegressor

# Load the dataset
dt = pd.read_csv('flights.csv',sep=';')

num_dt = dt.select_dtypes(include=["number"])


selected_columns = [
    "CRS_DEP_TIME",
    "CRS_ARR_TIME",
    "DISTANCE",
    "CRS_ELAPSED_TIME",
    "ORIGIN",
    "DEST",
    "OP_CARRIER",
    "ARR_DELAY"
]

df = dt[selected_columns].copy()

df = df.dropna(subset=["ARR_DELAY"]).copy()
df["CRS_ELAPSED_TIME"] = df["CRS_ELAPSED_TIME"].fillna(df["CRS_ELAPSED_TIME"].median())

df = pd.get_dummies(df, columns=["ORIGIN","DEST","OP_CARRIER"],dtype=int)

x = df.drop("ARR_DELAY",axis=1)
y = df["ARR_DELAY"]

# First training with all features to get feature importances
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)
model_rf = RandomForestRegressor(n_estimators=20, max_depth=10, random_state=42, n_jobs=-1)
model_rf.fit(X_train, y_train)


# Get feature importances from the trained model
importances = model_rf.feature_importances_
feature_names = x.columns

feat_imp = pd.Series(importances, index=feature_names)
feat_imp = feat_imp.sort_values(ascending=False)
top_features = feat_imp.head(15).index

X_reduced = x[top_features]

X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

model_rf_final = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
model_lr = LinearRegression()
dummy = DummyRegressor(strategy="mean")
dummy.fit(X_train, y_train)



start_time = time.time()
model_rf_final.fit(X_train, y_train)
model_lr.fit(X_train, y_train)
end_time = time.time()



y_pred_rf = model_rf_final.predict(X_test)
y_pred_lr = model_lr.predict(X_test)
y_pred_dummy = dummy.predict(X_test)
mae_rf = mean_absolute_error(y_test,y_pred_rf)
mse_rf = mean_squared_error(y_test,y_pred_rf)
mae_lr = mean_absolute_error(y_test,y_pred_lr)
mse_lr = mean_squared_error(y_test,y_pred_lr)
mae_dummy = mean_absolute_error(y_test,y_pred_dummy)
mse_dummy = mean_squared_error(y_test,y_pred_dummy)


rf_results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted_RF": y_pred_rf
})

rf_results["Error_RF"] = (rf_results["Actual"] - rf_results["Predicted_RF"]).abs()

rf_results["Predicted_LR"] = y_pred_lr
rf_results["Error_LR"] = (rf_results["Actual"] - rf_results["Predicted_LR"]).abs()

rf_results["Predicted_Dummy"] = y_pred_dummy
rf_results["Error_Dummy"] = (rf_results["Actual"] - rf_results["Predicted_Dummy"]).abs()

rf_results = rf_results.round(2)

print("\n=== Prediction Comparison Table ===")
print(rf_results.head(20))



metrics_df = pd.DataFrame({
    "Model": ["Random Forest", "Linear Regression", "Dummy Regressor"],
    "MAE": [mae_rf, mae_lr, mae_dummy],
    "MSE": [mse_rf, mse_lr, mse_dummy],
    "Training Time (s)": [
        end_time - start_time,
        end_time - start_time,
        0 
    ]
})

metrics_df = metrics_df.round(2)

print("\n=== Model Comparison Metrics ===")
print(metrics_df)