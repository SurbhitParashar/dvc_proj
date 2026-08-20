import pandas as pd
import numpy as np
import pickle
import json

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# -----------------------------
# Load data
# -----------------------------

df = pd.read_csv(
    "data/processed/data.csv"
)


# -----------------------------
# Features
# -----------------------------

features = [
    'vendor_id',
    'passenger_count',
    'pickup_longitude',
    'pickup_latitude',
    'dropoff_longitude',
    'dropoff_latitude',
    'distance_km',
    'pickup_hour',
    'pickup_dayofweek',
    'pickup_month',
    'is_weekend',
    'is_rush_hour',
    'store_and_fwd_flag'
]


X = df[features]

y = df['trip_duration']


# -----------------------------
# Train Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# -----------------------------
# Model
# -----------------------------

model = LinearRegression()

model.fit(
    X_train,
    y_train
)


# -----------------------------
# Prediction
# -----------------------------

y_pred = model.predict(X_test)


# -----------------------------
# Metrics
# -----------------------------

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)

r2 = r2_score(
    y_test,
    y_pred
)


print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2)


# -----------------------------
# Save model
# -----------------------------

with open(
    "models/model.pkl",
    "wb"
) as f:

    pickle.dump(
        model,
        f
    )


# -----------------------------
# Save metrics
# -----------------------------

metrics = {
    "MAE": mae,
    "RMSE": rmse,
    "R2": r2
}


with open(
    "metrics.json",
    "w"
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )