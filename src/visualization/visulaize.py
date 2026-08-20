import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# Load data

df = pd.read_csv(
    "data/processed/data.csv"
)


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


# Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Train

model = LinearRegression()

model.fit(
    X_train,
    y_train
)


# Predict

y_pred = model.predict(X_test)


# Visualization

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.3
)

plt.xlabel("Actual Trip Duration")

plt.ylabel("Predicted Trip Duration")

plt.title(
    "Actual vs Predicted Trip Duration"
)

plt.savefig(
    "reports/figures/actual_vs_predicted.png"
)

plt.close()

print("Visualization saved.")