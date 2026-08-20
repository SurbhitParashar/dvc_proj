import pandas as pd


input_path = "data/raw/train.csv"
output_path = "data/processed/data.csv"


df = pd.read_csv(input_path)


# Datetime
df['pickup_datetime'] = pd.to_datetime(
    df['pickup_datetime']
)

df['pickup_hour'] = df['pickup_datetime'].dt.hour

df['pickup_dayofweek'] = (
    df['pickup_datetime'].dt.dayofweek
)

df['pickup_month'] = (
    df['pickup_datetime'].dt.month
)

df['is_weekend'] = (
    df['pickup_dayofweek'] >= 5
).astype(int)

df['is_rush_hour'] = (
    df['pickup_hour'].isin(
        [7, 8, 9, 17, 18, 19]
    )
).astype(int)


# Categorical
df['store_and_fwd_flag'] = (
    df['store_and_fwd_flag'] == 'Y'
).astype(int)


# Remove unnecessary columns
df = df.drop(
    [
        'id',
        'pickup_datetime',
        'dropoff_datetime'
    ],
    axis=1
)


df.to_csv(
    output_path,
    index=False
)

print("Processed data saved.")