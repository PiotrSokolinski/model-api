import pandas as pd
from ast import literal_eval
import numpy as np
from scipy import stats
from sklearn.preprocessing import RobustScaler
import tensorflow as tf

TIME_STEPS = 24
STEP = 2


def create_dataset(X, y, time_steps=1, step=1):
    Xs, ys = [], []
    for i in range(0, len(X) - time_steps, step):
        v = X.iloc[i:(i + time_steps)].values
        labels = y.iloc[i: i + time_steps]
        Xs.append(v)
        ys.append(stats.mode(labels)[0][0])
    return np.array(Xs), np.array(ys).reshape(-1, 1)


def train_model():
    url = "https://model-api.s3.eu-central-1.amazonaws.com/smartphone.csv"

    report_path = "./report.csv"
    report = pd.read_csv(report_path)
    smartphone = pd.read_csv(url)

    report_eat = report[report["activity_type"] == "Eat"]
    eat_ranges = list(zip(report_eat["from"], report_eat["to"]))

    smartphone["values"] = smartphone["values"].apply(literal_eval)
    smartphone["timestamp"] = pd.to_datetime(smartphone["timestamp"])
    smartphone.set_index("timestamp", inplace=True, drop=True)
    smartphone.drop(columns=["index"], inplace=True)

    accelerometer = smartphone[smartphone["source"] == "accelerometer"].copy()

    accelerometer["eat"] = 0
    for eat_range in eat_ranges:
        accelerometer.loc[eat_range[0]:eat_range[1], ("eat")] = 1

    accelerometer[['accel_x', 'accel_y', 'accel_z']] = pd.DataFrame(accelerometer["values"].tolist(), index=accelerometer.index)
    accelerometer.drop(columns=["values"], inplace=True)
    accelerometer[["accel_x", "accel_y", "accel_z"]] = accelerometer[["accel_x", "accel_y", "accel_z"]].astype(float)

    accelerometer_resampled = accelerometer[["accel_x", "accel_y", "accel_z", "eat"]].resample("5S").mean().fillna(0)
    accelerometer_resampled["eat"] = accelerometer_resampled["eat"].astype(int)

    gyroscope = smartphone[smartphone["source"] == "gyroscope"].copy()

    gyroscope["eat"] = 0
    for eat_range in eat_ranges:
        gyroscope.loc[eat_range[0]:eat_range[1], ("eat")] = 1

    gyroscope[['gyro_x', 'gyro_y', 'gyro_z']] = pd.DataFrame(gyroscope["values"].tolist(), index=gyroscope.index)
    gyroscope.drop(columns=["values"], inplace=True)
    gyroscope[["gyro_x", "gyro_y", "gyro_z"]] = gyroscope[["gyro_x", "gyro_y", "gyro_z"]].astype(float)
    gyroscope_resampled = gyroscope[["gyro_x", "gyro_y", "gyro_z", "eat"]].resample("5S").mean().fillna(0)
    gyroscope_resampled["eat"] = gyroscope_resampled["eat"].astype(int)

    s1 = accelerometer_resampled
    s2 = gyroscope_resampled
    fused = s1.join(s2, lsuffix="_l").drop(columns=["eat_l"])

    df = fused.copy()
    n = len(df)

    train_df = df[0:int(n * 0.7)]
    test_df = df[int(n * 0.7):]

    scale_columns = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']

    scaler = RobustScaler()

    scaler = scaler.fit(train_df[scale_columns])

    train_df.loc[:, scale_columns] = scaler.transform(train_df[scale_columns].to_numpy())
    test_df.loc[:, scale_columns] = scaler.transform(test_df[scale_columns].to_numpy())

    X_train, y_train = create_dataset(
        train_df.drop(columns=['eat']),
        train_df['eat'],
        TIME_STEPS,
        STEP
    )
    print(X_train[:200])
    X_test, y_test = create_dataset(
        test_df.drop(columns=['eat']),
        test_df['eat'],
        TIME_STEPS,
        STEP
    )

    model = tf.keras.Sequential()

    model.add(
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                units=128,
                input_shape=[X_train.shape[1], X_train.shape[2]]
            ),
        )
    )
    model.add(tf.keras.layers.Dropout(rate=0.5))
    model.add(tf.keras.layers.Dense(units=128, activation='relu'))
    model.add(tf.keras.layers.Dense(y_train.shape[1], activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        metrics=['acc'])

    model.fit(
        X_train, y_train,
        epochs=3,
        batch_size=16,
        validation_split=0.1,
        shuffle=False)

    model.evaluate(X_test, y_test)

    return model
