# -*- coding: utf-8 -*-
"""
Voting Spike Detection - Modular Script
Detects anomalous ballot drop patterns using LSTM on synthetic data.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

def generate_synthetic_data(csv_path='synthetic_ballot_drops_per_box.csv'):
    """Generate and save synthetic ballot drop data with injected anomalies."""
    timestamps = pd.date_range('2025-07-01 07:00:00', periods=500, freq='5T')
    np.random.seed(42)
    ballots_dropped = np.ones(500, dtype=int)
    anomaly_indices = np.random.choice(500, 15, replace=False)
    ballots_dropped[anomaly_indices] = np.random.randint(2, 5, size=15)
    df = pd.DataFrame({
        'timestamp': timestamps,
        'ballot_box_id': 'Box_1',
        'ballots_dropped': ballots_dropped
    })
    df.to_csv(csv_path, index=False)
    return df

def load_data(csv_path='synthetic_ballot_drops_per_box.csv'):
    """Load the synthetic ballot drop data."""
    return pd.read_csv(csv_path)

def preprocess_for_lstm(df, window_size=10):
    """Scale ballots_dropped and create LSTM sequences."""
    scaler = MinMaxScaler()
    df['scaled_ballots_dropped'] = scaler.fit_transform(df[['ballots_dropped']])
    data = df['scaled_ballots_dropped'].values
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    X = np.array(X)
    y = np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y, scaler

def build_and_train_lstm(X, y, window_size=10, epochs=20, batch_size=32):
    """Build and train the LSTM model."""
    model = Sequential([
        LSTM(50, input_shape=(window_size, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    history = model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
    return model, history

def detect_anomalies(model, X, y, window_size, df):
    """Detect anomalies based on LSTM prediction errors."""
    predictions = model.predict(X, verbose=0)
    errors = np.abs(predictions.flatten() - y)
    threshold = errors.mean() + 2 * errors.std()
    anomalies = errors > threshold
    df = df.copy()
    df['anomaly'] = False
    df.loc[window_size:, 'anomaly'] = anomalies
    return df, errors, threshold

def plot_anomalies(df):
    """Visualize ballot drops and detected anomalies."""
    plt.figure(figsize=(18,5))
    plt.plot(df['timestamp'], df['ballots_dropped'], label='Ballots Dropped')
    plt.scatter(
        df.loc[df['anomaly'], 'timestamp'],
        df.loc[df['anomaly'], 'ballots_dropped'],
        color='red', label='Anomaly')
    plt.xlabel('Time')
    plt.ylabel('Ballots Dropped')
    plt.title('Ballot Drop Monitoring - Anomaly Detection')
    plt.legend()
    plt.grid(True)
    plt.xticks(ticks=range(0, len(df), 50), rotation=45)
    plt.tight_layout()
    plt.show()

def run_voting_spike_detection():
    # Step 1: Generate synthetic data
    generate_synthetic_data()
    # Step 2: Load data
    df = load_data()
    # Step 3: Preprocess for LSTM
    window_size = 10
    X, y, scaler = preprocess_for_lstm(df, window_size)
    # Step 4: Build and train LSTM
    model, history = build_and_train_lstm(X, y, window_size)
    # Step 5: Detect anomalies
    df_with_anomalies, errors, threshold = detect_anomalies(model, X, y, window_size, df)
    print(f"Anomaly threshold: {threshold:.4f}")
    print(f"Number of detected anomalies: {df_with_anomalies['anomaly'].sum()}")
    # Step 6: Visualize results
    plot_anomalies(df_with_anomalies)

if __name__ == "__main__":
    run_voting_spike_detection()

