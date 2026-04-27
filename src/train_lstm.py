"""
LSTM model for RUL prediction
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def prepare_sequences(data, sequence_length=30):
    """
    Prepare sequences for LSTM input
    X shape: (samples, sequence_length, features)
    y shape: (samples,)
    """
    sequences = []
    targets = []
    
    for unit in data['unit'].unique():
        unit_data = data[data['unit'] == unit].sort_values('cycle')
        
        # Select features (sensors + rolling features)
        feature_cols = [col for col in unit_data.columns 
                       if col.startswith('sensor') or 'rolling' in col]
        
        unit_features = unit_data[feature_cols].values
        unit_rul = unit_data['RUL'].values
        
        # Create sequences
        for i in range(len(unit_data) - sequence_length):
            sequences.append(unit_features[i:i+sequence_length])
            targets.append(unit_rul[i+sequence_length])
    
    return np.array(sequences), np.array(targets)

def create_lstm_model(sequence_length, n_features):
    """
    Create LSTM model architecture
    """
    model = keras.Sequential([
        layers.LSTM(64, return_sequences=True, 
                   input_shape=(sequence_length, n_features)),
        layers.Dropout(0.2),
        layers.LSTM(32, return_sequences=False),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)  # RUL prediction
    ])
    
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    return model

if __name__ == "__main__":
    print("LSTM model module loaded!")
