"""
Data preprocessing and feature engineering for NASA C-MAPSS
"""
import numpy as np
import pandas as pd

def add_remaining_useful_life(df):
    """
    Calculate RUL (Remaining Useful Life) for each engine
    RUL = max_cycle - current_cycle
    """
    # Get max cycle for each engine (failure point)
    max_cycles = df.groupby('unit')['cycle'].max().reset_index()
    max_cycles.columns = ['unit', 'max_cycle']
    
    # Merge and calculate RUL
    df = df.merge(max_cycles, on='unit', how='left')
    df['RUL'] = df['max_cycle'] - df['cycle']
    df.drop('max_cycle', axis=1, inplace=True)
    
    return df

def add_rolling_features(df, sensors, window=5):
    """
    Add rolling mean and std for sensor readings
    """
    df = df.sort_values(['unit', 'cycle'])
    
    for sensor in sensors:
        # Rolling mean
        df[f'{sensor}_rolling_mean'] = df.groupby('unit')[sensor].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        # Rolling std
        df[f'{sensor}_rolling_std'] = df.groupby('unit')[sensor].transform(
            lambda x: x.rolling(window=window, min_periods=1).std()
        ).fillna(0)
    
    return df

def normalize_data(train_df, test_df, columns):
    """
    Normalize features using training set statistics
    """
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    train_df[columns] = scaler.fit_transform(train_df[columns])
    test_df[columns] = scaler.transform(test_df[columns])
    
    return train_df, test_df, scaler

if __name__ == "__main__":
    print("Preprocessing module loaded successfully!")
