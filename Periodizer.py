# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 21:50:18 2024

@author: kagan
"""
import numpy      as np
import tensorflow as tf
import pandas as pd

def consecutive_periods_tf(series, period_size):
    data = pd.Series(series)
    X, y = [], []

    for i in range(0, len(data) - 2 * period_size + 1, period_size):
        period1 = data.iloc[i:i + period_size].tolist()
        period2 = data.iloc[i + period_size:i + 2 * period_size].tolist()
        X.append(period1)
        y.append(period2)

    # Convert lists to TensorFlow tensors
    X_tensor = tf.constant(X, dtype=tf.float32)
    y_tensor = tf.constant(y, dtype=tf.float32)
    
    batch_size = 1
    dataset = tf.data.Dataset.from_tensor_slices((X_tensor, y_tensor)).batch(batch_size)

    return dataset, X_tensor, y_tensor
def periodize_signal(signal,
                     period_size):
    dataset_length = len(signal)
    number_of_periods = dataset_length // period_size
    
    upper, lower = 0, period_size
    boundaries = [(upper, lower)]
    for i in range(number_of_periods):
        upper += period_size
        lower += period_size
        boundaries.append((upper, lower))
    
    X = []
    y = []
    for i in range(1, len(boundaries) - 1):
        l, u   = boundaries[i][0]    , boundaries[i][1]
        ln, un = boundaries[i + 1][0], boundaries[i + 1][1]
        sub_sequence_current = signal[l:u]
        sub_sequence_next    = signal[ln:un]
        X.append(sub_sequence_current)
        y.append(sub_sequence_next)
    
    print("X ", X)
    print("y", y)
    # Parameters
    input_length = period_size  # Length of the input sequence
    num_features = 1  # Number of features in each time step of the input sequence (univariate time series)
    
    # Prepare for output
    X = np.array(X)
    y = np.array(y)
    
    X = X.reshape(-1, period_size, num_features)
    y = y.reshape(-1, period_size, num_features)
    # Prepare for neural nets
    batch_size = 1
    dataset = tf.data.Dataset.from_tensor_slices((X, y)).batch(batch_size)
    
    return dataset, X, y

    