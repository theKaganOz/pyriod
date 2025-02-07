import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.api import OLS, add_constant

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

def expanding_window_arima(series,
                           horizon,
                          hypothesized_order,
                           series_name
                          ):
    """ A function to carry out expanding-window ARIMA forecasting.
    Inputs: A time series, a horizon, a hypothesized lag order, 
    Outputs: Mean squared forecasting error and an ARIMA model
    """
    test_series = series[(len(series)-horizon+1):]
    print(test_series) # to validate if the set is correctly sliced
    model = ARIMA(series, 
                  order = hypothesized_order, 
                  ).fit()
    print(model.summary())
    print("Forecast for: ", model.forecast())
    print("Expected change compared to last sample: ", model.forecast() - series.iloc[-1])
    forecasts = []
    for i in range(1, horizon):
        data = series[:(-horizon+i)]
        model_expanding = ARIMA(data, 
                                order = hypothesized_order,
                                ).fit()
        forecast = model_expanding.forecast()
        forecasts.append(forecast)
    forecast_error = test_series.to_numpy() - np.array(forecasts).flatten()
    print("Mean squared forecast error: ", mean_squared_error(np.array(forecasts).flatten(), test_series.to_numpy()))
    _ = plt.figure(figsize = (20, 5))
    _ = plt.plot(forecasts, "r-*", label = "Forecasts")
    _ = plt.plot(test_series.to_numpy(), "g-o", label = "Test Set ({0})".format(series_name))
    _ = plt.title("Test vs Forecast Series {0}".format(series_name))
    _ = plt.legend()
    _ = plt.grid()
    return forecasts, model
    

def periodCNN(period_size, 
              kernel_size, 
              train_dataset, 
              test_dataset, 
              num_features=1, 
              epochs=3):
    # Example CNN model
    cnn_model = Sequential()
    cnn_model.add(Conv1D(filters=10, kernel_size=kernel_size, activation="relu", input_shape=(period_size, num_features)))

    cnn_model.add(Flatten())
    cnn_model.add(Dense(16, activation="relu"))
    cnn_model.add(Dense(period_size, activation='linear'))  # Output layer for sequence prediction
    
    # Compile the model
    cnn_model.compile(optimizer='adam', loss="mean_squared_error", metrics=['mean_squared_error'])
    
    # Train the model
    cnn_model.fit(train_dataset, epochs=epochs, validation_data=test_dataset)
    
    # Evaluate the model on the test dataset
    loss, accuracy = cnn_model.evaluate(test_dataset)
    print(cnn_model.summary())
    predictions_cnn = cnn_model.predict(test_dataset)
    print(f'Test loss: {loss}, Test accuracy: {accuracy}')
    
    return cnn_model, predictions_cnn


