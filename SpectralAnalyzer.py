# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 03:01:33 2024

@author: kagan

This module contains spectral time series analysis tools, namely:
    - A unit root test based on spectral properties, as proposed by Dickey & Akdi (1995)
    - A generalization of Fisher's test for hidden periodicities (Akdi, 2012)
    - A periodogram-based method of computing the spectral distance between two times series
    
"""

import pandas as pd
import numpy  as np

from statsmodels.tsa.stattools import adfuller, pacf
from statsmodels.tsa.arima.model import ARIMA
from scipy.signal import periodogram

class SpectralAnalyzer:
    """
    This class is initialized with a time series, and upon initialization,
    it automatically computes the stationarity of a time series.
    If "auto_diff" is enabled, the series is differenced until it no longer
    contains a unit root. 
    If any difference of the series does not contain a unit root,
    hidden periods will automatically be identified.
    A method named subdivide will divide the time series into equal periods of 
    given length. 
    """

    def __init__(self, time_series, 
                 significance_level,
                 n_hidden_components):
        self.__n_hidden_components = n_hidden_components
        self.__significance_level = significance_level
        # The signal itself
        self.__signal = time_series
        # Residual variance estimate
        self.__residual_variance_estimate = self.form_model_and_forecast(time_series,
                                                                         pacf_threshold = 0.1)[0].params["sigma2"]
        # Linearly tested unit root 
        self.__linear_unit_root   =  adfuller(time_series)[1] 
        # Periodically tested unit root
        self.__periodic_unit_root = self.__periodogram_unit_root_test()[0]
        # Form period-amplitude map
        self.__hidden_periodic_components = self.__form_period_amplitude_map()
    def get_hidden_periodic_components(self):
        return self.__hidden_periodic_components
    def get_periodogram_unit_root_results(self):
        return self.__periodic_unit_root
    def form_model_and_forecast(self, series, pacf_threshold):
    # Perform stationarity test
        stationarity_test = adfuller(series)[1] < 0.05 
        print("Stationarity: ", stationarity_test)
        integration_order = 0
        # If the series is stationary, get partial autocorrelation function and model using lags that are above the threshold
        partial_autocorrelations = []
        # Integration order is zero
        if stationarity_test == True:
            print("Stationarity at order ", integration_order)
            partial_autocorrelations = list(np.argwhere(pacf(series) > pacf_threshold))[1:]
        else:
            differenced_series = series.diff().dropna()
            print("Series is differenced")
            stationarity_test = adfuller(differenced_series)[1] < 0.05
            print("Differenced stationarity test: ", stationarity_test)
            if stationarity_test:
                integration_order = 1
                partial_autocorrelations = list(np.argwhere(pacf(differenced_series) > pacf_threshold))[1:]
                
        print("Stationarity at order", integration_order)
        model = ARIMA(series, order = (partial_autocorrelations, integration_order, 0)).fit()
        print("Model mean sum of squared errors: ", model.mse)
        print("Model AICC: ", model.aicc)
        print("Model mean forecast: ", model.get_forecast().predicted_mean)
        print("Model forecast confidence interval: ", model.get_forecast().conf_int(alpha=0.05))
        return model, model.aicc, model.mse, model.get_forecast()

    
    def __periodogram_unit_root_test(self):
    
        significance_level = str(self.__significance_level)
        critical_values = {"0.01" : 0.0348,
                           "0.05": 0.178,
                           "0.1" : 0.368}
    
        frequencies, amplitudes = periodogram(self.__signal)
        omega = np.degrees(frequencies[1])
        # I(w1) 
        per = amplitudes[1]
        test_statistic = (2 * (1 - np.cos(2 * np.pi/len(self.__signal) )) / self.__residual_variance_estimate) * per
        
        
        print("Test stat for {0} is {1}".format(self.__signal.name, test_statistic))
        stationary = test_statistic < critical_values[significance_level]
        if stationary:
            print("The series is stationary according to Periodogram Unit Root Test")
        else:
            print("The series is not stationary according to Periodogram Unit Root Test")
        return stationary, test_statistic


    def __form_period_amplitude_map(self):
        def important_frequencies(signal, n, extract_important = True):
            def get_v_statistic_at_order_n(amplitude_dataframe, n):
                total = amplitude_dataframe.sum()
                print("Total: {0}".format(total))
                sum_at_n = amplitude_dataframe[:n].sum()
                print("Total up to n={0}: {1}".format(n, sum_at_n))
                amplitude_at_n = amplitude_dataframe[n]
                print("Amplitude at n={0}: {1}".format(n, amplitude_at_n))
                return amplitude_at_n / (total - sum_at_n)
            def compute_critical_value(confidence_level, series):
                return 1 - (confidence_level/len(series)) ** (1/(len(series)-1))
            # get highest n frequencies
            # get v values
            v_statistics    = [get_v_statistic_at_order_n(period_ampl["Amplitudes"], i) for i in range(n)]
            critical_value = compute_critical_value(0.05, signal)
            results_table   = period_ampl.copy()[0:n]
            results_table["V"] = v_statistics
            print("Critical value: ", critical_value)
            print(results_table)
            # Filter important frequencies
            important_components = results_table[results_table["V"] >= critical_value]
            if extract_important:
                return important_components
            else:
                return results_table
               
        frequencies, amplitudes = periodogram(self.__signal)
        # to overcome the initial zero frequency.
        frequencies = frequencies[1:]
        amplitudes  = amplitudes[1:]
        print("Computed periodogram.")
        # convert frequencies to periods
        print("Converting frequencies to periods.")
        periods = np.reciprocal(frequencies) 
        
        # form a dictionary to hold frequency amplitude pairs
        period_ampl = pd.DataFrame({"Periods": periods, "Amplitudes":amplitudes})
        # sort these values by amplitudes
        period_ampl = period_ampl.sort_values(by = "Amplitudes", ascending = False)
        # get the sum of all amplitudes
        amplitude_sum = period_ampl["Amplitudes"].sum()
        important_components = important_frequencies(self.__signal, self.__n_hidden_components)

        return important_components
        

        
        





