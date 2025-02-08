# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 21:32:25 2024

@author: kagan
"""

from SpectralAnalzyer import *

def harmonic_predict(t, harmonic_regression, periods):
    intercept = harmonic_regression.params[0]
    harmonic_coefficients = list(harmonic_regression.params[1::])
    sum_ = intercept
    for i in range(len(periods)):
        coeff = harmonic_coefficients[i]
        period = periods[i]
        sum_  += coeff * np.cos((np.pi * t)/period)
    return sum_
def compute_critical_value(confidence_level, series):
    return 1 - (confidence_level/(len(series)/2)) ** (1/((len(series)/2)-1))
def generate_harmonic(length, period, phase):
    if phase == "sin":
        return np.array([np.sin((np.pi * t)/period) for t in range(length)])
    elif phase == "cos":
        return np.array([np.cos((np.pi * t)/period) for t in range(length)])
def form_harmonic_dataframe(length, periods, phases):
    harmonic_dataframe = {}
    for i in range(len(periods)):
        period = periods[i]
        phase  = phases[i]
        title = "p" + str(period) + phase
        
        harmonic = generate_harmonic(length, period, phase)
        harmonic_dataframe[title] = harmonic
    return pd.DataFrame.from_dict(harmonic_dataframe)
import numpy as np
import pandas as pd
import statsmodels.api as sm

def harmonic_regression_analysis(periods, t_max, target_series, p_threshold=0.05):
    """
    Constructs a dataframe with cosine functions of given periods,
    performs OLS regression, and selects statistically significant components.
    
    Parameters:
    - periods: List of periods for the cosine functions
    - t_max: Maximum time value (i.e., the length of the series)
    - target_series: The dependent variable time series
    - p_threshold: The significance level for selecting components

    Returns:
    - significant_model: The regression model using only significant components
    - significant_df: DataFrame with only significant cosine components
    """
    # Time index
    t = np.arange(t_max)

    # Construct DataFrame with harmonic components
    harmonic_data = {f'cos_p{p}': np.cos(2 * np.pi * t / p) for p in periods}
    df = pd.DataFrame(harmonic_data)
    
    # Add the target time series
    df['target'] = target_series

    # Perform OLS regression
    X = sm.add_constant(df.drop(columns=['target']))  # Add intercept
    y = df['target']
    model = sm.OLS(y, X).fit()
    
    # Select significant components based on p-values
    significant_columns = [col for col, p_val in model.pvalues.items() if p_val < p_threshold and col != 'const']
    
    if not significant_columns:
        print("No significant components found.")
        return None, None

    # Perform regression again with only significant components
    significant_X = sm.add_constant(df[significant_columns])
    significant_model = sm.OLS(y, significant_X).fit()

    return significant_model, df[significant_columns + ['target']]


