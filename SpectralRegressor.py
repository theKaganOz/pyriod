# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 21:32:25 2024

@author: kagan
"""

from SpectralAnalyzer import *

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

import numpy as np
import pandas as pd
import statsmodels.api as sm

def generate_harmonic_dataframe(periods, t_max):
    """
    Generates a DataFrame containing cosine functions of given periods.

    Parameters:
    - periods: List of periods for the cosine functions
    - t_max: Maximum time value (i.e., the length of the series)

    Returns:
    - harmonic_df: DataFrame containing cosine components
    """
    t = np.arange(t_max)
    harmonic_data = {f'cos_p{p}': np.cos(2 * np.pi * t / p) for p in periods}
    return pd.DataFrame(harmonic_data)

def add_target_series(harmonic_df, target_series):
    """
    Adds the target time series to the harmonic dataframe.

    Parameters:
    - harmonic_df: DataFrame containing harmonic components
    - target_series: The dependent variable time series

    Returns:
    - ols_df: DataFrame including the target series
    """
    ols_df = harmonic_df.copy()
    ols_df['target'] = target_series
    return ols_df

def perform_ols_regression(ols_df):
    """
    Performs an initial OLS regression on the given DataFrame.

    Parameters:
    - ols_df: DataFrame containing independent variables and the target series

    Returns:
    - initial_model: Fitted OLS model
    """
    X = sm.add_constant(ols_df.drop(columns=['target']))  # Add intercept
    y = ols_df['target']
    return sm.OLS(y, X).fit()

def filter_significant_components(model, p_threshold):
    """
    Filters significant components based on p-values from an OLS model.

    Parameters:
    - model: Fitted OLS model
    - p_threshold: The significance level for selecting components

    Returns:
    - significant_columns: List of significant component names
    """
    return [col for col, p_val in model.pvalues.items() if p_val < p_threshold and col != 'const']

def perform_final_regression(ols_df, significant_columns):
    """
    Performs OLS regression using only the significant components.

    Parameters:
    - ols_df: DataFrame containing independent variables and target series
    - significant_columns: List of significant component names

    Returns:
    - final_model: Fitted OLS model with significant components
    - significant_df: DataFrame with selected components and target series
    """
    significant_X = sm.add_constant(ols_df[significant_columns])
    y = ols_df['target']
    return sm.OLS(y, significant_X).fit(), ols_df[significant_columns + ['target']]

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
    harmonic_df = generate_harmonic_dataframe(periods, t_max)
    print(harmonic_df)

    ols_df = add_target_series(harmonic_df, target_series)
    print(ols_df)

    initial_model = perform_ols_regression(ols_df)
    significant_columns = filter_significant_components(initial_model, p_threshold)

    if not significant_columns:
        print("No significant components found. Returning primitive model.")
        return initial_model, None

    significant_model, significant_df = perform_final_regression(ols_df, significant_columns)

    return significant_model, significant_df



