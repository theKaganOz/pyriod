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
periods = list(period_ampl.Periods)[0:10]
phases  = [ "cos" for i in range(len(periods))]
length = len(stock)
harmonics = form_harmonic_dataframe(length, periods, phases)
from statsmodels.regression.linear_model import OLS
from statsmodels.api import add_constant
harmonic_regression = OLS(np.array(stock), add_constant(harmonics)).fit()
harmonic_predictions = harmonic_regression.fittedvalues
harmonic_residuals  = harmonic_regression.resid
print(harmonic_regression.summary())
_ = plt.figure()
_ = plt.plot(np.array(harmonic_predictions))
_ = plt.plot(np.array(stock))
_ = plt.figure()
_ = plt.plot(harmonic_residuals)
_ = plt.figure()
_ = plt.hist(harmonic_residuals)
_ = plot_pacf(harmonic_residuals)
_ = plt.figure()
harmonic_predictions = [harmonic_predict(t, harmonic_regression, periods) for t in range(501, 501 + 3 * 60)]
_ = plt.plot(harmonic_predictions) 