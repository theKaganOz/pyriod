import numpy as np
import statsmodels.api as sm

class SpectralRegressor:
    def __init__(self, periods):
        """
        Initialize with a list of periods.

        Parameters:
        periods (array-like): Periods for which sin/cos terms will be created.
        """
        self.periods = np.asarray(periods)
        self.frequencies = 1 / self.periods
        self.model = None
        self.results = None

    def _build_design_matrix(self, t):
        """
        Construct the design matrix using sine and cosine terms of each frequency.

        Parameters:
        t (array-like): Time points

        Returns:
        X (ndarray): Design matrix with sin/cos terms
        """
        t = np.asarray(t)
        X = np.column_stack([
            func(2 * np.pi * freq * t)
            for freq in self.frequencies
            for func in (np.sin, np.cos)
        ])
        return X

    def fit(self, t, y):
        """
        Fit the spectral model.

        Parameters:
        t (array-like): Time points
        y (array-like): Observed data
        """
        X = self._build_design_matrix(t)
        X = sm.add_constant(X)
        self.model = sm.OLS(y, X)
        self.results = self.model.fit()

    def predict(self, t):
        """
        Predict values based on the fitted model.

        Parameters:
        t (array-like): Time points

        Returns:
        y_pred (ndarray): Predicted values
        """
        X = self._build_design_matrix(t)
        X = sm.add_constant(X)
        return self.results.predict(X)

    def residuals(self, y, t):
        """
        Compute residuals between actual and predicted values.

        Parameters:
        y (array-like): True values
        t (array-like): Time points

        Returns:
        residuals (ndarray): Residuals
        """
        return y - self.predict(t)

    def summary(self):
        """
        Return the model's statistical summary.

        Returns:
        statsmodels summary object
        """
        return self.results.summary()
