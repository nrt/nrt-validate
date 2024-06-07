import numpy as np


class PartitionedHarmonicTrendModel:
    """Least square harmonic fitting of partitioned time-series

    Args:
        dates (np.ndarray): Array of datetime64 dates
        delta_predict (np.timedelta64): Temporal interval for smooth predictions

    Examples:
        >>> import numpy as np
        >>> from nrt.validate.fitting import PartitionedHarmonicTrendModel

        >>> np.random.seed(42)

        >>> # Random time-series with irregularly spaced observations
        >>> start_date = np.datetime64('2020-01-01')
        >>> n_samples = 120
        >>> days_between = np.random.randint(5, 21, size=n_samples)
        >>> dates = np.cumsum(days_between)
        >>> dates = start_date + np.array(dates, dtype='timedelta64[D]')
        >>> y = np.random.random(size=n_samples)
        >>> breakpoints = [dates[0], dates[40], dates[-1]]

        >>> model = PartitionedHarmonicTrendModel(dates)
        >>> dates, predictions = model.fit_predict(y, breakpoints, 2)
        >>> print(dates)
        >>> print(predictions)
    """
    def __init__(self, dates,
                 delta_predict=np.timedelta64(5, 'D')):
        self.dates = dates
        self.delta_predict = delta_predict
        self.predict_dates = np.arange(dates.min(),
                                       dates.max(),
                                       dtype='datetime64[D]')
        self.X = self._regressors(dates)
        self.X_full = self._regressors(self.predict_dates)


    def fit_predict(self, y, breakpoints, order):
        """Fit a harmonic trend model for each segment and returns a smoothed prediction

        Note that the last value of the time-series is not used for the fitting.
        A segment starts at a breakpoint and ends one observation before the next
        breakpoint

        Args:
            y (np.ndarray): Array of values
            breakpoints (list): List of datetime64 dates corresponding to breakpoints
                dates. Usually include extremities of the time-series too.
            order (int): Harmonic order, between 0 and 5

        Returns:
            - Dates: List of datetime64 arrays each corresponding to a segment
            - predicted values: List of matching arrays containing predicted values
        """
        dates = []
        predictions = []
        for i in range(len(breakpoints) - 1):
            begin = breakpoints[i]
            end = breakpoints[i + 1]
            # Subset X and y
            mask = np.logical_and(self.dates >= begin, self.dates < end)
            y_sub = y[mask]
            X_sub = self.X[mask][:,range(2 + 2*order)]
            # Second subset for nan
            isna = np.isnan(y_sub)
            X_sub = X_sub[~isna]
            y_sub = y_sub[~isna]
            beta = np.linalg.solve(np.dot(X_sub.T, X_sub),
                                   np.dot(X_sub.T, y_sub))
            # Predict smoothed values
            mask = np.logical_and(self.predict_dates >= begin,
                                  self.predict_dates < end)
            predict_dates_sub = self.predict_dates[mask]
            X_full_sub = self.X_full[mask][:,range(2 + 2*order)]
            y_pred = np.dot(X_full_sub, beta)
            dates.append(predict_dates_sub)
            predictions.append(y_pred)
        return dates, predictions

    @staticmethod
    def _regressors(dates):
        """Prepare a design matrix for the provided dates with intercept, trend and 5 harmonic components
        """
        shape = (dates.size, 12) # Max harmonic order of 5
        ddates = PartitionedHarmonicTrendModel.decimal_dates(dates)
        X = np.zeros(shape, dtype=np.float64)
        X[:,0] = 1 # Intercept
        X[:,1] = dates.astype('datetime64[D]').astype(int) # Trend
        # Compute harmonic components
        X_harmon = np.empty((len(dates), 5))
        for i in range(5):
            X_harmon[:,i] = 2 * np.pi * ddates * (i + 1)
        X_harmon = np.concatenate([np.cos(X_harmon), np.sin(X_harmon)], 1)
        X[:, range(2,12)] = X_harmon
        return X

    @staticmethod
    def decimal_dates(dates):
        """Convert a datetime64 array to decimal years
        """
        start_of_year = dates.astype('datetime64[Y]')
        years = start_of_year.astype(int) + 1970
        doy = (dates - start_of_year).astype('timedelta64[D]').astype(int) + 1
        ddates = years + doy/365.25
        return ddates


if __name__ == "__main__":
    import doctest
    doctest.testmod()
