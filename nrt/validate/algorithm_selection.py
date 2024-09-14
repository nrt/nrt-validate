import itertools
import datetime

from rasterio import features
import numpy as np


class GridSearch:
    def __init__(self, algorithm, param_grid, scoring):
        """Initialize the GridSearch with an algorithm, parameter grid, and scoring function.

        Args:
            algorithm (class): The detection algorithm class, e.g., EWMA, IQR, MoSum, etc.
            param_grid (dict): Dictionary of parameters to explore.
            scoring (callable): Scoring function that takes y_true, y_pred, and other keyword arguments.

        Example:
            >>> import random, itertools

            >>> from shapely.geometry import Point
            >>> import numpy as np

            >>> from nrt.monitor.ewma import EWMA
            >>> from nrt import data
            >>> from nrt.validate.indices import NDVI
            >>> from nrt.validate.metrics import f1_score_at_lag

            >>> # Load data
            >>> cube = data.romania_10m()
            >>> ndvi_da = NDVI(nir='B08')(cube)
            >>> # Split into historical and monitoring datasets
            >>> history_da = ndvi_da.sel(time=slice(None, '2019-01-01'))
            >>> monitor_da = ndvi_da.sel(time=slice('2019-01-01', None))
            >>> # Randomly sample 60 values ((x, y) pairs from history_da.x and history_da.y)
            >>> random.seed(42)  # For reproducibility
            >>> sampled_points = random.sample(list(itertools.product(history_da.x.values,
            ...                                                       history_da.y.values)), 60)
            >>> # Build point geometries from sampled (x, y) coordinates
            >>> geometries = [Point(x, y) for x, y in sampled_points]
            >>> # Associate value zero to 30 of these geometries and a random number between 17900 and 18200 to the rest
            >>> values = [0] * 30 + [random.randint(17900, 18200) for _ in range(30)]
            >>> shapes_true = list(zip(geometries, values))
            >>> # Define parameter grid for the algorithm (e.g., EWMA)
            >>> param_grid = {
            ...     'trend': [True, False],
            ...     'harmonic_order': [1, 2],
            ...     'sensitivity': [2, 3]
            ... }
            >>> # Instantiate and run GridSearch
            >>> gs = GridSearch(algorithm=EWMA, param_grid=param_grid,
            ...                 scoring=f1_score_at_lag)
            >>> gs.fit(history_da=history_da, monitor_da=monitor_da, shapes_true=shapes_true, reduce_shape=False,
            ...        begin=datetime.datetime(2019,1,1), negative_tolerance=20,
            ...        lag=60)
            >>> # Get the best parameters and score
            >>> best_params = gs.best_params()
            >>> best_score = gs.best_score()
            >>> print(f"Best Parameters: {best_params}")
            Best Parameters: {'trend': True, 'harmonic_order': 1, 'sensitivity': 2}
            >>> print(f"Best Score: {best_score}")
            Best Score: 0.9354838709677419
        """
        self.algorithm = algorithm
        self.param_grid = param_grid
        self.scoring = scoring
        self.results = []

    def fit(self, history_da, monitor_da, shapes_true,
            reduce_shape=False, **scoring_params):
        """Fit the algorithm using the parameter grid and calculate scores for each set of parameters.

        Args:
            history_da (xarray.DataArray): DataArray for the historical period.
            monitor_da (xarray.DataArray): DataArray for the monitoring period.
            shapes_true (iterable): Iterable of (geometry, value) pairs for the reference samples.
            reduce_shape (bool): If ``True``, the algorithm restricts computations to the rows and columns
                corresponding to the reference samples, potentially increasing computation speed by
                operating on a smaller subset of the data. However, if the number of reference samples
                exceeds the size of the array dimensions (e.g., more samples than the number of rows or columns),
                this may lead to the creation of a larger intermediate array due to the indexing strategy used,
                which can increase computation time instead of reducing it.
            scoring_params: Additional parameters for the scoring function.
        """
        # Rasterize the reference feature collection (shapes_true)
        y_true_2d = features.rasterize(
            shapes=shapes_true,
            out_shape=history_da.rio.shape,
            fill=-1,
            transform=history_da.rio.transform(),
            dtype=np.int16
        )
        mask = np.where(y_true_2d == -1, 0, 1)
        if reduce_shape:
            # Attempt to reduce mask
            y_indices, x_indices = np.where(mask == 1)
            ixgrid = np.ix_(y_indices, x_indices)
            mask = mask[ixgrid]
            diag = np.eye(mask.shape[0],dtype=bool)
            mask[~diag] = 0
            y_true_2d = y_true_2d[ixgrid]
            y_true_2d[~diag] = 0
            # Same for history and monitor dataarrays
            history_da = history_da.isel(indexers={'y':y_indices, 'x':x_indices})
            monitor_da = monitor_da.isel(indexers={'y':y_indices, 'x':x_indices})
        y_true = y_true_2d[mask == 1]

        # Generate all combinations of parameter values
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        param_combinations = list(itertools.product(*param_values))

        # Iterate over all combinations of parameters in the grid
        for param_set in param_combinations:
            params = dict(zip(param_names, param_set))
            # Initialize the algorithm with current parameters
            nrt_class = self.algorithm(mask=mask, **params)
            # Fit the model
            nrt_class.fit(dataarray=history_da)
            # Monitor for the given monitoring period
            for array, date in zip(monitor_da.values, monitor_da.time.values.astype('datetime64[s]').tolist()):
                nrt_class.monitor(array=array, date=date)
            # Retrieve the predicted detection dates
            y_pred = nrt_class.detection_date[mask == 1]
            # Calculate the score using the provided scoring function
            score = self.scoring(y_true=y_true, y_pred=y_pred, **scoring_params)
            # Save results
            self.results.append({
                'params': params,
                'score': score
            })

    def best_params(self):
        """Return the parameters that achieved the best score.

        Returns:
            dict: The parameters that yielded the highest score.
        """
        if not self.results:
            return None
        return max(self.results, key=lambda x: x['score'])['params']

    def best_score(self):
        """Return the best score from the search.

        Returns:
            float: The highest score from the parameter search.
        """
        if not self.results:
            return None
        return max(result['score'] for result in self.results)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
