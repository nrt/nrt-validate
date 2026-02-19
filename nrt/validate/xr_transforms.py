"""Callables to apply a transformation to a xarray Dataset
Usually returns another xarray Dataset
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import xarray as xr
import pandas as pd


@dataclass
class BaseS2Transform(ABC):
    """Abstract base class for Sentinel2 based indices"""
    blue: str = 'B02'
    green: str = 'B03'
    red: str = 'B04'
    nir: str = 'B8A'
    re1: str = 'B05'
    re2: str = 'B06'
    re3: str = 'B07'
    swir1: str = 'B11'
    swir2: str = 'B12'
    scl: str = 'SCL'

    @abstractmethod
    def __call__(self, ds):
        """Apply a transformation to the Dataset"""
        pass


@dataclass
class S2CloudMasking(BaseS2Transform):
    """Replace pixels identified as clouds, shadows, snow or dark pixels by np.nan
    """
    valid_scl_values: list = field(default_factory=lambda: [2, 4, 5, 6, 7])

    def __call__(self, ds):
        cloud_mask = ds[self.scl].isin(self.valid_scl_values)
        ds = ds.where(cloud_mask)
        return ds


@dataclass
class S2MonthlyBest(BaseS2Transform):
    """Select the best image per month based on valid data percentage and date

    The best image selection algorithm works by computing a score for each image based on
    the percentage of valid data and the temporal proximity to the center of the month.

    Args:
        ds (xr.Dataset): xarray Dataset with time dimension
        scl (str): name of the SCL variable in the dataset
        valid_scl_values (list): list of valid SCL values used to compute
            the valid data score
        weight_valid (float): weight for the valid data score
        weight_time (float): weight for the temporal proximity score

    Example:
        >>> from nrt import data
        >>> from nrt.validate.xr_transforms import S2MonthlyBest

        >>> # Load a small spatial-temporal subset of the dataset
        >>> ds = data.germany_zarr()
        >>> ds_sub = ds.isel(x=slice(100, 110), y=slice(200, 210))

        >>> # Apply the transformation
        >>> transform = S2MonthlyBest()
        >>> ds_best = transform(ds_sub)
        >>> print(ds_best)
        <xarray.Dataset> Size: 261kB
        Dimensions:      (time: 50, y: 10, x: 10)
        Coordinates:
            spatial_ref  int32 4B 3035
          * time         (time) datetime64[ns] 400B 2018-02-22T10:40:29.028000 ... 20...
          * x            (x) float64 80B 4.134e+06 4.134e+06 ... 4.134e+06 4.134e+06
          * y            (y) float64 80B 3.111e+06 3.111e+06 ... 3.111e+06 3.111e+06
        Data variables:
            B02          (time, y, x) float64 40kB 0.0501 0.0656 0.0736 ... nan nan nan
            B03          (time, y, x) float64 40kB 0.0722 0.0863 0.0909 ... nan nan nan
            B04          (time, y, x) float64 40kB 0.0577 0.0862 0.1035 ... nan nan nan
            B08          (time, y, x) float64 40kB 0.3225 0.2855 0.2602 ... nan nan nan
            B11          (time, y, x) float64 40kB 0.2112 0.2219 0.2331 ... nan nan nan
            B12          (time, y, x) float64 40kB 0.1264 0.1362 0.1457 ... nan nan nan
            SCL          (time, y, x) float32 20kB 4.0 4.0 4.0 4.0 ... 0.0 0.0 0.0 0.0

    """
    valid_scl_values: list = field(default_factory=lambda: [4, 5, 6, 7, 11])
    weight_valid: float = 1.0
    weight_time: float = 0.5

    def __call__(self, ds: xr.Dataset) -> xr.Dataset:
        # Compute clear observation score
        scl = ds[self.scl].compute()
        valid_data_mask = scl.isin(self.valid_scl_values)
        clear_obs_score = valid_data_mask.mean(dim=['x', 'y']).rename('Valid')

        # Compute temporal proximity score to the 15th of each month
        target_dates = pd.to_datetime([t.replace(day=15) for t in pd.to_datetime(ds.time.values)])
        day_diff = np.abs((pd.to_datetime(ds.time.values) - target_dates).days)
        temporal_score = xr.DataArray(1 - np.clip(day_diff / 20, 0, 1), coords=[ds.time], dims=["time"])

        # Combine scores
        score = clear_obs_score * self.weight_valid + temporal_score * self.weight_time

        # Select the best image per month
        month_index = score.time.dt.strftime('%Y-%m')
        tmask = score.groupby(month_index).apply(lambda group: group == group.max())

        ds_best = ds.where(tmask, drop=True)
        return ds_best


@dataclass
class S2OffsetCorrection(BaseS2Transform):
    """Apply radiometric offset correction to Sentinel-2 bands.

    Starting from processing baseline 04.00, Sentinel-2 data includes a
    radiometric offset (1000). This transform subtracts that offset
    from all spectral bands while preserving the SCL (Scene Classification Layer).

    Args:
        offset (int): The value to subtract from the bands. Default is 1000.
    """
    offset: int = 1000

    def __call__(self, ds: xr.Dataset) -> xr.Dataset:
        all_bands = {
            self.blue, self.green, self.red, self.nir,
            self.re1, self.re2, self.re3, self.swir1, self.swir2
        }
        # Intersection ensures we only try to transform bands actually present in ds
        bands_to_correct = [b for b in all_bands if b in ds.data_vars]

        # 2. Convert to float to avoid underflow/wrap-around in uint16
        # and to allow for negative values if they occur during processing
        ds_corrected = ds.copy()
        for band in bands_to_correct:
            ds_corrected[band] = ds_corrected[band].astype(np.float32) - self.offset
        return ds_corrected


if __name__ == "__main__":
    import doctest
    doctest.testmod()
