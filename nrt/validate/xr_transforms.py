"""Callables to apply a transformation to a xarray Dataset
Usually returns another xarray Dataset
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class BaseS2Transform(ABC):
    """Abstract base class for Sentinel2 based indices"""
    blue: str = 'B02_20'
    green: str = 'B03_20'
    red: str = 'B04_20'
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
