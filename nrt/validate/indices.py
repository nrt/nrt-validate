from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class BaseS2Index(ABC):
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

    @abstractmethod
    def __call__(self, ds):
        """Compute an index as a DataArray from a Dataset"""
        pass


class NDVI(BaseS2Index):
    """NDVI calculator for Sentinel 2 data organized in an xarray Dataset

    By default, the red channel must be named ``'B03'`` and the nir channel
    ``'B8A'`` as per the ``BaseS2Index`` base class. These defaults can be modified
    at instantiation by passing for instance ``ndvi = NDVI(red='B03_20', nir='B08_20')``
    """
    def __call__(self, ds):
        ds = ds.astype(np.float32)
        da = (ds[self.nir] - ds[self.red]) / (ds[self.nir] + ds[self.red] + 0.0000001)
        return da


class NDMI(BaseS2Index):
    """NDMI calculator for Sentinel 2 data organized in an xarray Dataset

    By default, the swir channel must be named ``'B11'`` and the nir channel
    ``'B8A'`` as per the ``BaseS2Index`` base class. These defaults can be modified
    at instantiation by passing for instance ``ndmi = NDMI(nir='B08')``
    """
    def __call__(self, ds):
        ds = ds.astype(np.float32)
        da = (ds[self.nir] - ds[self.swir1]) / (ds[self.nir] + ds[self.swir1] + 0.0000001)
        return da


class CR_SWIR(BaseS2Index):
    """CR_SWIR calculator for Sentinel 2 data organized in an xarray Dataset

    This Continuum removal corresponds to the division between the observed SWIR1
    value and a value interpolated between NIR and SWIR2 hence amplifying the
    absoption feature of the SWIR1 region. The index was first proposed by
    Dutrieux et al. (2021) for the development of a near real time spruce dieback
    detection system named FORDEAD.
    Note that Sentinel 2A and 2B have slightly different central wavelengths, especially
    for SWIR2. The mean of the two sensors is used here
    """
    def __call__(self, ds):
        ds = ds.astype(np.float32)
        da = np.divide(ds[self.swir1],
                       ds[self.nir] - (1612 - 864) * np.divide(ds[self.swir2] - ds[self.nir],
                                                               2194 - 864))
        return da


class NCDI(BaseS2Index):
    """Experimental Normalized SWIR1 Continuum Difference Index

    A normalized variation of the ``CR_SWIR`` index. Instead of dividing the
    SWIR1 reflectance by the SWIR1 continuum, this index computes a normalized
    difference between them. It is therefore calculated as:

    .. math::

        NDCI = \frac{{\text{SWIR}_{1-C} - \text{SWIR}_{1-R}}}{{\text{SWIR}_{1-C} + \text{SWIR}_{1-R}}}

    Where:

        - :math:`SWIR_{1-C}` is the continuum interpolated between NIR and SWIR2
          for SWIR1 wavelength.
        - :math:`SWIR_{1-R}` is the SWIR1 reflectance value.
    """
    def __call__(self, ds):
        ds = ds.astype(np.float32)
        swirc = ds[self.nir] - (1612 - 864) * np.divide(ds[self.swir2] - ds[self.nir],
                                                        2194 - 864)
        swirr = ds[self.swir1]
        da = (swirc - swirr) / (swirc + swirr)
        return da
