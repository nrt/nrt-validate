"""Data loaders from various sources (disk, STAC, gee) to be used by the Interface

Loaders inputs are at least a feature collection with a property that can be used
as a primary key and a way to retrieve xarray Dataset (a single or multiple
netcdf files/zarr stores; a STAC API collection; etc).
The loader are subscritable (loader[n] is used to access data of n-th element),
have at least a __len__ method. Data returned by subscript are in the form of a
6 element tuple (unique_id, dates, chips, ts, geom, crs).

TODOs/questions:
    - Should WKT geometries be stored in the database? To facilitate disaster recovery
    - CRS handling, none for now. Needed? Yes, it may be needed for webmap overlay.
    - The dates array must be numpy.datetime64 with Day precision. Document that
      somewhere for people who wish to write their own loader
"""
from typing import List, Dict, Any, Optional, Union, Callable, TYPE_CHECKING
from abc import ABC, abstractmethod
import datetime
import threading
from collections import OrderedDict
import functools

import xarray as xr
import numpy as np
from shapely.geometry import shape, box
from rtree.index import Index
import rioxarray

from nrt.validate import utils

if TYPE_CHECKING:
    from pystac_client import Client


class BaseLoader(ABC):
    def __init__(self,
                 fc: List[Dict[str, Any]],
                 key: str,
                 crs: Any,
                 prefetch: Optional[int] = None,
                 cache_size: Optional[int] = 20):
        self.fc = fc
        self.key = key
        self.crs = crs
        self.prefetch = prefetch
        self.cache_size = cache_size
        self.cache = OrderedDict()
        self.lock = threading.Lock()

        self._validate_unique_key(fc, key)

    def _validate_unique_key(self, fc: List[Dict[str, Any]], key: str):
        try:
            unique_idx = [feat['properties'][key] for feat in fc]
        except KeyError:
            raise ValueError(f"Key '{key}' not found in one or more features' properties.")
        if len(unique_idx) != len(set(unique_idx)):
            raise ValueError(f"Provided key '{key}' contains non-unique values.")

    @functools.cached_property
    def fids(self):
        """Return the list of unique feature ids"""
        return [(idx, feat['properties'][self.key]) for idx, feat in enumerate(self.fc)]

    def __len__(self):
        return len(self.fc)

    @abstractmethod
    def _load_data(self, idx):
        """Abstract method to be implemented in child classes to load data for a given id."""
        pass

    def _prefetch(self, start_idx):
        def prefetch_task(start_idx):
            for i in range(start_idx, min(start_idx + self.prefetch, len(self.fc))):
                if i not in self.cache:
                    data = self._load_data(i)
                    with self.lock:
                        self.cache[i] = data
                        # Trim cache if it exceeds the cache_size
                        while len(self.cache) > self.cache_size:
                            self.cache.popitem(last=False)

        thread = threading.Thread(target=prefetch_task, args=(start_idx,))
        thread.daemon = True
        thread.start()

    def __getitem__(self, idx):
        if self.prefetch is not None:
            self._prefetch(idx + 1)

        with self.lock:
            if idx in self.cache:
                return self.cache[idx]

        data = self._load_data(idx)
        with self.lock:
            self.cache[idx] = data
            # Trim cache if it exceeds the cache_size
            while len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)

        return data


class STACLoader(BaseLoader):
    """Loader to prepare data indexed into a STAC Catalogue

    Args:
        fc (list): A feature collection. Can be Points, Polygons or a mix of the two.
            Must contain a property that can be used as a unique key
        key (str): Name of the feature collection property to be used as unique
            identifier
        crs (CRS): A coordinate reference object, from fiona, rasterio, pyproj, etc
            representing projection of both ``fc`` and ``datasets``
        client (Client): The STAC API client.
        collection_id (str): The STAC collection ID to query.
        bands (list): List of bands to load from the STAC collection.
        datetime (list): List of datetime objects or strings defining the time range to query.
        resampling (str or dict, optional): Resampling method(s) for the bands.
        vis (dict): Dictionary of callables to compute vegetation indices
            see ``nrt.validate.indices`` module for examples and already implemented
            indices.
        window_size (float): Size of the bounding box used for cropping (created around
            the centroid of ``geom``). In CRS unit.
        compositor (callable): Callable to transform a temporal slice of the provided
            Dataset into a 3D numpy array. See ``nrt.validate.composites`` module
            for examples.
        query (dict, optional): Additional query parameters for the STAC API.
        res (float, optional): Spatial resolution for the output data.
        prefetch (int, optional): Number of items to prefetch and cache.
        cache_size (int, optional): Maximum size of the cache.
        xr_transform (callable): Callable that takes an xarray Dataset as input
            and returns another xarray Dataset. This operation is applied to the
            spatial subset of the cube used for generating chips and VI time-series.
            Generally used for pre-processing steps such as removal of "empty"
            slices, or data scaling.
        kwargs (dict, optional): Additional arguments passed to ``nrt.validate.utils.get_chips``.

    Returns:
        tuple: A tuple of 6 elements:
            - The unique key of the sample
            - Array of numpy.datetime64
            - List of ipywidgets.Image (the image chips)
            - Dictionary of values for multiple vegetation indices
            - The sample geometry
            - The CRS object

    Examples:
        >>> import datetime
        >>> from nrt.validate.loaders import STACLoader
        >>> from pystac_client import Client
        >>> import planetary_computer as pc
        >>> from pyproj import CRS
        >>> from nrt.validate import utils
        >>> from nrt.validate.indices import *
        >>> from nrt.validate.composites import *
        >>> from nrt.validate.xr_transforms import *

        >>> fc = [{'geometry': {'type': 'Point', 'coordinates': (4033880, 3217980)},
        ...        'properties': {'idx': 1}},
        ...       {'geometry': {'type': 'Point', 'coordinates': (4395490, 3038090)},
        ...        'properties': {'idx': 2}},
        ...       {'geometry': {'type': 'Point', 'coordinates': (4713260, 2931020)},
        ...                     'properties': {'idx': 3}}]
        >>> key = 'idx'
        >>> crs = CRS.from_epsg(3035)
        >>> catalog = Client.open('https://planetarycomputer.microsoft.com/api/stac/v1',
        ...                         modifier=pc.sign_inplace)
        >>> collection_id = 'sentinel-2-l2a'
        >>> bands = ['B02', 'B03', 'B04', 'B08', 'B11', 'B12', 'SCL']
        >>> resampling = {band: 'nearest' if band == 'SCL' else 'cubic' for band in bands}
        >>> dt = [datetime.datetime(2019, 1, 1), datetime.datetime(2021,12,31)]
        >>> vis = {'NDVI': utils.combine_transforms(S2CloudMasking(), CR_SWIR(nir='B08')),
        ...        'CR-SWIR': utils.combine_transforms(S2CloudMasking(), NDVI(red='B04', nir='B08'))}
        >>> window_size = 300
        >>> compositor = SimpleComposite(r='B04', g='B03', b='B02')
        >>> query = {"eo:cloud_cover": {"lt": 10}}
        >>> res = 10
        >>> prefetch = 5
        >>> loader = STACLoader(fc=fc,
        ...                     key=key,
        ...                     crs=crs,
        ...                     client=catalog,
        ...                     collection_id=collection_id,
        ...                     bands=bands,
        ...                     resampling=resampling,
        ...                     datetime=dt,
        ...                     vis=vis,
        ...                     window_size=window_size,
        ...                     compositor=compositor,
        ...                     query=query,
        ...                     res=res,
        ...                     prefetch=prefetch)
        >>> print(loader[0])
    """
    def __init__(self,
                 fc: List[Dict[str, Any]],
                 key: str,
                 crs: Any,
                 client: 'Client',
                 collection_id: str,
                 bands: List[str],
                 datetime: List[Union[datetime.datetime, str]],
                 resampling: Optional[Union[str, Dict[str, str]]],
                 vis: Dict[str, Callable[[xr.DataArray], Any]],
                 window_size: float,
                 compositor: Callable[[xr.Dataset], np.ndarray],
                 query: Optional[Dict],
                 xr_transform: Optional[Callable[[xr.Dataset], xr.Dataset]] = None,
                 res: Optional[float] = None,
                 prefetch: Optional[int] = 5,
                 cache_size: Optional[int] = 20,
                 **kwargs):
        super().__init__(fc, key, crs, prefetch, cache_size)
        self.client = client
        self.collection_id = collection_id
        self.bands = bands
        self.datetime = datetime
        self.resampling = resampling
        self.vis = vis
        self.window_size = window_size
        self.compositor = compositor
        self.query = query
        self.xr_transform = xr_transform
        self.res = res
        self.kwargs = kwargs

        try:
            from odc.stac import stac_load
            from odc.geo.geobox import GeoBox
        except ImportError:
            raise ImportError("You must install both odc-stac and odc-geo to use STACLoader.")

    def _load_data(self, idx):
        from odc.stac import stac_load
        from odc.geo.geobox import GeoBox

        feature = self.fc[idx]
        unique_idx = feature['properties'][self.key]
        bbox = shape(feature['geometry']).centroid.buffer(self.window_size / 2.0).bounds
        gbox = GeoBox.from_bbox(bbox, crs=self.crs, resolution=self.res)

        query = self.client.search(
            collections=[self.collection_id],
            bbox=gbox.geographic_extent.boundingbox.bbox,
            datetime=self.datetime,
            query=self.query,
        )

        ds = stac_load(
            query.items(),
            bands=self.bands,
            groupby='solar_day',
            chunks={'time': 1},
            geobox=gbox,
            resampling=self.resampling,
            fail_on_error=False
        ).compute()
        if self.xr_transform:
            ds = self.xr_transform(ds)
        # 'Reduce' time precision for compatibility with segments module 
        ds = ds.assign_coords(time=ds.time.values.astype('datetime64[D]'))

        dates = ds.time.values
        values = {
            k: utils.get_ts(ds=ds, geom=feature['geometry'], vi_calculator=v)[1]
            for k, v in self.vis.items()
        }
        chips = utils.get_chips(
            ds=ds,
            geom=feature['geometry'],
            size=self.window_size,
            compositor=self.compositor,
            res=self.res,
            **self.kwargs
        )

        return unique_idx, dates, chips, values, feature['geometry'], self.crs


class FileLoader(BaseLoader):
    """A loader to prepare locally accessible data

    Args:
        fc (list): A feature collection. Can be Points, Polygons or a mix of the two.
            Must contain a property that can be used as a unique key
        key (str): Name of the feature collection property to be used as unique
            identifier
        crs (CRS): A coordinate reference object, from fiona, rasterio, pyproj, etc
            representing projection of both ``fc`` and ``datasets``
        datasets (xr.Dataset or list): (list of) xarray Datasets containing the
            multispectral spatio-temporal data. They must all be in the same CRS
            as ``fc``. They can lazy loaded using dask (see ``chunks`` argument
            in ``open_dataset``.
        vis (dict): Dictionary of callables to compute vegetation indices
            see ``nrt.validate.indices`` module for examples and already implemented
            indices
        compositor (callable): Callable to transform a temporal slice of the provided
            Dataset into a 3D numpy array. See ``nrt.validate.composites`` module
            for examples
        window_size (float): Size of the bounding box used for cropping (created around
            the centroid of ``geom``). In CRS unit.
        xr_transform (callable): Callable that takes an xarray Dataset as input
            and returns another xarray Dataset. This operation is applied to the
            spatial subset of the cube used for generating chips and VI time-series.
            Generally used for pre-processing steps such as removal of "empty"
            slices, or data scaling.
        **kwargs: Additional arguments passed to ``nrt.validate.utils.get_chips``

    Returns:
        tuple: A tuple of 6 elements:
            - The unique key of the sample
            - Array of numpy.datetime64
            - List of ipywidgets.Image (the image chips)
            - Dictionary of values for multiple vegetation indices
            - The sample geometry
            - The CRS object

    Examples:
        >>> import xarray as xr
        >>> import numpy as np
        >>> from shapely.geometry import Point, mapping
        >>> import rioxarray

        >>> from nrt.validate.loaders import FileLoader
        >>> from nrt.validate import utils
        >>> from nrt.validate.indices import *
        >>> from nrt.validate.composites import *
        >>> from nrt.validate.xr_transforms import *

        >>> cube = xr.open_dataset('/home/loic/Downloads/czechia_nrt_test.nc', chunks=-1)
        >>> cube = cube.rename({'B02_20':'B02', 'B03_20': 'B03', 'B04_20': 'B04'})

        >>> geom = {'type': 'Point', 'coordinates': [4813210, 2935950]}
        >>> fc = [{'geometry': mapping(Point(4813210, 2935950)),
        ...        'properties': {'pid': 1}},
        ...       {'geometry': mapping(Point(4813350, 2934998)),
        ...        'properties': {'pid': 2}}]

        >>> loader = FileLoader(fc=fc,
        ...                     key='pid',
        ...                     crs=cube.rio.crs,
        ...                     datasets=cube,
        ...                     vis={'NDVI': utils.combine_transforms(S2CloudMasking(), CR_SWIR()),
        ...                          'CR-SWIR': utils.combine_transforms(S2CloudMasking(), NDVI())},
        ...                     window_size=300,
        ...                     compositor=SimpleComposite(),
        ...                     res=None)
        >>> print(len(loader[0]))
        6
    """
    def __init__(self,
                 fc: List[Dict[str, Any]],
                 key: str,
                 crs: Any,
                 datasets: Union[xr.Dataset, List[xr.Dataset]],
                 vis: Dict[str, Callable[[xr.DataArray], Any]],
                 window_size: float,
                 compositor: Callable[[xr.Dataset], np.ndarray],
                 xr_transform: Optional[Callable[[xr.Dataset], xr.Dataset]] = None,
                 res: Optional[float] = None,
                 prefetch: Optional[int] = 5,
                 cache_size: Optional[int] = 20,
                 **kwargs):
        super().__init__(fc, key, crs, prefetch, cache_size)
        self.datasets = self._validate_datasets(datasets)
        self.vis = vis
        self.window_size = window_size
        self.compositor = compositor
        self.xr_transform = xr_transform
        self.res = res
        self.kwargs = kwargs

        self.rtree = Index()
        for i, ds in enumerate(self.datasets):
            self.rtree.insert(i, ds.rio.bounds())

    def _validate_datasets(self,
                           datasets: Union[xr.Dataset, List[xr.Dataset]]) -> List[xr.Dataset]:
        """Validate the datasets input and normalize it to a list of xarray.Dataset objects."""
        if isinstance(datasets, xr.Dataset):
            return [datasets]
        if isinstance(datasets, list):
            if all(isinstance(ds, xr.Dataset) for ds in datasets):
                return datasets
            else:
                raise ValueError("All elements in the list must be xarray.Dataset objects")
        raise ValueError("Datasets must be either an xarray.Dataset or a list of xarray.Dataset objects")

    def _find_intersects(self, idx: int) -> xr.Dataset:
        geom = self.fc[idx]['geometry']
        shape_ = shape(geom)
        intersects_maybe = list(self.rtree.intersection(shape_.bounds))
        intersects_confirmed = [i for i in intersects_maybe if box(*self.datasets[i].rio.bounds()).intersects(shape_)]
        if not intersects_confirmed:
            raise ValueError(f'Geometry of feature {idx} does not intersect with any provided datasets')
        if len(intersects_confirmed) > 1:
            distance = [shape_.centroid.distance(box(*self.datasets[i].rio.bounds()).centroid)
                        for i in intersects_confirmed]
            intersects_confirmed = [intersects_confirmed[distance.index(min(distance))]]
        return self.datasets[intersects_confirmed[0]]

    def _load_data(self, idx):
        ds = self._find_intersects(idx)
        feature = self.fc[idx]
        bbox = shape(feature['geometry']).centroid.buffer(self.window_size / 2).bounds
        ds = ds.rio.clip_box(*bbox).compute()
        if self.xr_transform:
            ds = self.xr_transform(ds)
        ds = ds.assign_coords(time=ds.time.values.astype('datetime64[D]'))
        unique_idx = feature['properties'][self.key]
        dates = ds.time.values
        values = {k: utils.get_ts(ds=ds, geom=feature['geometry'], vi_calculator=v)[1]
                  for k, v in self.vis.items()}
        chips = utils.get_chips(ds=ds, geom=feature['geometry'], size=self.window_size,
                                compositor=self.compositor, res=self.res, **self.kwargs)
        return unique_idx, dates, chips, values, feature['geometry'], self.crs


if __name__ == "__main__":
    import doctest
    doctest.testmod()

