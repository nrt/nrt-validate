import io

from PIL import Image as PILImage
from PIL import ImageDraw
import ipywidgets as ipw
import numpy as np
from affine import Affine
import rioxarray
from shapely.geometry import Point, Polygon, shape
from shapely.affinity import affine_transform
from rasterio import features

from nrt.validate.composites import SimpleComposite
from nrt.validate.indices import NDVI


def np2ipw(arr, geom=None, transform=None, res=20, scale=4,
           outline_color='magenta'):
    """Convert a 3 bands numpy array to an ipywidgets Image

    Args:
        arr (np.ndarray): Array of rescaled values between 0 and 1, 3 bands in
            RGB order. Bands are the last dimension
        geom (dict): A geojson geometry to draw on top of the image. Must be in
            the same Coordinate Reference System as the array
        transform (affine.Affine): The affine transform of the image/array
        res (float): The image resolution in the unit of CRS. It is used to outline
            the center pixel in case ``geom`` is a Point and does not necessarily
            need to match the actual image resolution (e.g. if a value of 3 times
            the actual resolution is provided, the 9 central pixels will be
            outlined).
        scale (int): Scaling factor to increase image size
        outline_color (str): Color of the (expanded) geometry outline. See the
            `Matplotlib Named Colors Gallery <https://matplotlib.org/stable/gallery/color/named_colors.html#css-colors>`_

    Returns:
        ipywidgets.Image: The image with geometry overlay, in ipywidgets Image format
    """
    img = PILImage.fromarray((arr * 255).astype(np.uint8))
    img = img.resize(size=(arr.shape[0] * scale, arr.shape[1] * scale),
                     resample=PILImage.NEAREST)
    # Modify transform to new scale
    scaled_transform = transform * Affine.scale(1 / scale)
    scaled_transform = ~scaled_transform
    # Get polygon coordinates in PIL format
    shape_ = shape(geom)
    if isinstance(shape_, Point):
        shape_ = shape_.buffer(res/2, cap_style=3)
    shape_ = affine_transform(shape_, scaled_transform.to_shapely())
    x,y = shape_.exterior.coords.xy
    polygon_coordinates = list(zip(x,y))
    # Draw polygon on image
    draw = ImageDraw.Draw(img)
    draw.polygon(polygon_coordinates, fill=None, outline=outline_color)
    # Save image to fileobject and reload as ipw.Image
    with io.BytesIO() as fileobj:
        img.save(fileobj, 'PNG')
        img_b = fileobj.getvalue()
    img_w = ipw.Image(value=img_b)
    return img_w


def get_chips(ds, geom, size, compositor=SimpleComposite(), res=None, scale=4,
              outline_color='magenta'):
    """Prepare a list of chips (ipywidget.Images) croped from a xarray Dataset

    Args:
        ds (xarray.Dataset): A mulivariate and multidimentional data cube containing
            the data to create the chips
        geom (dict): A geojson geometry to define the cropping location and overlay
            on each chip. Must spatially intersect with ``ds`` and be in the same
            CRS
        size (float): Size of the bounding box used for cropping (created around
            the centroid of ``geom``). In CRS unit.
        compositor (callable): Callable to transform a temporal slice of the provided
            Dataset into a 3D numpy array. See the ``nrt.validate.composites`` module
            for examples
        res (float): An optional value defining the size of the overlayed square
            geometry in case ``geom`` is a Point. If ``None`` and ``geom`` is a
            point, the Dataset resolution is used.
        scale (int): Scaling factor to increase image size
        outline_color (str): Color of the (expanded) geometry outline. See the
            `Matplotlib Named Colors Gallery <https://matplotlib.org/stable/gallery/color/named_colors.html#css-colors>`_

    Examples:
        >>> import xarray as xr
        >>> from nrt.validate import utils
        >>> from nrt.validate.composites import TasseledCapComposite
        >>> import ipywidgets as ipw
        >>> #TODO: Change the line below with cube from nrt.data package
        >>> cube = xr.open_dataset('/home/loic/Downloads/czechia_nrt_test.nc')
        >>> geom_point = {'type': 'Point', 'coordinates': [4813210, 2935950]}
        >>> geom_poly = {"type": "Polygon", "coordinates": [[[4813283, 2935951],
        ...                                                  [4813250, 2935998],
        ...                                                  [4813193, 2936019],
        ...                                                  [4813159, 2936013],
        ...                                                  [4813134, 2935956],
        ...                                                  [4813146, 2935899],
        ...                                                  [4813204, 2935877],
        ...                                                  [4813232, 2935869],
        ...                                                  [4813279, 2935927],
        ...                                                  [4813277, 2935967],
        ...                                                  [4813283, 2935951]]]}
        >>> box_layout = ipw.Layout(display='flex',
        ...                         flex_flow='row wrap',
        ...                         align_items='stretch',
        ...                         width='100%',
        ...                         height='800px',
        ...                         overflow='auto')
        >>> chips_point = utils.get_chips(ds=cube, geom=geom_point, size=300,
        ...                               compositor=TasseledCapComposite(), res=None)
        >>> box = ipw.Box(children=chips_point, layout=box_layout)
        >>> box
        >>> chips_poly = utils.get_chips(ds=cube, geom=geom_poly, size=200, res=None)
        >>> box = ipw.Box(children=chips_poly, layout=box_layout)
        >>> box

    Returns:
        list: List of ipywidgets.Image with geometry overlay
    """
    if res is None:
        res = ds.rio.resolution()[0]
    shape_ = shape(geom)
    centroid = shape_.centroid
    bbox = centroid.buffer(size/2.0).bounds
    cube_sub = ds.rio.clip_box(*bbox)
    cube_sub = cube_sub.rio.pad_box(*bbox, constant_values=0)
    transform = cube_sub.rio.transform()
    imgs = []
    for date in cube_sub.time.values:
        slice_ = cube_sub.sel(time=date)
        rgb = compositor(slice_)
        imgs.append(np2ipw(rgb, geom=geom, transform=transform,
                           res=res, outline_color=outline_color,
                           scale=scale))
    return imgs


def get_ts(ds, geom, vi_calculator=NDVI()):
    """Extract a time-series and compute desired index for a geometry overlayed on a Dataset

    Args:
        ds (xarray.Dataset): The dataset from which to compute and extract the
            time-series
        geom (dict): A geojson geometry (Point or Polygon). If point, the nearest
            pixel is extracted; if Polygon, spatial average excluding ``nan`` s is
            computed for each time-step.
        vi_calculator (callable): A callable to process a DataArray containing the
            desired index from the dataset. See the ``nrt.validate.indices`` module for
            examples and already implemented simple transforms

    Examples:
        >>> import xarray as xr
        >>> from nrt.validate import utils
        >>> from nrt.validate.indices import NDVI, CR_SWIR
        >>> from nrt.validate.xr_transforms import S2CloudMasking
        >>> import ipywidgets as ipw
        >>> from bqplot import DateScale, LinearScale, Axis, Scatter, Figure

        >>> cube = xr.open_dataset('/home/loic/Downloads/czechia_nrt_test.nc')
        >>> geom_point = {'type': 'Point', 'coordinates': [4813210, 2935950]}
        >>> geom_poly = {"type": "Polygon", "coordinates": [[[4813283, 2935951],
        ...                                                  [4813250, 2935998],
        ...                                                  [4813193, 2936019],
        ...                                                  [4813159, 2936013],
        ...                                                  [4813134, 2935956],
        ...                                                  [4813146, 2935899],
        ...                                                  [4813204, 2935877],
        ...                                                  [4813232, 2935869],
        ...                                                  [4813279, 2935927],
        ...                                                  [4813277, 2935967],
        ...                                                  [4813283, 2935951]]]}
        >>> dates, ts_point = utils.get_ts(ds=cube, geom=geom_point,
        ...                         vi_calculator=utils.combine_transforms(S2CloudMasking(), NDVI()))
        >>> _, ts_poly = utils.get_ts(ds=cube, geom=geom_poly,
        ...                         vi_calculator=utils.combine_transforms(S2CloudMasking(), CR_SWIR()))
        >>> # Visualize using bqplot
        >>> x_scale = DateScale()
        >>> y_scale = LinearScale()
        >>> x_ax = Axis(label='Date', scale=x_scale, tick_format='%m-%Y', tick_rotate=45)
        >>> y_ax = Axis(label='Value', scale=y_scale, orientation='vertical')
        >>> point_values = Scatter(x=dates, y=ts_point, scales={'x': x_scale, 'y': y_scale}, colors='green')
        >>> polygon_values = Scatter(x=dates, y=ts_poly, scales={'x': x_scale, 'y': y_scale}, colors='blue')
        >>> fig = Figure(marks=[point_values, polygon_values], axes=[x_ax, y_ax])
        >>> fig

    Returns:
        tuple: Tuple of two elements; Array of dates and array of VI values
    """
    shape_ = shape(geom)
    if isinstance(shape_, Point):
        # Handle point geometry
        centroid = shape_
        cube_sub = ds.sel(x=centroid.x, y=centroid.y, method='nearest')
        da = vi_calculator(cube_sub)
        return da.time.values, da.values

    elif isinstance(shape_, Polygon):
        # Handle polygon geometry
        ds_clipped = ds.rio.clip([geom])
        da = vi_calculator(ds_clipped)
        spatial_avg = da.mean(dim=['x', 'y'], skipna=True)
        return spatial_avg.time.values, spatial_avg.values
    else:
        raise ValueError('Unsuported geometry type')


def combine_transforms(*transforms):
    """Utility function to combine multiple transforms"""
    def combined(ds):
        for transform in transforms:
            ds = transform(ds)
        return ds
    return combined

