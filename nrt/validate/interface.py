import copy, bisect
import functools
from typing import Dict, TYPE_CHECKING
import sqlite3

from traitlets import HasTraits, Int, Unicode, List, observe
from IPython.display import display
import ipywidgets as ipw
from ipyevents import Event
from ipyleaflet import GeoJSON
import numpy as np
from bqplot import Scatter, Lines, LinearScale, DateScale, Axis, Figure
from shapely.geometry import shape, mapping, Point
from rasterio import warp # TODO: use pyproj instead
from rasterio.crs import CRS

from nrt.validate import utils
from nrt.validate.composites import SimpleComposite
from nrt.validate.indices import *
from nrt.validate.fitting import PartitionedHarmonicTrendModel
from nrt.validate.segments import Segmentation

if TYPE_CHECKING:
    from nrt.validate.loader import BaseLoader
    from ipyleaflet import Map


class SegmentLabellingMenu:
    pass



class Chips(HasTraits):
    breakpoints = List()
    highlight = Int(allow_none=True)
    """A container with observable traits and many elementary methods to host image chips

    Examples:
        >>> import xarray as xr
        >>> import numpy as np
        >>> from nrt.validate.interface import Chips

        >>> cube = xr.open_dataset('/home/loic/Downloads/czechia_nrt_test.nc')
        >>> geom = {'type': 'Point', 'coordinates': [4813210, 2935950]}
        >>> chips = Chips.from_cube_and_geom(ds=cube, geom=geom,
        ...                                  breakpoints=[np.datetime64('2018-09-28T10:00:19.024000000'),
        ...                                               np.datetime64('2019-02-27T09:50:31.024000000'),
        ...                                               np.datetime64('2021-10-29T09:50:29.024000000')])
        >>> chips.display()
        >>> # Add breakpoint either by clicking on a chip, or running the following method
        >>> chips.add_or_remove_breakpoint(33)
    """
    def __init__(self, dates, images, breakpoints=[]):
        self.dates = dates
        self.images = images
        self.breakpoints = breakpoints
        box_layout = ipw.Layout(
            display='flex',
            flex_flow='row wrap',
            align_items='stretch',
            width='100%',
            height='800px',  # Set a fixed height (modify as needed)
            overflow='auto'  # Add scrollability
        )
        self.box_layout = box_layout
        self.widget = ipw.Box(children=self.images,
                              layout=box_layout)
        self.highlight = None # This is a trait that changes when individual chips are hovered
        # Add event handler to each chip
        for idx, image in enumerate(self.images):
            event = Event(source=image,
                          watched_events = ['mouseenter', 'mouseleave', 'click'])
            event.on_dom_event(functools.partial(self._handle_chip_event, idx))

        # Add border around chips for breakpoints present at instantiation
        for bp in self.breakpoints:
            idx = np.where(self.dates == bp)[0][0]
            self.images[idx].layout.border = '2px solid blue'

    def update(self, dates, images, breakpoints=[]):
        """Update all elements without requiring creation of a new instance
        """
        self.dates = dates
        self.images = images
        self.breakpoints = breakpoints
        self.widget = ipw.Box(children=self.images,
                              layout=self.box_layout)
        self.highlight = None # This is a trait that changes when individual chips are hovered
        # Add event handler to each chip
        for idx, image in enumerate(self.images):
            event = Event(source=image,
                          watched_events = ['mouseenter', 'mouseleave', 'click'])
            event.on_dom_event(functools.partial(self._handle_chip_event, idx))

        # Add border around chips for breakpoints present at instantiation
        for bp in self.breakpoints:
            idx = np.where(self.dates == bp)[0][0]
            self.images[idx].layout.border = '2px solid blue'


    @classmethod
    def from_cube_and_geom(cls, ds, geom, breakpoints=[],
                           compositor=SimpleComposite(),
                           window_size=500,
                           **kwargs):
        """Instantiate Chips from an xarray Dataset and a geometry

        Geometry and cube/Dataset must share the same coordinate reference system

        Args:
            ds (xarray.Dataset): The Dataset containing the data to display
            geom (dict): A geojson geometry (Point or Polygon) around which
                Dataset will be cropped and for which index time-series will
                be extracted
            breakpoints (list): Optional list of dates
            compositor (callable): Callable to transform a temporal slice of the provided
                Dataset into a 3D numpy array. See ``nrt.validate.composites`` module
                for examples
            window_size (float): Size of the bounding box used for cropping (created around
                the centroid of ``geom``). In CRS unit.
            **kwargs: Additional arguments passed to ``nrt.validate.utils.get_chips``
        """
        chips = utils.get_chips(ds=ds, geom=geom, size=window_size,
                                compositor=compositor, **kwargs)
        dates = ds.time.values
        instance = cls(dates=dates, images=chips, breakpoints=breakpoints)
        return instance

    def _handle_chip_event(self, idx, event):
        """Change the value of the highligh attribute to idx of the hovered chip"""
        # TODO: Using date would be safer (chips not in order) but adds some logic too find back idx, etc
        date = self.dates[idx]
        if event['type'] == 'mouseenter':
            self.highlight = idx
        if event['type'] == 'mouseleave':
            self.highlight = None
        if event['type'] == 'click':
            self.add_or_remove_breakpoint(idx)

    def add_or_remove_breakpoint(self, idx):
        date = self.dates[idx]
        bp = copy.deepcopy(self.breakpoints)
        if date in bp:
            bp.remove(date)
            self.images[idx].layout.border = ''
        else:
            bisect.insort(bp, date)
            self.images[idx].layout.border = '2px solid blue'
        self.breakpoints = bp

    def display(self):
        display(self.widget)


class Vits(HasTraits):
    breakpoints = List()
    order = Int(1) # HArmonic order
    current_vi = Unicode('NDVI')
    """Handle and display the vegetation index time-series
    """
    def __init__(self, dates, values,
                 breakpoints=[], default_vi='NDVI'):
        super().__init__()
        self.x_sc = DateScale()
        self.y_sc = LinearScale(min=float(np.nanmin(values[default_vi])),
                                max=float(np.nanmax(values[default_vi])))
        self.dates = dates
        self.values = values # Let's say this is a dict
        self.default_vi = default_vi
        # Dummy bqplot highlighted point out of view
        self.vi_values = Scatter(x=self.dates, y=self.values[self.default_vi],
                                 scales={'x': self.x_sc, 'y': self.y_sc})
        self.highlighted_point = Scatter(x=[-1000], y=[-1000],
                                         scales={'x': self.x_sc, 'y': self.y_sc},
                                         preserve_domain={'x': True, 'y': True},
                                         colors=['red'])
        self.vlines = [self._create_vline(bp) for bp in self.breakpoints]
        # Smooth fit lines
        self.model = PartitionedHarmonicTrendModel(dates)
        self.fitted_lines = self._create_fit_lines()
        self.plot = self._create_plot()
        self.breakpoints = breakpoints

    @classmethod
    def from_cube_and_geom(cls, ds, geom, breakpoints=[],
                           vis={'NDVI': NDVI(),
                                'CR-SWIR': CR_SWIR()},
                           default_vi='NDVI'):
        """Instantiate Vits from an xarray Dataset and a geometry

        Geometry and cube/Dataset must share the same coordinate reference system

        Args:
            ds (xarray.Dataset): The Dataset containing the data to display
            geom (dict): A geojson geometry (Point or Polygon) with which the
                time-series will be extracted (nearest pixel in case of Point,
                spatial average for Polygons)
            breakpoints (list): Optional list of dates
            vis (dict): Dictionary of callables to compute vegetation indices
                see ``nrt.validate.indices`` module for examples and already implemented
                indices
        """
        values = {k:utils.get_ts(ds=ds,
                                 geom=geom,
                                 vi_calculator=v)[1] for k,v in vis.items()}
        dates = ds.time.values
        instance = cls(dates=dates,
                       values=values,
                       breakpoints=breakpoints,
                       default_vi=default_vi)
        return instance

    def _create_vline(self, bp):
        return Lines(x=[bp, bp], y=[-1000, 1000],
                     scales={'x': self.x_sc, 'y': self.y_sc},
                     colors=['red'])

    def _create_plot(self):
        # Create axes
        x_ax = Axis(label='Dates (Year-month)', scale=self.x_sc,
                    tick_format='%m-%Y', tick_rotate=0)
        y_ax = Axis(label='Vegetation Index', scale=self.y_sc,
                    orientation='vertical', side='left')
        # Create and display the figure
        self.figure = Figure(marks=[self.vi_values,
                                    self.highlighted_point,
                                    *self.vlines,
                                    *self.fitted_lines],
                       axes=[x_ax, y_ax],
                       title='Sample temporal profile',
                       layout=ipw.Layout(width='100%',
                                         height='400px'),
                       animation_duration=500)

        # Add a dropdown widget to select VI
        dropdown_vi = ipw.Dropdown(options=self.values.keys(),
                                   value=self.default_vi,
                                   description='Index:')
        dropdown_order = ipw.Dropdown(options=[0,1,2,3,4,5],
                                      value=1,
                                      description='Order:')

        def update_scatter(change):
            self.vi_values.y = self.values[change['new']]
            self.y_sc.min = float(np.nanmin(self.values[change['new']]))
            self.y_sc.max = float(np.nanmax(self.values[change['new']]))
            self.current_vi = change['new']

        def update_order(change):
            self.order = change['new']

        dropdown_vi.observe(update_scatter, names='value')
        dropdown_order.observe(update_order, names='value')
        return ipw.VBox([ipw.HBox([dropdown_vi, dropdown_order]),
                         self.figure])

    def update_highlighted_point(self, idx):
        """Update the coordinates of the highlighted point based on idx.

        Args:
            idx (int or None): Index of the point to highlight or None.
        """
        if idx is not None:
            self.highlighted_point.x = [self.dates[idx]]
            self.highlighted_point.y = [self.values[self.current_vi][idx]]
        else:
            self.highlighted_point.x = [-1000]
            self.highlighted_point.y = [-1000]

    def _create_fit_lines(self):
        dates, predictions = self.model.fit_predict(self.values[self.current_vi],
                                                    self.breakpoints,
                                                    self.order)
        return  [Lines(x=d, y=p, scales={'x': self.x_sc, 'y': self.y_sc},
                       colors=['grey'])
                 for d,p in zip(dates, predictions)]

    @observe('breakpoints', 'order', 'current_vi')
    def redraw_fit_lines(self, change):
        self.fitted_lines = self._create_fit_lines()
        self.figure.marks = [self.vi_values,
                             self.highlighted_point,
                             *self.vlines,
                             *self.fitted_lines]

    @observe('breakpoints')
    def redraw_vlines(self, change):
        """Method to be called when a change event is detected on breakpoints
        """
        self.vlines = [self._create_vline(bp) for bp in self.breakpoints]
        # Update the figure with the new vlines
        self.figure.marks = [self.vi_values,
                             self.highlighted_point,
                             *self.vlines,
                             *self.fitted_lines]

    def display(self):
        display(self.plot)


class SegmentsLabellingInterface(HasTraits):
    current_idx = Int()
    def __init__(self, loader: 'BaseLoader', webmap: 'Map',
                 res: float,
                 labels: list, db_path: str = ':memory:'):
        self.current_idx = 0
        self.conn = sqlite3.connect(db_path)
        self.loader = loader
        self.webmap = webmap
        self.res = res
        self.labels = labels
        # Layouts
        self.webmap_layout = ipw.Layout(width='30%')
        self.chips_layout = ipw.Layout(width='70%')
        self.sidebar_layout = ipw.Layout(width='30%')
        self.vits_layout = ipw.Layout(width='70%')
        # 
        self.navigation_menu = ipw.Dropdown(options=range(len(loader)),
                                            value=self.current_idx)
        # Get data of first sample and build interface 
        fid, dates, images, values, geom, crs = self.loader[self.current_idx]
        self.seg = Segmentation.from_db_or_datelist(
            feature_id=fid,
            conn=self.conn,
            dates=dates,
            labels=self.labels)
        self.chips = Chips(dates, images, self.seg.breakpoints)
        self.vits = Vits(dates, values, self.seg.breakpoints)
        self.draw_webmap(geom=geom, res=self.res, crs=crs)
        # Apply layout
        self.vits.plot.layout = self.vits_layout
        self.chips.widget.layout = self.chips_layout
        # interface
        self.sidebar = ipw.VBox([self.navigation_menu,
                                 self.seg.segment_widgets])
        self.interface = ipw.VBox([
            ipw.HBox([self.vits.plot, self.webmap]),
            ipw.HBox([self.chips.widget, self.sidebar])
        ])
        # Connections between elements
        self.chips.observe(self._on_chip_hover, names=['highlight'])
        self.chips.observe(self._on_chip_click, names=['breakpoints'])
        self.navigation_menu.observe(self._on_idx_change, names=['value'])

    def _on_idx_change(self, change):
        self.current_idx = change['new']

    @observe('current_idx')
    def update_interface(self, change):
        """Current idx just changed, new data need to be loaded and the displayed
        elements updated accordingly
        """
        fid, dates, images, values, geom, crs = self.loader[change['new']]
        self.seg = Segmentation.from_db_or_datelist(
            feature_id=fid,
            conn=self.conn,
            dates=dates,
            labels=self.labels)
        self.chips = Chips(dates, images, self.seg.breakpoints)
        self.vits = Vits(dates, values, self.seg.breakpoints)
        # Apply layout
        self.vits.plot.layout = self.vits_layout
        self.chips.widget.layout = self.chips_layout
        # Update elements of the interface (oroginally immutable)
        first_row = list(self.interface.children[0].children) # vits, webmap
        second_row = list(self.interface.children[1].children) # chips, sidebar
        first_row[0] = self.vits.plot
        second_row[0] = self.chips.widget
        self.interface.children[0].children = tuple(first_row)
        self.interface.children[1].children = tuple(second_row)
        # Re-set callbacks (is that actually necessary)
        self.chips.observe(self._on_chip_hover, names=['highlight'])
        self.chips.observe(self._on_chip_click, names=['breakpoints'])
        # Update webmap
        self.update_webmap(geom=geom,
                           res=self.res,
                           crs=crs)

    def _on_chip_hover(self, change):
        self.vits.update_highlighted_point(change['new'])

    def _on_chip_click(self, change):
        self.vits.breakpoints = copy.deepcopy(change['new'])
        self.seg.breakpoints = copy.deepcopy(change['new'])

    def display(self):
        return self.interface

    def load_sample(self, idx):
        # Get 6 element tuple from loader
        # Check if sample already exist in the database and build breakpoints accordingly
        pass

    def draw_webmap(self, geom, res, crs):
        # Simply creates a geometry, add it to the map and center the map on it
        current_shape = shape(geom)
        if isinstance(current_shape, Point):
            current_shape = current_shape.buffer(res/2, cap_style=3)
        # TODO: use shapely ops + pyproj here instead of rasterio
        current_geom = warp.transform_geom(src_crs = crs,
                                           dst_crs = CRS.from_epsg(4326),
                                           geom=mapping(current_shape))
        centroid = shape(current_geom).centroid
        webmap_geom = GeoJSON(data=current_geom,
                               style = {'opacity': 1, 'fillOpacity': 0,
                                        'weight': 1, 'color': 'magenta'})
        self.webmap.add(webmap_geom)
        self.webmap.center = [centroid.y, centroid.x]
        self.webmap.zoom = 17

    def update_webmap(self, geom, res, crs):
        # Remove the last layer (that's normally where the GeoJSON layer is; to be improved)
        self.webmap.layers = self.webmap.layers[:-1]
        self.draw_webmap(geom, res, crs)

