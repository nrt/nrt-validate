import copy, bisect
import functools

from traitlets import HasTraits
from IPython.display import display
import ipywidgets as ipw
from ipyevents import Event
import numpy as np

from nrt.validate import utils
from nrt.validate.composites import SimpleComposite


class Chips(HasTraits):
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
        self.breakpoints = breakpoints # Should this one be observed?
        box_layout = ipw.Layout(
            display='flex',
            flex_flow='row wrap',
            align_items='stretch',
            width='100%',
            height='800px',  # Set a fixed height (modify as needed)
            overflow='auto'  # Add scrollability
        )
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
