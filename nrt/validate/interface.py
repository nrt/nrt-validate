import os
import copy, bisect
import functools
from typing import Dict, TYPE_CHECKING
import sqlite3

from traitlets import HasTraits, Int, Unicode, List, observe
from IPython.display import display
import ipywidgets as ipw
from ipyevents import Event
from ipyleaflet import GeoJSON, LayersControl, TileLayer
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

class ResizableSplitter(ipw.HBox):
    _splitter_counter = 0
    
    def __init__(self, left_widget, right_widget, orientation='horizontal', 
                 initial_left_size='50%', min_left_size='10%', min_right_size='10%'):
        
        self.orientation = orientation
        self.initial_left_size = initial_left_size
        self.min_left_size = min_left_size
        self.min_right_size = min_right_size
        self.left_widget = left_widget
        self.right_widget = right_widget
 
        ResizableSplitter._splitter_counter += 1
        self.splitter_id = f'splitter-{ResizableSplitter._splitter_counter}'

        splitter_width = '8px' if orientation == 'horizontal' else '100%' 
        splitter_height = '100%' if orientation == 'horizontal' else '8px'
        
        splitter_html = f"""
        <div id="{self.splitter_id}" 
             style="width: 100%; 
                    height: 100%; 
                    background: #e0e0e0; 
                    cursor: {'col-resize' if orientation == 'horizontal' else 'row-resize'};
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    transition: background 0.2s;
                    position: relative;
                    z-index: 100;">
            <div style="width: {'4px' if orientation == 'horizontal' else '30px'};
                        height: {'30px' if orientation == 'horizontal' else '4px'};
                        background: #a0a0a0;
                        border-radius: 2px;
                        pointer-events: none;">
            </div>
        </div>
        <style>
        #{self.splitter_id}:hover {{
            background: #bdbdbd !important;
        }}
        </style>
        """
        
        self.splitter = ipw.HTML(
            value=splitter_html,
            layout=ipw.Layout(
                width=splitter_width,
                height=splitter_height,
                padding='0',
                margin='0',
                flex='0 0 auto',
                overflow='hidden'
            )
        )
        
        left_layout = ipw.Layout(
            width=initial_left_size if orientation == 'horizontal' else '100%',
            height='100%' if orientation == 'horizontal' else initial_left_size,
            overflow='auto',
            padding='0',
            margin='0',
            flex='0 0 auto'  
        )
        left_widget.layout = left_layout
        
        right_layout = ipw.Layout(
            width='auto' if orientation == 'horizontal' else '100%',
            height='100%' if orientation == 'horizontal' else 'auto',
            flex='1 1 auto',  
            overflow='auto',
            padding='0',
            margin='0'
        )
        right_widget.layout = right_layout
        
        container_layout = ipw.Layout(
            width='100%',
            height='100%',
            display='flex',
            flex_flow='row' if orientation == 'horizontal' else 'column',
            align_items='stretch', 
            padding='0',
            margin='0'
        )
        
        super().__init__(
            [left_widget, self.splitter, right_widget],
            layout=container_layout
        )
        
        self._inject_js()
    
    def _inject_js(self):

        js_code = f"""
        (function() {{
            const splitterId = '{self.splitter_id}';
            const orientation = '{self.orientation}';
            
            console.log('Script loaded for splitter:', splitterId);

            function initSplitter() {{
                const splitterEl = document.getElementById(splitterId);
                if (!splitterEl) {{
                    setTimeout(initSplitter, 200);
                    return;
                }}
                
                let widgetWrapper = splitterEl;
                while (widgetWrapper && 
                       !widgetWrapper.classList.contains('jupyter-widget') && 
                       !widgetWrapper.classList.contains('widget-inline-hbox') &&
                       !widgetWrapper.classList.contains('widget-inline-vbox')) {{
                    widgetWrapper = widgetWrapper.parentElement;
                }}
                
                if (!widgetWrapper) widgetWrapper = splitterEl.parentElement;

                if (!widgetWrapper || !widgetWrapper.parentElement) {{
                     console.error('Widget wrapper or parent not found');
                     return;
                }}
                
                const container = widgetWrapper.parentElement;
                const siblings = Array.from(container.children);
                const splitterIndex = siblings.indexOf(widgetWrapper);
                
                console.log('Splitter found at index:', splitterIndex, 'of', siblings.length);

                if (splitterIndex <= 0 || splitterIndex >= siblings.length - 1) {{
                    console.error('Splitter is not in the middle of widgets');
                    return;
                }}
                
                const leftWidgetWrapper = siblings[splitterIndex - 1];
                const rightWidgetWrapper = siblings[splitterIndex + 1];
                
                let isResizing = false;
                let startX, startY, startSize;
                
                splitterEl.addEventListener('mousedown', function(e) {{
                    console.log('Mousedown on splitter');
                    isResizing = true;
                    startX = e.clientX;
                    startY = e.clientY;
                    
                    const rect = leftWidgetWrapper.getBoundingClientRect();
                    startSize = (orientation === 'horizontal') ? rect.width : rect.height;
                    
                    document.body.style.cursor = (orientation === 'horizontal') ? 'col-resize' : 'row-resize';
                    document.body.style.userSelect = 'none';
                    
                    e.preventDefault();
                    e.stopPropagation();
                }});
                
                window.addEventListener('mousemove', function(e) {{
                    if (!isResizing) return;
                    
                    e.preventDefault();
                    
                    const containerRect = container.getBoundingClientRect();
                    const totalSize = (orientation === 'horizontal') ? containerRect.width : containerRect.height;
                    
                    let delta = (orientation === 'horizontal') ? (e.clientX - startX) : (e.clientY - startY);
                    let newSize = startSize + delta;
                    
                    const minSize = 50; 
                    const maxSize = totalSize - 50;
                    
                    if (newSize >= minSize && newSize <= maxSize) {{
                        if (orientation === 'horizontal') {{
                            leftWidgetWrapper.style.flex = '0 0 ' + newSize + 'px';
                            leftWidgetWrapper.style.width = newSize + 'px';
                            leftWidgetWrapper.style.minWidth = newSize + 'px';
                            leftWidgetWrapper.style.maxWidth = newSize + 'px';
                        }} else {{
                            leftWidgetWrapper.style.flex = '0 0 ' + newSize + 'px';
                            leftWidgetWrapper.style.height = newSize + 'px';
                            leftWidgetWrapper.style.minHeight = newSize + 'px';
                            leftWidgetWrapper.style.maxHeight = newSize + 'px';
                        }}
                        window.dispatchEvent(new Event('resize'));
                    }}
                }});
                
                window.addEventListener('mouseup', function(e) {{
                    if (isResizing) {{
                        console.log('Mouseup, resizing stopped');
                        isResizing = false;
                        document.body.style.cursor = '';
                        document.body.style.userSelect = '';
                    }}
                }});
                
                console.log('✓ Splitter initialized successfully:', splitterId);
            }}
            
            setTimeout(initSplitter, 100);
            setTimeout(initSplitter, 500);
            setTimeout(initSplitter, 1000);
        }})();
        """
        
        self._js_code = js_code


class Chips(HasTraits):
    breakpoints = List()
    highlight = Int(allow_none=True)
    selected_indices = List()  
    on_select_change = None   
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
    def __init__(self, dates, images, breakpoints=[], chip_size='150px', sample_box_size=10):
        self.dates = dates
        self.images = images
        self.breakpoints = breakpoints
        self.selected_indices = []  
        self.on_select_change = None
        self.chip_size = chip_size
        self.sample_box_size = sample_box_size

        self.wrapped_widgets = self._create_wrapped_widgets()
        
        box_layout = ipw.Layout(
            display='flex',
            flex_flow='row wrap',  
            align_items='flex-start', 
            align_content='flex-start',
            width='100%',  
            height='100%',
            overflow='auto' 
        )
        
        self.box_layout = box_layout
        
        self.widget = ipw.Box(children=self.wrapped_widgets,
                              layout=box_layout)
        self.highlight = None 

        fix_flex_js = '''
        <script>
        document.addEventListener('DOMContentLoaded', function() {
            setTimeout(function() {

                var chipsContainers = document.querySelectorAll('.widget-container');
                chipsContainers.forEach(function(container) {
                    if (container.style.display === 'flex' && container.style.flexFlow === 'row wrap') {
                        container.style.display = 'grid';
                        container.style.display = 'flex';
                    }
                });
            }, 1000);
        });
        </script>
        '''

        js_widget = ipw.HTML(value=fix_flex_js, layout=ipw.Layout(display='none'))
        self.widget = ipw.VBox([self.widget, js_widget], layout=box_layout)

        for idx, wrapper in enumerate(self.wrapped_widgets):
            event = Event(source=wrapper,
                          watched_events = ['mouseenter', 'mouseleave', 'click'])
            event.on_dom_event(functools.partial(self._handle_chip_event, idx))

        for bp in self.breakpoints:
            idx = np.where(self.dates == bp)[0][0]
            self.images[idx].layout.border = '2px solid blue'

    def _create_wrapped_widgets(self):
        wrapped = []
        try:
            chip_size_val = int(str(self.chip_size).replace('px', ''))
        except ValueError:
            chip_size_val = 150 
            
        size_str = f"{chip_size_val}px"

        show_boxes = True

        base_size = getattr(self, 'base_chip_size', 150)
        
        s_pct = (self.sample_box_size / base_size) * 100

        b_pct = s_pct * 3
        
        for image in self.images:
            if image.layout is None:
                image.layout = ipw.Layout()

            image.layout.width = '100%'
            image.layout.height = '100%'
            image.layout.min_width = '100%' 
            image.layout.min_height = '100%'
            image.layout.object_fit = 'fill' 
            image.layout.margin = '0px'
            image.layout.flex = '1 1 auto'
            
            overlay_html = ''
            if show_boxes:
                s_top = (100 - s_pct) / 2
                s_left = (100 - s_pct) / 2
                
                b_top = (100 - b_pct) / 2
                b_left = (100 - b_pct) / 2
                
                overlay_html = f'''
                <div style="position: relative; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; z-index: 10;">
                    <div style="position: absolute; top: {b_top}%; left: {b_left}%; width: {b_pct}%; height: {b_pct}%; border: 2px solid yellow; box-sizing: border-box;"></div>
                    <div style="position: absolute; top: {s_top}%; left: {s_left}%; width: {s_pct}%; height: {s_pct}%; border: 2px solid magenta; box-sizing: border-box;"></div>
                </div>
                '''
            else:
                overlay_html = '<div style="width: 100%; height: 100%; pointer-events: none;"></div>'

            overlay = ipw.HTML(
                value=overlay_html,
                layout=ipw.Layout(width='100%', height='100%')
            )
            
            container = ipw.GridBox(
                children=[image, overlay],
                layout=ipw.Layout(
                    width=size_str,
                    height=size_str,
                    grid_template_areas='"content"',
                    grid_template_columns='1fr',
                    grid_template_rows='1fr',
                    margin='2px',
                    flex='0 0 auto', 
                    overflow='hidden'
                )
            )

            image.layout.grid_area = 'content'
            overlay.layout.grid_area = 'content'
            
            wrapped.append(container)
        return wrapped

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
                Dataset into a 3D numpy array. See `nrt.validate.composites module
                for examples
            window_size (float): Size of the bounding box used for cropping (created around
                the centroid of `geom). In CRS unit.
            **kwargs: Additional arguments passed to `nrt.validate.utils.get_chips
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
            # self.add_or_remove_breakpoint(idx)
            if idx in self.selected_indices:
                self._unselect_index(idx)
            else:
                self._select_index(idx)

    def _select_index(self, idx):
        
        if idx not in self.selected_indices:
            self.selected_indices.append(idx)

            self.images[idx].layout.border = '2px solid red'

            if self.on_select_change:
                self.on_select_change(self.selected_indices)

    def _unselect_index(self, idx):

        if idx in self.selected_indices:
            self.selected_indices.remove(idx)

            if self.dates[idx] in self.breakpoints:
                self.images[idx].layout.border = '2px solid blue'
            else:
                self.images[idx].layout.border = ''

            if self.on_select_change:
                self.on_select_change(self.selected_indices)               


    def sync_selected_from_vits(self, selected_indices):

        for idx in self.selected_indices:
            if self.dates[idx] in self.breakpoints:
                self.images[idx].layout.border = '2px solid blue'
            else:
                self.images[idx].layout.border = ''

        self.selected_indices = selected_indices.copy()
        for idx in self.selected_indices:

            self.images[idx].layout.border = '2px solid red'

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

    def update_data(self, dates, images, breakpoints, selected_indices=None):
        """Update chips data without recreating the widget container"""
        self.dates = dates
        self.images = images
        self.breakpoints = breakpoints

        if selected_indices is not None:
            self.selected_indices = selected_indices
        else:
            self.selected_indices = []  

        # Re-create wrapped widgets with current settings
        self.wrapped_widgets = self._create_wrapped_widgets()
        
        # Add event handler to each chip container
        for idx, wrapper in enumerate(self.wrapped_widgets):
            event = Event(source=wrapper,
                          watched_events = ['mouseenter', 'mouseleave', 'click'])
            event.on_dom_event(functools.partial(self._handle_chip_event, idx))

        # Add border around chips for breakpoints
        for bp in self.breakpoints:
            # Check if bp is in dates to avoid errors if breakpoints don't match dates
            indices = np.where(self.dates == bp)[0]
            if len(indices) > 0:
                idx = indices[0]
                self.images[idx].layout.border = '2px solid blue'
        
        for idx in self.selected_indices:
            self.images[idx].layout.border = '2px solid red'
        
        # Update the children of the image container (the Box inside the VBox)
        # self.widget is VBox([box, js])
        if len(self.widget.children) > 0:
            image_box = self.widget.children[0]
            image_box.children = tuple(self.wrapped_widgets)

    def display(self):
        display(self.widget)


class Vits(HasTraits):
    breakpoints = List()
    order = Int(1) 
    current_vi = Unicode('NDVI')
    selected_indices = List()  
    on_select_change = None   
    """Handle and display the vegetation index time-series
    """
    def __init__(self, dates, values,
                 breakpoints=[], default_vi='NDVI'):
        super().__init__()
        self.x_sc = DateScale()
        self.y_sc = LinearScale(min=float(np.nanmin(values[default_vi])),
                                max=float(np.nanmax(values[default_vi])))
        self.dates = dates
        self.values = values 
        self.default_vi = default_vi
        self.colors =  ['blue'] * len(self.dates)
        self.selected_indices = []  
        self.on_select_change = None

        self.vi_values = Scatter(
            x=self.dates, y=self.values[self.default_vi],
            scales={'x': self.x_sc, 'y': self.y_sc},
            colors=self.colors,
            enable_hover=True,  
        )
        self.vi_values.on_element_click(self._handle_point_click)

        self.vlines = [self._create_vline(bp) for bp in self.breakpoints]

        self.model = PartitionedHarmonicTrendModel(dates)
        self.fitted_lines = self._create_fit_lines()
        self.plot = self._create_plot()
        self.breakpoints = breakpoints

    def _handle_point_click(self, element, event):

        if not event.get('data') or 'index' not in event.get('data', {}):
            return

        idx = event['data']['index']
        if idx is None:
            return
        if idx in self.selected_indices:
            self._unselect_index(idx)
        else:
            self._select_index(idx)

    def _select_index(self, idx):
        if idx not in self.selected_indices:
            self.selected_indices.append(idx)
            self._update_selected_colors()
            self._update_vlines_and_fit()
            if self.on_select_change:
                self.on_select_change(self.selected_indices)

    def _unselect_index(self, idx):
        if idx in self.selected_indices:
            self.selected_indices.remove(idx)
            self._update_selected_colors()
            self._update_vlines_and_fit()
            if self.on_select_change:
                self.on_select_change(self.selected_indices)

    def _update_selected_colors(self):
        self.colors = ['blue'] * len(self.dates)
        for idx in self.selected_indices:
            self.colors[idx] = 'red'
        self.vi_values.colors = self.colors

    def _update_vlines_and_fit(self):
        selected_dates = [self.dates[idx] for idx in sorted(self.selected_indices)]
        self.vlines = [self._create_vline(date) for date in selected_dates]

        self.fitted_lines = self._create_fit_lines_with_selected()

        self.figure.marks = [self.vi_values,
                             *self.vlines,
                             *self.fitted_lines]

    def _create_fit_lines_with_selected(self):
        if len(self.selected_indices) > 0:
            selected_dates = [self.dates[idx] for idx in sorted(self.selected_indices)]
            dates, predictions = self.model.fit_predict(self.values[self.current_vi],
                                                        selected_dates,
                                                        self.order)
            return [Lines(x=d, y=p, scales={'x': self.x_sc, 'y': self.y_sc},
                          colors=['grey'])
                    for d, p in zip(dates, predictions)]
        else:
            return []

    def sync_selected_from_chips(self, selected_indices):

        self.selected_indices = selected_indices.copy()
        self._update_selected_colors()
        self._update_vlines_and_fit()

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
                see `nrt.validate.indices module for examples and already implemented
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
                    tick_format='%Y-%m', tick_rotate=0)
        y_ax = Axis(label='Vegetation Index', scale=self.y_sc,
                    orientation='vertical', side='left')
        # Create and display the figure
        self.figure = Figure(marks=[self.vi_values,
                                    *self.vlines,
                                    *self.fitted_lines],
                       axes=[x_ax, y_ax],
                       title='Sample temporal profile',
                       animation_duration=500,
                       fig_margin={'top': 50, 'bottom': 50, 'left': 50, 'right': 50},
                       layout=ipw.Layout(flex='1 1 auto', width='100%', height='100%'))

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
        return ipw.VBox([ipw.HBox([dropdown_vi, dropdown_order],
                                 layout=ipw.Layout(overflow='visible')),
                         self.figure],
                        layout=ipw.Layout(height='100%', width='100%'))

    def update_highlighted_point(self, idx):
        """Update the color of the highlighted point based on idx.

        Args:
            idx (int or None): Index of the point to highlight or None.
        """

        self.colors = ['blue'] * len(self.colors)
        for selected_idx in self.selected_indices:
            self.colors[selected_idx] = 'red'

        if idx is not None and idx not in self.selected_indices:
            self.colors[idx] = 'orange'
        self.vi_values.colors = self.colors

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
                             *self.vlines,
                             *self.fitted_lines]

    @observe('breakpoints')
    def redraw_vlines(self, change):
        """Method to be called when a change event is detected on breakpoints
        """
        self.vlines = [self._create_vline(bp) for bp in self.breakpoints]
        # Update the figure with the new vlines
        self.figure.marks = [self.vi_values,
                             *self.vlines,
                             *self.fitted_lines]

    def update_data(self, dates, values, breakpoints):
        self.dates = dates
        self.values = values
        self.breakpoints = breakpoints
        
        self.model = PartitionedHarmonicTrendModel(dates)
        
        self.x_sc.min = None
        self.x_sc.max = None
        
        current_data = self.values[self.current_vi]
        self.y_sc.min = float(np.nanmin(current_data))
        self.y_sc.max = float(np.nanmax(current_data))
        
        self.vi_values.x = self.dates
        self.vi_values.y = current_data
        
        self.colors = ['blue'] * len(self.dates)
        self.vi_values.colors = self.colors
        
        self.selected_indices = []
        self.colors = ['blue'] * len(self.dates)
        self.vi_values.colors = self.colors

        self.redraw_vlines(None) 

    def display(self):
        display(self.plot)


class SegmentsLabellingInterface(HasTraits):
    current_idx = Int()
    def __init__(self, loader: 'BaseLoader', webmap: 'Map',
                 res: float,
                 labels: list, db_path: str = ':memory:', sample_box_size: int = 10):
        self.current_idx = 0
        self.sample_layer = None
        self.buffer_layer = None
        self.conn = sqlite3.connect(db_path)
        self.loader = loader
        self.webmap = webmap
        self.res = res
        self.labels = labels
        self.sample_box_size = sample_box_size
        
        for lyr in self.webmap.layers:
            if hasattr(lyr, 'base'):
                lyr.base = True
                
        # Check if vector layer already exists or just add OSM
        has_osm = any(getattr(l, 'name', '') == 'OpenStreetMap' for l in self.webmap.layers)
        if not has_osm:
            self.osm_layer = TileLayer(url='https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
                                       name='OpenStreetMap',
                                       base=True) 
            self.webmap.add_layer(self.osm_layer)
        else:
            # Find existing layer
            for l in self.webmap.layers:
                if getattr(l, 'name', '') == 'OpenStreetMap':
                    self.osm_layer = l
                    self.osm_layer.base = True
                    break
        
        # Add LayersControl if not present
        has_control = any(isinstance(c, LayersControl) for c in self.webmap.controls)
        if not has_control:
            control = LayersControl(position='topright')
            self.webmap.add_control(control)

        # Chip size configuration input
        initial_chip_size = 150
        min_required_size = int(6.1 * self.sample_box_size)
        if initial_chip_size < min_required_size:
            initial_chip_size = min_required_size
            
        self.base_chip_size = initial_chip_size
        
        self.chip_size_input = ipw.IntText(value=initial_chip_size, description='Chip Size:', layout=ipw.Layout(width='200px'))
        self.chip_size_input.observe(self._on_chip_size_change, names='value')

        # Layouts
        self.webmap_layout = ipw.Layout(width='30%', height='100%')
        self.sidebar_layout = ipw.Layout(width='100%',
                                         min_height='400px',  # Set minimum height
                                         height='100%',       # Fill parent container
                                         overflow='auto',
                                         align_items='stretch')
        self.sample_container_layout = ipw.Layout(width='100%',
                                                  height='100%',
                                                  overflow_y='auto',
                                                  border='1px solid black')
        # 
        self.present_in_db, self.not_present_in_db = self.get_fids()
        self.interpreted_list = self.create_interactive_list(self.present_in_db,
                                                             'lightcoral')
        self.not_interpreted_list = self.create_interactive_list(self.not_present_in_db,
                                                                 'darkgreen')
        
        container_style = ipw.Layout(flex='1 1 auto', width='50%', height='100%', margin='0 5px')
        
        self.interpreted_container = ipw.VBox([
            ipw.HTML('<h3 style="text-align: center; margin: 5px;">Interpreted Samples</h3>'),
            ipw.VBox([self.interpreted_list],
                    layout=self.sample_container_layout)
        ], layout=container_style)
        
        self.not_interpreted_container = ipw.VBox([
            ipw.HTML('<h3 style="text-align: center; margin: 5px;">To Interpret</h3>'),
            ipw.VBox([self.not_interpreted_list],
                    layout=self.sample_container_layout)
        ], layout=container_style)
        
        self.navigation_menu = ipw.HBox([self.not_interpreted_container,
                                         self.interpreted_container],
                                        layout=ipw.Layout(width='100%',
                                                          min_height='250px',
                                                          flex='1 1 auto',
                                                          align_items='stretch'))
        self.save_button = ipw.Button(description="Save",
                                      layout=ipw.Layout(width='80%',
                                                        max_width='300px',
                                                        min_height='30px',
                                                        align_self='center'),
                                      style={'button_color': 'blue'})
        self.logo = ipw.Image(value=open(os.path.join(os.path.dirname(__file__), 'static', 'ec-logo.png'), 'rb').read(),
                              format='png',
                              layout=ipw.Layout(
                                  width='90%',
                                  max_width='300px',
                                  height='50px',
                                  object_fit='contain',
                                  align_self='center'
                              ))
        self.save_button.on_click(self.save_to_db)
        # Get data of first sample and build interface 
        self.fid, dates, images, values, geom, crs = self.loader[self.current_idx]
        self.dates = dates  
        self.seg = Segmentation.from_db_or_datelist(
            feature_id=self.fid,
            conn=self.conn,
            dates=dates,
            labels=self.labels)
        
        self.window_size = getattr(self.loader, 'window_size', None)
        
        self.chips = Chips(self.dates, images, self.seg.breakpoints, 
                           chip_size=self.chip_size_input.value,
                           sample_box_size=self.sample_box_size)
        self.vits = Vits(self.dates, values, self.seg.breakpoints)

        self.vits.on_select_change = self._handle_selection_change
        self.chips.on_select_change = self._handle_selection_change

        selected_indices = []
        for bp in self.seg.breakpoints:
            indices = np.where(self.dates == bp)[0]
            if len(indices) > 0:
                selected_indices.append(indices[0])
        
        if not selected_indices and len(self.dates) > 0:
             first_idx = 0
             last_idx = len(self.dates) - 1
             selected_indices.append(first_idx)
             if last_idx != first_idx:
                 selected_indices.append(last_idx)
             self.seg.breakpoints = sorted([self.dates[idx] for idx in selected_indices])
        
        if selected_indices:
            for idx in selected_indices:
                if idx not in self.chips.selected_indices:
                    self.chips.selected_indices.append(idx)
                    self.chips.images[idx].layout.border = '2px solid red'
            
            self.vits.selected_indices = self.chips.selected_indices.copy()
            self.vits._update_selected_colors()
            self.vits._update_vlines_and_fit()

        chip_size = '150px'
        for image in self.chips.images:
            if image.layout is None:
                image.layout = ipw.Layout()
            image.layout.width = chip_size
            image.layout.height = chip_size
            image.layout.object_fit = 'contain' 
            image.layout.margin = '2px'  

        self.chips.widget.layout.align_items = 'flex-start'
        self.chips.widget.layout.align_content = 'flex-start'
        self.chips.widget.layout.overflow = 'auto'

        self.draw_webmap(geom=geom, res=self.res, crs=crs)
        # interface
        self.sidebar = ipw.VBox([
                                 self.chip_size_input, # Add config to sidebar
                                 self.navigation_menu,
                                 self.seg.segment_widgets,
                                 ipw.Box(layout=ipw.Layout(height='20px')),
                                 self.save_button],
                                 layout=self.sidebar_layout)
        self.sidebar_with_logo = ipw.VBox(
                                    [self.sidebar, self.logo],
                                    layout=ipw.Layout(height='100%',
                                                      width='100%',
                                                      align_items='stretch',
                                                      overflow='hidden')
                                )

        self.top_splitter = ResizableSplitter(
            left_widget=self.vits.plot,
            right_widget=self.webmap,
            orientation='horizontal',
            initial_left_size='70%',
            min_left_size='20%',
            min_right_size='20%'
        )
        self.top_splitter.layout.height = '45vh'
        self.top_splitter.layout.overflow = 'visible'

        self.bottom_splitter = ResizableSplitter(
            left_widget=self.chips.widget,
            right_widget=self.sidebar_with_logo,
            orientation='horizontal',
            initial_left_size='70%',
            min_left_size='20%',
            min_right_size='20%'
        )
        self.bottom_splitter.layout.height = '55vh'
        self.bottom_splitter.layout.overflow = 'auto'

        self.interface = ResizableSplitter(
            left_widget=self.top_splitter,
            right_widget=self.bottom_splitter,
            orientation='vertical',
            initial_left_size='400px',
            min_left_size='200px',
            min_right_size='200px'
        )
        self.interface.layout.height = '96vh'
        self.interface.layout.overflow = 'visible'
        # Connections between elements
        self.chips.observe(self._on_chip_hover, names=['highlight'])
        self.chips.observe(self._on_chip_click, names=['breakpoints'])
        self.navigation_menu.observe(self._on_idx_change, names=['value'])

    def _on_idx_change(self, change):
        self.current_idx = change['new']

    def _handle_selection_change(self, selected_indices):
        if self.chips.selected_indices != selected_indices:
             self.chips.sync_selected_from_vits(selected_indices)
        if self.vits.selected_indices != selected_indices:
             self.vits.sync_selected_from_chips(selected_indices)
        
        selected_dates = sorted([self.dates[idx] for idx in selected_indices])
        
        if list(self.seg.breakpoints) != selected_dates:
            self.seg.breakpoints = selected_dates
        else:
             pass

    @observe('current_idx')
    def update_interface(self, change):
        """Current idx just changed, new data need to be loaded and the displayed
        elements updated accordingly
        """
        try:
            self.fid, dates, images, values, geom, crs = self.loader[change['new']]
            self.dates = dates 
            
            self.seg = Segmentation.from_db_or_datelist(
                feature_id=self.fid,
                conn=self.conn,
                dates=self.dates,
                labels=self.labels)
            
            # Optimize: update data instead of recreating widgets
            self.chips.update_data(self.dates, images, self.seg.breakpoints)
            self.vits.update_data(self.dates, values, self.seg.breakpoints)

            self.vits.on_select_change = self._handle_selection_change
            self.chips.on_select_change = self._handle_selection_change

            selected_indices = []
            for bp in self.seg.breakpoints:
                indices = np.where(self.dates == bp)[0]
                if len(indices) > 0:
                    selected_indices.append(indices[0])
            
            if not selected_indices and len(self.dates) > 0:
                first_idx = 0
                last_idx = len(self.dates) - 1
                selected_indices.append(first_idx)
                if last_idx != first_idx:
                    selected_indices.append(last_idx)
                self.seg.breakpoints = sorted([self.dates[idx] for idx in selected_indices])
            
            if selected_indices:
               
                self.chips.selected_indices = [] 
                for idx in selected_indices:
                    self.chips.selected_indices.append(idx)
                    self.chips.images[idx].layout.border = '2px solid red'
                
                self.vits.selected_indices = self.chips.selected_indices.copy()
                self.vits._update_selected_colors()
                self.vits._update_vlines_and_fit()

            # Update sidebar
            sidebar = list(self.sidebar.children)
            # Index needs adjustment because we inserted chip_size_input at 0
            # [chip_size_input, navigation_menu, segment_widgets, spacer, save_button]
            sidebar[2] = self.seg.segment_widgets 
            self.sidebar.children = tuple(sidebar)
            
            # Update webmap
            self.update_webmap(geom=geom,
                            res=self.res,
                            crs=crs)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error updating interface: {e}")

    def _on_chip_hover(self, change):
        self.vits.update_highlighted_point(change['new'])

    def _on_chip_click(self, change):
        self.vits.breakpoints = copy.deepcopy(change['new'])
        self.seg.breakpoints = copy.deepcopy(change['new'])

    def display(self):
        js_parts = []
        def collect_js_code(widget):
            if isinstance(widget, ResizableSplitter):
                if hasattr(widget, '_js_code'):
                    js_parts.append(widget._js_code)
            if hasattr(widget, 'children'):
                for child in widget.children:
                    collect_js_code(child)
        
        collect_js_code(self.interface)

        if js_parts:
            all_js = '\n'.join(js_parts)
            
            try:
                from IPython.display import Javascript
                from IPython import get_ipython
                ipython = get_ipython()
  
                if ipython is not None:
                    display(Javascript(all_js))
            except Exception as e:
                pass  
        
        return self.interface

    def load_sample(self, idx):
        # Get 6 element tuple from loader
        # Check if sample already exist in the database and build breakpoints accordingly
        pass

    def _on_chip_size_change(self, change):
        """Handle chip size change"""
        new_size = change['new']

        min_required = int(6.1 * self.sample_box_size)
        
        min_limit = max(50, min_required)
        
        if new_size < min_limit:
            new_size = min_limit
            if self.chip_size_input.value != new_size:
                self.chip_size_input.value = new_size
                return # set value will trigger this again
        elif new_size > 1000:
            new_size = 1000
            if self.chip_size_input.value != new_size:
                self.chip_size_input.value = new_size
                return

        self.chips.chip_size = new_size
        # Force update layout, preserving selected indices
        self.chips.update_data(self.chips.dates, self.chips.images, self.chips.breakpoints, 
                               selected_indices=self.chips.selected_indices)

    def draw_webmap(self, geom, res, crs):
        # Creates geometries for sample and buffer, add them to the map and center
        current_shape = shape(geom)
        
        # Calculate geospatial sizes based on chip pixel sizes
        # sample_box_size (pixels) -> geo size
        # window_size (geo) / chip_size (pixels) = geo_per_pixel
        
        try:
            chip_pixels = int(str(self.chip_size_input.value).replace('px', ''))
        except:
            chip_pixels = 150
            
        if self.window_size and chip_pixels > 0:
            geo_per_pixel = self.window_size / chip_pixels
            # sample box width (geo)
            sample_geo_width = self.sample_box_size * geo_per_pixel
            # buffer box width (geo) = 3 * sample box width
            buffer_geo_width = 3 * sample_geo_width
            
            sample_radius = sample_geo_width / 2
            buffer_radius = buffer_geo_width / 2
        else:
            # Fallback if window_size missing (should not happen per requirements)
            sample_radius = res / 2
            buffer_radius = res
        
        if isinstance(current_shape, Point):
            sample_shape = current_shape.buffer(sample_radius, cap_style=3)
        else:
            sample_shape = current_shape
            
        if isinstance(current_shape, Point):
            buffer_shape = current_shape.buffer(buffer_radius, cap_style=3)
        else:
            buffer_shape = sample_shape.buffer(sample_radius, cap_style=3) 

        # Transform geometries
        sample_geom = warp.transform_geom(src_crs=crs,
                                          dst_crs=CRS.from_epsg(4326),
                                          geom=mapping(sample_shape))
        buffer_geom = warp.transform_geom(src_crs=crs,
                                          dst_crs=CRS.from_epsg(4326),
                                          geom=mapping(buffer_shape))
                                          
        centroid = shape(sample_geom).centroid
        
        # Helper to update or create layer
        def update_layer(layer_attr, data, style):
            layer = getattr(self, layer_attr)
            if layer is not None and layer in self.webmap.layers:
                layer.data = data
            else:
                if layer is not None:
                    try:
                        self.webmap.remove_layer(layer)
                    except Exception:
                        pass
                
                new_layer = GeoJSON(data=data, style=style)
                self.webmap.add_layer(new_layer)
                setattr(self, layer_attr, new_layer)

        # Update layers
        # Buffer first (bottom)
        update_layer('buffer_layer', buffer_geom, 
                     {'opacity': 1, 'fillOpacity': 0, 'weight': 1, 'color': 'yellow'})
        if self.buffer_layer: self.buffer_layer.name = 'Buffer'
        
        # Sample on top
        update_layer('sample_layer', sample_geom, 
                     {'opacity': 1, 'fillOpacity': 0, 'weight': 2, 'color': 'magenta'})
        if self.sample_layer: self.sample_layer.name = 'Sample'
            
        self.webmap.center = [centroid.y, centroid.x]
        self.webmap.zoom = 17

    def update_webmap(self, geom, res, crs):
        self.draw_webmap(geom, res, crs)

    def get_fids(self):
        """Get two mutually exclusive lists of feature ids
        First list is the not yet interpreted
        Second list is the already interpreted
        """
        fids_loader = self.loader.fids
        fids_db = Segmentation.get_fids_db(conn=self.conn)
        # Split fids_loader depending on whether it is present in db or not
        present_in_db = []
        not_present_in_db = []
        for idx, feature_id in fids_loader:
            if feature_id in fids_db:
                present_in_db.append((idx, feature_id))
            else:
                not_present_in_db.append((idx, feature_id))
        return present_in_db, not_present_in_db

    def create_interactive_list(self, samples, color):
        """Create lists of samples buttons

        Agrs:
            samples (list): List of (idx, feature_id) tuples
        """
        buttons = [self.create_button(idx, feature_id, color) 
                        for idx, feature_id in samples]
        return ipw.VBox(buttons, layout=ipw.Layout(align_items='center', width='100%'))

    def on_sample_click(self, button):
        self.current_idx = button.idx
        self.refresh_button_styles()

    def create_button(self, idx, feature_id, color):
        """Create a button related to a sample"""
        is_active = (idx == self.current_idx)
        if is_active:
            border_style = '3px solid #FFD700'
            bg_color = 'orange'
        else:
            border_style = 'none'
            bg_color = color

        # outline_style = '2px solid white' if idx == self.current_idx else 'none'
        button = ipw.Button(description=f"Sample {feature_id}",
                            layout=ipw.Layout(width='95%',
                                              flex='0 0 auto',
                                              border=border_style,
                                              align_self='center'))
        button.style.button_color = bg_color
        button.idx = idx
        button.original_color = color
        button.on_click(self.on_sample_click)
        return button

    def update_lists(self):
        """Update lists of samples
        """
        self.present_in_db, self.not_present_in_db = self.get_fids()
        self.interpreted_list.children = [self.create_button(idx,
                                                             feature_id,
                                                             'lightcoral')
                                          for idx, feature_id in self.present_in_db]
        self.not_interpreted_list.children = [self.create_button(idx,
                                                                 feature_id,
                                                                 'darkgreen')
                                              for idx, feature_id in self.not_present_in_db]

    def save_to_db(self, button):
        """Save current segmentation to database"""
        self.seg.to_db(self.fid)
        self.update_lists()

    def refresh_button_styles(self):
        all_buttons = list(self.interpreted_list.children) + \
                      list(self.not_interpreted_list.children)
        
        for btn in all_buttons:
            if btn.idx == self.current_idx:
                btn.style.button_color = 'orange' 
                btn.layout.border = '3px solid #FFD700'
            else:
                if hasattr(btn, 'original_color'):
                    btn.style.button_color = btn.original_color
                else:
                    btn.style.button_color = 'lightcoral' if btn in self.interpreted_list.children else 'darkgreen'
                
                btn.layout.border = 'none'