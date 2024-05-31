"""Contains elements and classes useful to the response design of an
accuracy assessment estimate
"""
import bisect
import datetime, random
import copy
import sqlite3

import numpy as np
from traitlets import HasTraits, List, observe
import ipywidgets as ipw
from bqplot import Scatter, Lines, LinearScale, Axis, Figure

from nrt.validate import indices, composites, utils


class Segment(object):
    """Represents a temporal segment with a beginning and an end, optionally labeled.

    Args:
        begin (numpy.datetime64): The beginning of the segment.
        end (numpy.datetime64): The end of the segment.
        label (str): An optional label for the segment

    Examples:
        >>> import numpy as np
        >>> import sqlite3
        >>> s = Segment(np.datetime64('2020-01-01'),
        ...             np.datetime64('2020-01-02'),
        ...             'forest dieback')
        >>> print(s)
        Temporal segment
        begin: 2020-01-01
        end: 2020-01-02
        label: forest dieback

        >>> s.breakpoints
        [numpy.datetime64('2020-01-01'), numpy.datetime64('2020-01-02')]

        >>> conn = sqlite3.connect(':memory:')
        >>> s.to_db(6, conn)
        >>> cur = conn.cursor()
        >>> _ = cur.execute("SELECT id, feature_id, begin, end, label FROM segments WHERE id = ?", (1,))
        >>> row = cur.fetchone()
        >>> print(row)
        (1, 6, 18262, 18263, 'forest dieback')
    """
    def __init__(self, begin, end, label=None):
        # Type checks for begin and end
        if not isinstance(begin, np.datetime64) or not isinstance(end, np.datetime64):
            raise TypeError("Both 'begin' and 'end' must be of type numpy.datetime64.")
        # Ensure that begin is before end
        if not begin < end:
            raise ValueError("The begin date/time must be before the end date/time.")
        self.begin = begin
        self.end = end
        self.label = label

    @property
    def breakpoints(self):
        return [self.begin, self.end]

    @classmethod
    def from_db(cls, idx, conn):
        cur = conn.cursor()
        # Query the database for the segment with the given id
        cur.execute("SELECT id, begin, end, label FROM segments WHERE id = ?", (idx,))
        row = cur.fetchone()
        # Close the cursor
        cur.close()
        if row:
            return cls(np.datetime64(row[1], 'D'),
                       np.datetime64(row[2], 'D'),
                       row[3])
        else:
            return None

    def to_db(self, feature_id, conn):
        cur = conn.cursor()
        # Create the segments table if it does not exist
        cur.execute('''CREATE TABLE IF NOT EXISTS segments
                       (id INTEGER PRIMARY KEY, feature_id INTEGER, begin INTEGER, end INTEGER, label TEXT)''')
        # Insert the segment's data into the segments table
        cur.execute("INSERT INTO segments (feature_id, begin, end, label) VALUES (?, ?, ?, ?)",
                    (feature_id,
                     int(self.begin.astype('datetime64[D]').astype('int')),
                     int(self.end.astype('datetime64[D]').astype('int')),
                     self.label))
        # Commit the changes and close the cursor
        conn.commit()
        cur.close()

    def widget(self, labels=['forest', 'dieback', 'non-forest']):
        """Create a widget with a label and dropdown for segment label selection."""
        def on_change(change):
            if change['type'] == 'change' and change['name'] == 'value':
                self.label = change['new']

        segment_info = ipw.Label(value=f"Segment from {self.begin} to {self.end}")
        dropdown = ipw.Dropdown(options=labels, description='Label:', value=self.label)
        dropdown.observe(on_change)

        # Use VBox to vertically stack the label and dropdown
        widget_box = ipw.VBox([segment_info, dropdown])
        return widget_box

    def __str__(self):
        label = self.label if self.label else 'undefined'
        s = "Temporal segment\nbegin: {begin}\nend: {end}\nlabel: {label}".format(begin=self.begin,
                                                                                  end=self.end,
                                                                                  label=label)
        return s


class Segmentation(HasTraits):
    """Container for segmentation with callbacks and event handlers

    Args:
        breakpoints (list): A list of numpy.datetime64 corresponding to breakpoints
            around temporal segments.
        segments (list): A list of ``Segment``s. The corresponding attribute is
            dynamically computed and updated from the breakpoints attribute

    Examples:
        >>> import sqlite3
        >>> import numpy as np

        >>> # Generate 50 random dates between 2005 and 2008
        >>> start_date = np.datetime64('2005-01-01')
        >>> end_date = np.datetime64('2008-12-31')
        >>> num_days = (end_date - start_date).astype(int)
        >>> random_days = np.sort(np.random.randint(0, num_days, 50))
        >>> random_dates = start_date + random_days
        >>> random_dates = np.append(random_dates, end_date)

        >>> seg = Segmentation.from_datelist(random_dates)
        >>> print(seg)
        Temporal segmentation with 2 breakpoints and 1 segments
        >>> seg.add_breakpoint(np.datetime64('2006-11-21'))
        >>> print(seg)
        Temporal segmentation with 3 breakpoints and 2 segments
        >>> seg.remove_breakpoint(np.datetime64('2006-11-21'))
        >>> print(seg)
        Temporal segmentation with 2 breakpoints and 1 segments
        >>> # Write results to a sqlite database
        >>> conn = sqlite3.connect(':memory:')
        >>> seg.add_breakpoint(np.datetime64('2006-11-21'))
        >>> seg.to_db(12, conn)
        >>> cur = conn.cursor()
        >>> _ = cur.execute('SELECT * FROM segments WHERE feature_id = 12')
        >>> rows = cur.fetchall()
        >>> print(rows)
        [(1, 12, 12788, 13473, None), (2, 12, 13473, 14244, None)]

    """
    segments = List()
    breakpoints = List()

    def __init__(self, breakpoints=None, segments=None,
                 labels=['Stable forest',
                         'Forest dieback',
                         'Forest recovery',
                         'Non-forest']):
        super(Segmentation, self).__init__()
        if breakpoints is not None:
            self.breakpoints = breakpoints
        if segments is not None:
            self.segments = segments
        self.labels = labels
        self.segment_widgets = ipw.VBox([])
        self._update_segment_widgets()

    @classmethod
    def from_datelist(cls, dates):
        """Instantiates a Segmentation from a datetime64 array

        Assigns a single temporal segment spanning the entire time-series
        """
        begin = min(dates)
        end = max(dates)
        instance = cls()
        instance.breakpoints = [begin, end]
        return instance

    @classmethod
    def from_db(cls, feature_id, conn):
        # Retrieve ids of segments in database
        cur = conn.cursor()
        cur.execute('SELECT id FROM segments WHERE feature_id = ?', (feature_id))
        rows = cur.fetchall()
        cur.close()
        # TODO: Not sure how to best handle that
        if not rows:
            return None
        segments_idx = [row[0] for row in rows]
        segments = [Segment.from_db(idx, conn=conn) for idx in segments_idx]
        breakpoints = self.compute_breakpoints(segments)
        instance = cls(breakpoints=breakpoints, segments=segments)
        return instance

    def to_db(self, feature_id, conn):
        # If there are existing entries for that feature_id, they first need to
        # be deleted
        cur = conn.cursor()
        try:
            cur.execute('DELETE FROM segments WHERE feature_id = ?', (feature_id))
        except sqlite3.OperationalError:
            pass
        cur.close()
        [seg.to_db(feature_id, conn) for seg in self.segments]

    def add_breakpoint(self, date):
        bp = copy.deepcopy(self.breakpoints)
        bisect.insort(bp, date)
        self.breakpoints = bp

    def remove_breakpoint(self, date):
        bp = copy.deepcopy(self.breakpoints)
        if date in bp:
            bp.remove(date)
        else:
            ValueError('Not a valid breakpoint date')
        self.breakpoints = bp

    def add_or_remove_breakpoint(self, date):
        """If the date provided is already a breakpoint, remove it, otherwise add it
        """
        bp = copy.deepcopy(self.breakpoints)
        if date in bp:
            bp.remove(date)
        else:
            bisect.insort(bp, date)
        self.breakpoints = bp

    def update_marks(self, interface):
        """Given an Interface instance, update its vline attribute"""
        pass

    @staticmethod
    def compute_breakpoints(segments):
        """Compute breakpoints given a list of segments"""
        bp = []
        for seg in segments:
            bp += seg.breakpoints
        return sorted(set(bp))

    @observe('breakpoints')
    def _observe_breakpoints(self, change):
        segments = []
        for begin, end in zip(self.breakpoints[:-1], self.breakpoints[1:]):
            segments.append(Segment(begin, end))
        self.segments = segments

    @observe('segments')
    def _observe_segments(self, change):
        self._update_segment_widgets()

    def _update_segment_widgets(self):
        """Update the widgets for all segments."""
        widgets_list = [segment.widget(labels=self.labels) for segment in self.segments]
        self.segment_widgets.children = widgets_list

    def display_widgets(self):
        """Display the widgets for segment management."""
        display(self.segment_widgets)

    def __str__(self):
        message = 'Temporal segmentation with {n} breakpoints and {nn} segments'.format(n=len(self.breakpoints), nn=len(self.segments))
        return message


class ChipsTs(HasTraits):
    """A container of linked chips pannel and bqplot time-series plot
    """
    def __init__(self, dates, values, images, segmentation):
        self.segmentation = segmentation
        self.values = values
        self.dates = dates
        # Create components (ts plot + chips)
        x_sc = LinearScale()
        y_sc = LinearScale()
        # Create axes
        self.x_ax = Axis(label='Time', scale=x_sc,
                         tick_format='%m-%Y', tick_rotate=45)
        self.y_ax = Axis(label='Vegetation Index', scale=y_sc,
                         orientation='vertical', side='left')
        # Create scatter marks
        # TODO: right now univariate, but would be good to allow multiple variables passed as
        # a dict of arrays
        vi_values = Scatter(x=dates, y=values,
                            scales={'x': x_sc, 'y': y_sc})
        # Create a dummy highlighted point out of view
        highlighted_point = Scatter(x=[-1000], y=[-1000], # Dummy point out of view
                                    scales={'x': x_sc, 'y': y_sc},
                                    preserve_domain={'x': True, 'y': True},
                                    colors=['red'])
        # Get the initial vlines from the Segmentation instance
        vlines = [Lines(x=[bp, bp], y=[0, 1],
                        scales={'x': x_sc, 'y': y_sc}, colors=['red'])
                  for bp in segmentation.breakpoints]
        # Create and display the figure
        self.tsfig = Figure(marks=[vi_values, highlighted_point, *vlines],
                            axes=[x_ax, y_ax],
                            title='Sample temporal profile',
                            layout=ipw.Layout(width='100%',
                                              height='400px'))

        # Add event handler to each chip
        for idx, chip in enumerate(chips):
            event = Event(source=chip, watched_events = ['mouseenter', 'mouseleave', 'click'])
            event.on_dom_event(functools.partial(self._handle_chip_event, idx))

        box_layout = ipw.Layout(
            display='flex',
            flex_flow='row wrap',
            align_items='stretch',
            width='100%',
            height='800px',  # Set a fixed height (modify as needed)
            overflow='auto'  # Add scrollability
        )
        box = ipw.Box(children=chips, layout=box_layout)
        self.interface = ipw.VBox([ts_fig, box])

        @classmethod
        def from_cube_and_geom(cls, ds, geom, segmentation,
                               compositor=ColorComposite(),
                               vi_calculator=NDVI(),
                               window_size=500,
                               **kwargs):
            """Instantiate ChipsTs from an xarray Dataset and a geometry

            Geometry and cube/Dataset must share the same coordinate reference system

            Args:
                ds (xarray.Dataset): The Dataset containing the data to display
                geom (dict): A geojson geometry (Point or Polygon) around which
                    Dataset will be cropped and for which index time-series will
                    be extracted
                segmentation (nrt.validation.response.Segmentation): Segmentation instance
                compositor (callable): Callable to transform a temporal slice of the provided
                    Dataset into a 3D numpy array. See ``nrt.validate.composites`` module
                    for examples
                vi_calculator (callable): A callable to process a DataArray containing the
                    desired index from the dataset. See the ``nrt.validate.indices`` module for
                    examples and already implemented simple transforms
                window_size (float): Size of the bounding box used for cropping (created around
                    the centroid of ``geom``). In CRS unit.
                **kwargs: Additional arguments passed to ``nrt.validate.utils.get_chips``
            """
            dates, values = utils.get_ts(ds=ds, geom=geom,
                                         vi_calculator=vi_calculator)
            chips = utils.get_chips(ds=ds, geom=geom, size=window_size,
                                    compositor=compositor, **kwargs)
            instance = cls(dates=dates, values=values,
                           images=chips, segmentation=segmentation)
            return instance

        def toggle_vline(self, vline):
            """Add or remove a vline to the instance's tsplot
            """
            marks_list = list(self.tsfig.marks)  # Convert to list for easier manipulation
            if vline in self.tsfig.marks:
                marks_list.remove(vline)
            else:
                marks_list.append(vline)
            self.tsfig.marks = tuple(marks_list)  # Convert back to tuple

        def _handle_chip_event(self, idx, event):
            """Change the coordinates of the highlighted point to actual date and value when mouse enters chip"""
            value = self.values[idx]
            date = self.dates[idx]
            vline = Lines(x=[date, date], y=[0, 1],
                          scales={'x': x_sc, 'y': y_sc}, colors=['red'])
            if event['type'] == 'mouseenter':
                highlighted_point.x = [date]
                highlighted_point.y = [value]
            if event['type'] == 'mouseleave':
                # Reset dummy location
                highlighted_point.x = [-1000]
                highlighted_point.y = [-1000]
            if event['type'] == 'click':
                self.toggle_vline(vline)
                self.segmentation.add_or_remove_breakpoint(date)
                if chips[idx].layout.border == '2px solid blue':
                    # If it has a border, remove it
                    chips[idx].layout.border = ''
                else:
                    chips[idx].layout.border = '2px solid blue'


if __name__ == "__main__":
    import doctest
    doctest.testmod()
