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

    # Your existing methods here

    def display_widgets(self):
        """Display the widgets for segment management."""
        display(self.segment_widgets)

    def __str__(self):
        message = 'Temporal segmentation with {n} breakpoints and {nn} segments'.format(n=len(self.breakpoints), nn=len(self.segments))
        return message

if __name__ == "__main__":
    import doctest
    doctest.testmod()
