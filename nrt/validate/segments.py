"""
Module with data structures to handle temporal segmentation and interface with
sqlite database
"""
import bisect
import copy
import datetime
import random
import sqlite3
from typing import List

import numpy as np
from traitlets import HasTraits, observe, List as TraitletsList


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
        >>> s.to_db(conn, 6)
        >>> cur = conn.cursor()
        >>> _ = cur.execute("SELECT id, feature_id, begin, end, label FROM segments WHERE id = ?", (1,))
        >>> row = cur.fetchone()
        >>> print(row)
        (1, 6, 18262, 18263, 'forest dieback')
    """
    def __init__(self, begin: np.datetime64, end: np.datetime64, label=None):
        self.begin = begin
        self.end = end
        self.label = label

    @property
    def breakpoints(self):
        return [self.begin, self.end]

    @classmethod
    def from_db(cls, idx, conn):
        """
        Create a Segment instance from the database using the segment ID.

        Args:
            idx (int): The ID of the segment in the database.
            conn: sqlite database connection

        Returns:
            Segment: An instance of the Segment class.
        """
        cur = conn.cursor()
        cur.execute("SELECT id, begin, end, label FROM segments WHERE id = ?", (idx,))
        row = cur.fetchone()
        if row:
            begin = np.datetime64('1970-01-01') + np.timedelta64(row[1], 'D')
            end = np.datetime64('1970-01-01') + np.timedelta64(row[2], 'D')
            return cls(begin, end, row[3])
        else:
            return None

    def to_db(self, conn, feature_id):
        """
        Save the Segment instance to the database.

        Args:
            db_path (str): Path to the SQLite database.
            feature_id (int): The feature ID to associate with the segment.
        """
        cur = conn.cursor()
        cur.execute('''CREATE TABLE IF NOT EXISTS segments
                       (id INTEGER PRIMARY KEY, begin INTEGER, end INTEGER, label TEXT, feature_id INTEGER)''')
        begin = self.begin.astype('datetime64[D]').astype(int).item()
        end = self.end.astype('datetime64[D]').astype(int).item()
        cur.execute("INSERT INTO segments (begin, end, label, feature_id) VALUES (?, ?, ?, ?)",
                    (begin, end, self.label, feature_id))
        conn.commit()
        cur.close()

    def __str__(self):
        label = self.label if self.label else 'undefined'
        s = "Temporal segment\nbegin: {begin}\nend: {end}\nlabel: {label}".format(begin=self.begin,
                                                                                  end=self.end,
                                                                                  label=label)
        return s


class Segmentation(HasTraits):
    """Container for segmentation with observable traits

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
        >>> # Open a sqlite3 connection
        >>> conn = sqlite3.connect(':memory:')

        >>> seg = Segmentation.from_datelist(random_dates, conn)
        >>> print(seg)
        Temporal segmentation with 2 breakpoints and 1 segments
        >>> seg.add_breakpoint(np.datetime64('2006-11-21'))
        >>> print(seg)
        Temporal segmentation with 3 breakpoints and 2 segments
        >>> seg.remove_breakpoint(np.datetime64('2006-11-21'))
        >>> print(seg)
        Temporal segmentation with 2 breakpoints and 1 segments
        >>> # Write results to a sqlite database
        >>> seg.add_breakpoint(np.datetime64('2006-11-21'))
        >>> seg.to_db(12)
        >>> cur = conn.cursor()
        >>> _ = cur.execute('SELECT * FROM segments WHERE feature_id = 12')
        >>> rows = cur.fetchall()
        >>> print(rows)
        [(1, 12, 12788, 13473, None), (2, 12, 13473, 14244, None)]
    """
    segments: List[Segment] = TraitletsList()
    breakpoints: List[np.datetime64] = TraitletsList()
    conn: sqlite3.Connection

    def __init__(self, conn, breakpoints=None, segments=None):
        super(Segmentation, self).__init__()
        self.breakpoints = breakpoints if breakpoints else []
        self.segments = segments if segments else []
        self.conn = conn

    @classmethod
    def from_datelist(cls, dates, conn):
        """Create a Segmentation instance from a list of dates.

        Assigns a single temporal segment spanning the entire time-series

        Args:
            dates (list of np.datetime64): List of dates.
            db_path (str): Path to the SQLite database.

        Returns:
            Segmentation: An instance of the Segmentation class.
        """
        begin = min(dates)
        end = max(dates)
        instance = cls(conn=conn)
        instance.breakpoints = [begin, end]
        return instance

    @classmethod
    def from_db(cls, feature_id, conn):
        """
        Create a Segmentation instance from the database using the feature ID.

        Args:
            feature_id (int): The feature ID.
            db_path (str): Path to the SQLite database.

        Returns:
            Segmentation: An instance of the Segmentation class.
        """
        segments_idx = cls.get_db_idx(feature_id, conn)
        segments = [Segment.from_db(idx, conn=conn) for idx in segments_idx]
        breakpoints = cls.compute_breakpoints(segments)
        instance = cls(conn=conn, breakpoints=breakpoints, segments=segments)
        return instance

    @classmethod
    def from_db_or_datelist(cls, feature_id, conn, dates):
        """
        Create a Segmentation instance either from the database or from a list of dates.

        Args:
            feature_id (int): The feature ID.
            conn (sqlite3.Connection): Connection to a sqlite database
            dates (list of np.datetime64): List of dates.

        Returns:
            Segmentation: An instance of the Segmentation class.
        """
        if cls.exists(feature_id, conn):
            return cls.from_db(feature_id, conn)
        else:
            return cls.from_datelist(dates, conn)

    @staticmethod
    def exists(feature_id, conn):
        """Check if a feature ID exists in the segments table.

        Args:
            feature_id (int): The feature ID.
            db_path (str): Path to the SQLite database.

        Returns:
            bool: True if the feature ID exists, False otherwise.
        """
        try:
            cur = conn.cursor()
            cur.execute('SELECT 1 FROM segments WHERE feature_id = ?', (feature_id,))
            result = cur.fetchone()
        except sqlite3.OperationalError as e:
            if "no such table" in str(e):
                print("Error: The table does not exist in the database")
            else:
                print(f"OperationalError: {e}")
            return False
        except sqlite3.Error as e:
            print(f"Database error: {e}")
            return False
        return result is not None

    @staticmethod
    def get_db_idx(feature_id, conn):
        """
        Retrieve the IDs of segments with a given feature ID.

        Args:
            feature_id (int): The feature ID.
            conn (sqlite3.Connection): Database connection

        Returns:
            list of int: List of segment IDs.
        """
        cur = conn.cursor()
        cur.execute('SELECT id FROM segments WHERE feature_id = ?', (feature_id,))
        rows = cur.fetchall()
        cur.close()
        idx = [row[0] for row in rows]
        return idx

    def to_db(self, feature_id):
        """Save the Segmentation instance to the database.

        Args:
            feature_id (int): The feature ID.
        """
        try:
            cur = self.conn.cursor()
            cur.execute('DELETE FROM segments WHERE feature_id = ?', (feature_id,))
            conn.commit()
        except sqlite3.OperationalError as e:
            if "no such table" in str(e):
                print("Error: The table does not exist in the database")
            else:
                print(f"OperationalError: {e}")
        except sqlite3.Error as e:
            print(f"Database error: {e}")
        [s.to_db(self.conn, feature_id) for s in self.segments]

    def add_breakpoint(self, date):
        """
        Add a breakpoint date to the segmentation.

        Args:
            date (np.datetime64): The date to add as a breakpoint.
        """
        bp = copy.deepcopy(self.breakpoints)
        bisect.insort(bp, date)
        self.breakpoints = bp

    def remove_breakpoint(self, date):
        """
        Remove a breakpoint date from the segmentation.

        Args:
            date (np.datetime64): The date to remove from breakpoints.

        Raises:
            ValueError: If the date is not a valid breakpoint.
        """
        bp = copy.deepcopy(self.breakpoints)
        if date in bp:
            bp.remove(date)
        else:
            raise ValueError('Not a valid breakpoint date')
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

    @staticmethod
    def compute_breakpoints(segments):
        """
        Compute breakpoints given a list of segments.

        Args:
            segments (list of Segment): List of segments.

        Returns:
            list of np.datetime64: Sorted list of unique breakpoints.
        """
        bp = []
        for seg in segments:
            bp += seg.breakpoints
        return sorted(set(bp))

    @observe('breakpoints')
    def _observe_breakpoints(self, change):
        """
        Observer for the breakpoints trait.

        Args:
            change: The change event.
        """
        self.segments = [Segment(begin, end) for begin, end in
                         zip(self.breakpoints[:-1], self.breakpoints[1:])]

    def __str__(self):
        message = 'Temporal segmentation with {n} breakpoints and {nn} segments'.format(n=len(self.breakpoints), nn=len(self.segments))
        return message


if __name__ == "__main__":
    import doctest
    doctest.testmod()
