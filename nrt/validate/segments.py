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
    def __init__(self, begin: np.datetime64, end: np.datetime64, label=None):
        self.begin = begin
        self.end = end
        self.label = label

    @property
    def breakpoints(self):
        return [self.begin, self.end]

    @classmethod
    def from_db(cls, idx, db_path):
        """
        Create a Segment instance from the database using the segment ID.

        Args:
            idx (int): The ID of the segment in the database.
            db_path (str): Path to the SQLite database.

        Returns:
            Segment: An instance of the Segment class.
        """
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT id, begin, end, label FROM segments WHERE id = ?", (idx,))
        row = cur.fetchone()
        conn.close()
        if row:
            begin = np.datetime64('1970-01-01') + np.timedelta64(row[1], 'D')
            end = np.datetime64('1970-01-01') + np.timedelta64(row[2], 'D')
            return cls(begin, end, row[3])
        else:
            return None

    def to_db(self, db_path, feature_id):
        """
        Save the Segment instance to the database.

        Args:
            db_path (str): Path to the SQLite database.
            feature_id (int): The feature ID to associate with the segment.
        """
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute('''CREATE TABLE IF NOT EXISTS segments
                       (id INTEGER PRIMARY KEY, begin INTEGER, end INTEGER, label TEXT, feature_id INTEGER)''')
        begin = self.begin.astype('datetime64[D]').astype(int).item()
        end = self.end.astype('datetime64[D]').astype(int).item()
        cur.execute("INSERT INTO segments (begin, end, label, feature_id) VALUES (?, ?, ?, ?)",
                    (begin, end, self.label, feature_id))
        conn.commit()
        conn.close()

    def __str__(self):
        label = self.label if self.label else 'undefined'
        s = "Temporal segment\nbegin: {begin}\nend: {end}\nlabel: {label}".format(begin=self.begin,
                                                                                  end=self.end,
                                                                                  label=label)
        return s


class Segmentation(HasTraits):
    """
    Examples:
        >>> import os
        >>> import tempfile

        >>> import numpy as np

        >>> from nrt.validate.segments import Segment, Segmentation

        >>> # Define the start and end dates
        >>> start_date = np.datetime64('2021-01-01')
        >>> end_date = np.datetime64('2024-01-01')
        >>> total_days = (end_date - start_date).astype('timedelta64[D]').astype(int)
        >>> # Generate 20 evenly spaced days within the range
        >>> days = np.linspace(0, total_days, num=20, dtype=int)
        >>> dates = start_date + days

        >>> # db
        >>> temp_dir = tempfile.gettempdir()
        >>> db_path = os.path.join(temp_dir, 'nr_test_database.sqlite')
        >>> try:
        ...     os.remove(db_path)
        >>> except OSError:
        ...     pass

        >>> s1 = Segmentation.from_db_or_datelist(feature_id=12, db_path=db_path, dates=dates)
        >>> s1.add_breakpoints(dates[5])
        >>> s1.to_db(feature_id=12)
        >>> s2 = Segmentation.from_db_or_datelist(feature_id=12, db_path=db_path, dates=dates)
        >>> assert s2.breakpoints == s1.breakpoints
    """
    segments: List[Segment] = TraitletsList()
    breakpoints: List[np.datetime64] = TraitletsList()
    db_path: str

    def __init__(self, db_path, breakpoints=None, segments=None):
        super(Segmentation, self).__init__()
        self.breakpoints = breakpoints if breakpoints else []
        self.segments = segments if segments else []
        self.db_path = db_path

    @classmethod
    def from_datelist(cls, dates, db_path):
        """
        Create a Segmentation instance from a list of dates.

        Args:
            dates (list of np.datetime64): List of dates.
            db_path (str): Path to the SQLite database.

        Returns:
            Segmentation: An instance of the Segmentation class.
        """
        begin = min(dates)
        end = max(dates)
        instance = cls(db_path=db_path)
        instance.breakpoints = [begin, end]
        return instance

    @classmethod
    def from_db(cls, feature_id, db_path):
        """
        Create a Segmentation instance from the database using the feature ID.

        Args:
            feature_id (int): The feature ID.
            db_path (str): Path to the SQLite database.

        Returns:
            Segmentation: An instance of the Segmentation class.
        """
        segments_idx = cls.get_db_idx(feature_id, db_path)
        segments = [Segment.from_db(idx, db_path=db_path) for idx in segments_idx]
        breakpoints = cls.compute_breakpoints(segments)
        instance = cls(db_path=db_path, breakpoints=breakpoints, segments=segments)
        return instance

    @classmethod
    def from_db_or_datelist(cls, feature_id, db_path, dates):
        """
        Create a Segmentation instance either from the database or from a list of dates.

        Args:
            feature_id (int): The feature ID.
            db_path (str): Path to the SQLite database.
            dates (list of np.datetime64): List of dates.

        Returns:
            Segmentation: An instance of the Segmentation class.
        """
        if cls.exists(feature_id, db_path):
            return cls.from_db(feature_id, db_path)
        else:
            return cls.from_datelist(dates, db_path)

    @staticmethod
    def exists(feature_id, db_path):
        """Check if a feature ID exists in the segments table.

        Args:
            feature_id (int): The feature ID.
            db_path (str): Path to the SQLite database.

        Returns:
            bool: True if the feature ID exists, False otherwise.
        """
        try:
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute('SELECT 1 FROM segments WHERE feature_id = ?', (feature_id,))
            result = cur.fetchone()
        except sqlite3.OperationalError as e:
            if "no such table" in str(e):
                print(f"Error: The table does not exist in the database {db_path}.")
            else:
                print(f"OperationalError: {e}")
            return False
        except sqlite3.Error as e:
            print(f"Database error: {e}")
            return False
        finally:
            if 'conn' in locals():
                conn.close()
        return result is not None

    @staticmethod
    def get_db_idx(feature_id, db_path):
        """
        Retrieve the IDs of segments with a given feature ID.

        Args:
            feature_id (int): The feature ID.
            db_path (str): Path to the SQLite database.

        Returns:
            list of int: List of segment IDs.
        """
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute('SELECT id FROM segments WHERE feature_id = ?', (feature_id,))
        rows = cur.fetchall()
        idx = [row[0] for row in rows]
        conn.close()
        return idx

    def to_db(self, feature_id):
        """Save the Segmentation instance to the database.

        Args:
            feature_id (int): The feature ID.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            cur.execute('DELETE FROM segments WHERE feature_id = ?', (feature_id,))
            conn.commit()
        except sqlite3.OperationalError as e:
            if "no such table" in str(e):
                print(f"Error: The table does not exist in the database {self.db_path}.")
            else:
                print(f"OperationalError: {e}")
        except sqlite3.Error as e:
            print(f"Database error: {e}")
        finally:
            if 'conn' in locals():
                conn.close()
        [s.to_db(self.db_path, feature_id) for s in self.segments]

    def add_breakpoints(self, date):
        """
        Add a breakpoint date to the segmentation.

        Args:
            date (np.datetime64): The date to add as a breakpoint.
        """
        bp = copy.deepcopy(self.breakpoints)
        bisect.insort(bp, date)
        self.breakpoints = bp

    def remove_breakpoints(self, date):
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

