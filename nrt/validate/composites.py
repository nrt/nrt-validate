from abc import ABC, abstractmethod
import numpy as np


class BaseComposite(ABC):
    """Abstract base class for all color compositors
    """
    @staticmethod
    def stretch(arr, blim=[20,2000], glim=[50,2000], rlim=[20,2000]):
        """
        Apply color stretching and [0,1] clipping to a 3 bands image.

        Args:
            arr (np.ndarray): 3D array; bands as last dimension in RGB order.
            blim, glim, rlim (list): min and max values between which to stretch the individual bands.

        Returns:
            np.ndarray: Stretched and clipped array.
        """
        bottom = np.array([[[rlim[0], glim[0], blim[0]]]])
        top = np.array([[[rlim[1], glim[1], blim[1]]]])
        arr_stretched = (arr - bottom) / (top - bottom)
        return np.clip(arr_stretched, 0.0, 1.0)

    @abstractmethod
    def __call__(self, ds):
        """Transform a given xarray Dataset into a color composite.

        Subclasses must implement this method to create a color composite.

        Args:
            ds (xarray.Dataset): Dataset from which to create the composite.

        Returns:
            The color composite as specified by the subclass.
        """
        pass


class SimpleComposite(BaseComposite):
    def __init__(self, b='B02_20', g='B03_20', r='B04_20',
                 blim=[20, 2000], glim=[50, 2000], rlim=[20, 2000]):
        """Initialize the SimpleComposite with specific bands and limits.

        Args:
            b, g, r (str): Band names for blue, green, and red components.
            blim, glim, rlim (list): Stretching limits for blue, green, and red bands.
        """
        self.b = b
        self.g = g
        self.r = r
        self.blim = blim
        self.glim = glim
        self.rlim = rlim

    def __call__(self, ds):
        """Create a stretched color composite from a multivariate xarray Dataset.

        Works only for a Dataset with a single temporal slice (e.g. only x and y
        coordinate valiables)

        Args:
            ds (xarray.Dataset): Dataset from which to create the composite.

        Returns:
            np.ndarray: A 3D numpy array representing the color composite.
        """
        # Extract the specified bands as numpy arrays
        rgb = np.stack([ds[self.r].values, ds[self.g].values, ds[self.b].values],
                       axis=-1)
        # Apply stretching
        rgb_stretched = self.stretch(rgb, self.blim, self.glim, self.rlim)
        return rgb_stretched

